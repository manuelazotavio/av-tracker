import os
import re
import time
import logging
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
from boxmot import ByteTrack
from pathlib import Path

logger = logging.getLogger(__name__)


class _EdgeFaceXXS(nn.Module):
    """Minimal wrapper matching TimmFRWrapperV2 from otroshi/edgeface."""
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('edgenext_xx_small')
        self.model.reset_classifier(512)

    def forward(self, x):
        return self.model(x)


class PersonIDTracker:
    MATCH_THRESHOLD = 0.55  # cosine similarity to consider a known person
    MERGE_THRESHOLD = 0.55  # threshold for merging during auto-enrollment (same as MATCH)
    ENROLL_FRAMES = 50      # frames to accumulate (~5s) for more stable average embedding
    CONFIRM_FRAMES = 10     # consecutive frames needed to confirm a known-name assignment

    def __init__(self, model_path="od_model/edgeface_xxs.pt", device="cuda"):
        self.device = 'cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu'
        self.tracker = ByteTrack(track_buffer=90)  # ~3s at 30fps before track dies
        net = _EdgeFaceXXS()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        net.load_state_dict(state_dict)
        self.model = net.to(self.device).eval()
        self.known_embeddings = {}   # name -> np.array (512,)
        self.emb_dir = None

        # Auto-enrollment state
        self._track_buffer = {}      # track_id -> list of embeddings
        self._track_to_name = {}     # track_id -> assigned name (confirmed)
        self._person_counter = 0
        self._persist_fail_count = {}  # track_id -> consecutive persist failures
        self._name_confirm = {}      # track_id -> {"name": str, "count": int}
        self._override_last_frame = {}  # track_id -> frame index of last OVERRIDE
        self._track_last_seen = {}   # track_id -> time.time() of last visible frame
        self._perf = {}  # key -> list[float ms]
        self._perf_frame_count = 0
        self._perf_summary_interval = 100  # prints summary every N frames

        # --- Session metrics ---
        self._face_events = []           # structured events for post-session analysis
        self._match_scores_sample = []   # sampled (track_id, best_name, best_score, all_scores)
        self._sample_counter = 0

    def load_known_embeddings(self, emb_dir):
        self.emb_dir = emb_dir
        if not os.path.exists(emb_dir):
            return
        self._emb_files = {}  # name -> filename on disk (for consolidation cleanup)
        _raw_face_embs = {}  # name -> list of np.array (for averaging multiple entries)
        for file in os.listdir(emb_dir):
            if file.endswith(".npy"):
                base = file[:-4]  # remove .npy
                if base.endswith("_auto"):
                    base = base[:-5]  # strip _auto suffix first
                # Strip timestamp (YYYYMMDD_HHMMSS) if present
                parts = base.split('_')
                if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                    name = '_'.join(parts[:-2])
                else:
                    name = base
                # Skip generic names — they exist on disk for validation
                # to rename, but must not compete with real face embeddings.
                if re.match(r'^(Person_\d+|Unknown(_\d+)?)$', name):
                    continue
                emb = np.load(os.path.join(emb_dir, file))
                _raw_face_embs.setdefault(name, []).append(emb)
                self._emb_files[name] = file
        # Average of multiple embeddings per person (different lighting/angle conditions)
        for name, embs in _raw_face_embs.items():
            avg = np.mean(embs, axis=0)
            norm = np.linalg.norm(avg)
            self.known_embeddings[name] = avg / norm if norm > 0 else avg
        # Merge embeddings that are too similar (same person enrolled multiple times)
        self._consolidate_embeddings()
        # Sync person counter so new auto names don't collide
        for name in self.known_embeddings:
            if name.startswith("Person_"):
                try:
                    n = int(name.rsplit("_", 1)[1])
                    self._person_counter = max(self._person_counter, n)
                except (ValueError, IndexError):
                    pass

    def _consolidate_embeddings(self):
        """Merge known embeddings that are too similar (same person enrolled multiple times).

        Safety: only merges when at least one side is a generic name (Person_N).
        Two real names (e.g. Manuela, Heitor) are NEVER merged, even if their
        embeddings are similar — the model may simply not be discriminative enough.
        """
        _generic_re = re.compile(r'^Person_\d+$')
        names = list(self.known_embeddings.keys())
        merged_into = {}  # name -> canonical name it was merged into

        for i, name_a in enumerate(names):
            if name_a in merged_into:
                continue
            for name_b in names[i + 1:]:
                if name_b in merged_into:
                    continue
                # Never merge two real (non-generic) names
                a_is_generic = bool(_generic_re.match(name_a))
                b_is_generic = bool(_generic_re.match(name_b))
                if not a_is_generic and not b_is_generic:
                    continue
                emb_a = self.known_embeddings[name_a]
                emb_b = self.known_embeddings[name_b]
                sim = float(np.dot(emb_a, emb_b))
                if sim >= self.MERGE_THRESHOLD:
                    # Prefer keeping the real name as canonical
                    if b_is_generic and not a_is_generic:
                        canon, dup = name_a, name_b
                    elif a_is_generic and not b_is_generic:
                        canon, dup = name_b, name_a
                    else:
                        canon, dup = name_a, name_b  # both generic
                    avg = (self.known_embeddings[canon] + self.known_embeddings[dup]) / 2
                    avg = avg / (np.linalg.norm(avg) + 1e-8)
                    self.known_embeddings[canon] = avg
                    merged_into[dup] = canon
                    logger.info(f"Consolidated '{dup}' -> '{canon}' (sim={sim:.3f})")
                    print(f"[PersonIDTracker] Consolidated '{dup}' -> '{canon}' (sim={sim:.3f})")
                    # Remove duplicate file and update canonical file on disk
                    if self.emb_dir:
                        dup_file = self._emb_files.get(name_b)
                        if dup_file:
                            dup_path = os.path.join(self.emb_dir, dup_file)
                            if os.path.exists(dup_path):
                                os.remove(dup_path)
                        canon_file = self._emb_files.get(name_a)
                        if canon_file:
                            np.save(os.path.join(self.emb_dir, canon_file), avg)

        for name in merged_into:
            self.known_embeddings.pop(name, None)
            self._emb_files.pop(name, None)

    def _extract_face_embedding(self, face_img):
        face_img = cv2.resize(face_img, (112, 112))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.transpose((2, 0, 1))
        tensor = torch.from_numpy(face_img).float().div(255.0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def _best_match(self, emb, track_id=None):
        scores = []
        for name, known_emb in self.known_embeddings.items():
            score = float(np.dot(emb, known_emb))
            scores.append((name, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scores[0] if scores else ("Unknown", -1.0)
        # Log top matches for each face (every 30 frames to avoid clutter)
        if hasattr(self, '_match_log_counter'):
            self._match_log_counter += 1
        else:
            self._match_log_counter = 0
        if self._match_log_counter % 30 == 0 and scores:
            top_str = " | ".join(f"{n}: {s:.2f}" for n, s in scores[:3])
            tid_str = f"track={track_id}" if track_id is not None else ""
            logger.debug(f"👤 FaceMatch {tid_str}: [{top_str}] thr={self.MATCH_THRESHOLD}")
        # Sample match scores for session metrics (every 30 frames per track)
        self._sample_counter += 1
        if self._sample_counter % 30 == 0 and scores:
            self._match_scores_sample.append({
                "ts": time.time(),
                "track_id": int(track_id) if track_id is not None else -1,
                "best_name": best_name,
                "best_score": round(best_score, 4),
                "top3": [(n, round(s, 4)) for n, s in scores[:3]],
                "matched": best_score >= self.MATCH_THRESHOLD,
            })
        if best_score < self.MATCH_THRESHOLD:
            return "Unknown", best_score
        return best_name, best_score

    def _enroll(self, track_id, embeddings):
        """Average buffered embeddings, save to disk, register in memory."""
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

        # Check if close enough to a known person to merge (bypass MATCH_THRESHOLD, use MERGE_THRESHOLD directly)
        # so different-angle views of the same face don't create duplicates
        merge_name, merge_score = None, -1.0
        for name, known_emb in self.known_embeddings.items():
            score = float(np.dot(avg_emb, known_emb))
            if score > merge_score:
                merge_score = score
                merge_name = name
        # Don't merge if the target name is already confirmed for another active track.
        # Two distinct people cannot share the same identity.
        _merge_blocked = merge_name is not None and any(
            tid != track_id and tname == merge_name
            for tid, tname in self._track_to_name.items()
        )
        if merge_name is not None and merge_score >= self.MERGE_THRESHOLD and not _merge_blocked:
            logger.debug(f"👤 Enroll track={int(track_id)}: MERGED into '{merge_name}' (sim={merge_score:.3f} >= {self.MERGE_THRESHOLD})")
            self._track_to_name[track_id] = merge_name
            return merge_name
        if _merge_blocked:
            logger.debug(f"👤 Enroll track={int(track_id)}: merge BLOCKED ('{merge_name}' already belongs to another track) → new Person_N")

        self._person_counter += 1
        name = f"Person_{self._person_counter}"
        logger.debug(f"👤 Enroll track={int(track_id)}: NEW '{name}' (best_merge='{merge_name}' sim={merge_score:.3f} < {self.MERGE_THRESHOLD})")
        self.known_embeddings[name] = avg_emb
        self._track_to_name[track_id] = name

        # Save to disk so validation can rename later.
        # The face tracker skips generic names when loading (no cross-session competition).
        if self.emb_dir:
            os.makedirs(self.emb_dir, exist_ok=True)
            filename = f"{name}_auto.npy"
            np.save(os.path.join(self.emb_dir, filename), avg_emb)
            if hasattr(self, '_emb_files'):
                self._emb_files[name] = filename
            print(f"[PersonIDTracker] Auto-enrolled: {name} (track {int(track_id)})")
            # Sync to database
            try:
                from src.database import get_db
                db = get_db()
                spk_id = db.add_speaker(name)
                db.save_face_embedding(spk_id, avg_emb, source_file=filename)
            except Exception as e:
                logger.debug(f"DB sync failed for face embedding: {e}")

        self._face_events.append({
            "ts": time.time(), "event": "enroll",
            "track_id": int(track_id), "name": name,
            "merge_target": merge_name, "merge_score": round(merge_score, 4),
        })
        return name

    def rename_person(self, old_name, new_name, only_track_id=None):
        """Renames an identity in memory and on disk when the real name is detected.

        Args:
            only_track_id: if provided, only renames this specific track
                           (avoids renaming family faces that were merged).
        """
        if old_name not in self.known_embeddings:
            return
        if only_track_id is not None:
            # Only renames the requested track; keeps the old_name embedding
            # for the other tracks that still use it.
            other_tracks_use = any(
                tid != only_track_id and tname == old_name
                for tid, tname in self._track_to_name.items()
            )
            if other_tracks_use:
                # Copies the embedding to the new name (does not remove the old one)
                self.known_embeddings[new_name] = self.known_embeddings[old_name].copy()
                self._track_to_name[only_track_id] = new_name
            else:
                # Only track — can rename normally
                emb = self.known_embeddings.pop(old_name)
                self.known_embeddings[new_name] = emb
                self._track_to_name[only_track_id] = new_name
        else:
            emb = self.known_embeddings.pop(old_name)
            self.known_embeddings[new_name] = emb
            # Updates track → name
            for track_id, tname in self._track_to_name.items():
                if tname == old_name:
                    self._track_to_name[track_id] = new_name
        # Renames file on disk — uses timestamped names to accumulate
        # embeddings from different sessions/conditions (diversity improves
        # cross-session matching with small face models like EdgeFace XXS).
        if self.emb_dir:
            from datetime import datetime as _dt
            old_file = self._emb_files.pop(old_name, None)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            new_file = f"{new_name}_{ts}_auto.npy"
            new_path = os.path.join(self.emb_dir, new_file)
            if old_file:
                old_path = os.path.join(self.emb_dir, old_file)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            elif new_name in self.known_embeddings:
                # Generic Person_N was not saved to disk — save now under the real name
                os.makedirs(self.emb_dir, exist_ok=True)
                np.save(new_path, self.known_embeddings[new_name])
            self._emb_files[new_name] = new_file
            # Limit total files per person to prevent bloat (keep newest)
            MAX_FACE_FILES = 8
            all_files = sorted(
                [f for f in os.listdir(self.emb_dir)
                 if f.endswith(".npy") and f.startswith(new_name)],
            )
            if len(all_files) > MAX_FACE_FILES:
                for old_f in all_files[:len(all_files) - MAX_FACE_FILES]:
                    try:
                        os.remove(os.path.join(self.emb_dir, old_f))
                    except OSError:
                        pass
        # Cleanup orphan Person_N files: if no track uses the old generic name anymore,
        # delete the file from disk to prevent accumulation across sessions.
        if self.emb_dir and old_name and re.match(r'^Person_\d+$', old_name):
            still_used = any(tname == old_name for tname in self._track_to_name.values())
            if not still_used:
                for _f in list(os.listdir(self.emb_dir)):
                    _base = os.path.splitext(_f)[0]
                    if _base == old_name or _base.startswith(old_name + "_"):
                        try:
                            os.remove(os.path.join(self.emb_dir, _f))
                            logger.debug(f"👤 Cleaned orphan file: {_f}")
                        except OSError:
                            pass
                self.known_embeddings.pop(old_name, None)

        # Log face FN: voice identified the person but face tracker didn't recognise them
        _is_generic = bool(re.match(r'^(Person_\d+|Unknown)$', old_name or ""))
        _is_real_new = not bool(re.match(r'^(Person_\d+|Unknown)$', new_name or ""))
        if _is_generic and _is_real_new:
            self._face_events.append({
                "ts": time.time(), "event": "face_fn_rename",
                "old_name": old_name, "new_name": new_name,
                "track_id": int(only_track_id) if only_track_id is not None else None,
            })
        print(f"[PersonIDTracker] Renamed '{old_name}' -> '{new_name}'")
        # Re-consolidate after rename to merge duplicates created in the session
        self._consolidate_embeddings()
        # Fix _track_to_name for tracks that pointed to merged entities
        for tid in list(self._track_to_name.keys()):
            if self._track_to_name[tid] not in self.known_embeddings:
                self._track_to_name.pop(tid, None)

    def _log_perf(self, key: str, ms: float):
        self._perf.setdefault(key, []).append(ms)
        logger.debug(f"⏱ {key}: {ms:.0f}ms")

    def _print_perf_summary(self):
        lines = ["⏱ === Video Perf Summary ==="]
        for key in sorted(self._perf):
            vals = self._perf[key]
            if vals:
                lines.append(f"  {key:30s} avg={sum(vals)/len(vals):6.0f}ms  min={min(vals):5.0f}ms  max={max(vals):5.0f}ms  n={len(vals)}")
        print("\n".join(lines))

    def update(self, frame, bboxes):
        if len(bboxes) == 0:
            return []

        _t0_frame = time.perf_counter()
        _t0_bytetrack = time.perf_counter()
        tracks = self.tracker.update(np.array(bboxes), frame)
        self._log_perf("bytetrack", (time.perf_counter() - _t0_bytetrack) * 1000)
        results = []

        # Tracks which names have already been assigned in this frame to avoid duplicates
        # (a person cannot appear in two places at the same time)
        claimed_this_frame = {}  # name -> track_id

        # Pre-reserves real names already assigned to recently seen tracks, so that
        # another track doesn't "steal" the name when the person leaves the frame momentarily.
        # Names loaded from disk (persistent embeddings) get a long protection window (30s);
        # names auto-enrolled in the session but without a file yet get a short protection (5s).
        _generic_re = __import__('re').compile(r'^(Person_\d+|Unknown)$')
        active_track_ids = {int(t[4]) for t in tracks}
        _now = time.time()
        for tid, tname in self._track_to_name.items():
            if _generic_re.match(tname):
                continue
            last_seen = self._track_last_seen.get(tid, 0.0)
            # Names with persistent embeddings in the database get a longer window
            protection_secs = 30.0 if tname in self.known_embeddings else 5.0
            if tid in active_track_ids or (_now - last_seen) < protection_secs:
                claimed_this_frame[tname] = tid

        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, *_ = track
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]

            if face_img.size == 0:
                continue

            _t0_emb = time.perf_counter()
            current_emb = self._extract_face_embedding(face_img)
            self._log_perf("face_embedding", (time.perf_counter() - _t0_emb) * 1000)
            self._track_last_seen[int(track_id)] = time.time()
            best_name, best_score = self._best_match(current_emb, track_id=int(track_id))

            if best_name != "Unknown":
                # Check if the name has already been claimed by another track in this frame
                if best_name in claimed_this_frame and claimed_this_frame[best_name] != track_id:
                    # Same name on two faces — treat this one as unknown
                    best_name, best_score = "Unknown", 0.0
                else:
                    already_confirmed = self._track_to_name.get(track_id) == best_name
                    if already_confirmed:
                        # Name already confirmed — keep as normal
                        claimed_this_frame[best_name] = track_id
                        self._track_buffer.pop(track_id, None)
                        self._persist_fail_count.pop(track_id, None)
                        self._name_confirm.pop(track_id, None)
                    else:
                        # Block re-confirmation if the current name is still plausible,
                        # unless the new match is much stronger (different identity).
                        current_confirmed = self._track_to_name.get(track_id)
                        if current_confirmed is not None and current_confirmed in self.known_embeddings:
                            persist_sim = float(np.dot(current_emb, self.known_embeddings[current_confirmed]))
                            # Allow transition when: new score is high (>=0.75) AND current identity is weak (<0.35)
                            # Cooldown: blocks new OVERRIDE for 90 frames after the last one
                            override_cooldown_ok = (
                                time.time() - self._override_last_frame.get(track_id, 0.0) > 3.0
                            )
                            strong_new = best_score >= 0.75 and persist_sim < 0.35 and override_cooldown_ok
                            # Generic upgrade → real name: Person_N can be replaced by a real name
                            # with normal threshold (0.45) when the generic identity is already weak (<0.45)
                            import re as _re
                            _is_generic = lambda n: bool(_re.match(r'^(Person_\d+|Unknown)$', n or ""))
                            generic_upgrade = (
                                _is_generic(current_confirmed) and not _is_generic(best_name)
                                and best_score >= self.MATCH_THRESHOLD and persist_sim < 0.45
                                and override_cooldown_ok
                            )
                            if persist_sim >= 0.20 and not strong_new and not generic_upgrade:
                                # Current name still holds — ignore the new match
                                logger.debug(f"👤 Re-confirm BLOCKED track={int(track_id)}: keep '{current_confirmed}' (sim={persist_sim:.2f}) over '{best_name}'")
                                best_name, best_score = "Unknown", 0.0
                            elif strong_new or generic_upgrade:
                                self._override_last_frame[track_id] = time.time()
                                logger.debug(f"👤 Re-confirm OVERRIDE track={int(track_id)}: '{current_confirmed}' (sim={persist_sim:.2f}) → '{best_name}' (score={best_score:.2f})")
                                # Remove the weak identity before re-confirming
                                self._track_to_name.pop(track_id, None)
                                # If we're transitioning from a generic name (Person_N) to a real name,
                                # remove the generic embedding — it was absorbed by the real identity.
                                _generic_emb_re = __import__('re').compile(r'^Person_\d+$')
                                if _generic_emb_re.match(current_confirmed) and not _generic_emb_re.match(best_name):
                                    self.known_embeddings.pop(current_confirmed, None)
                                    # Remove file from disk as well
                                    if self.emb_dir and os.path.exists(self.emb_dir):
                                        for _f in list(os.listdir(self.emb_dir)):
                                            _base = os.path.splitext(_f)[0]
                                            if _base == current_confirmed or _base.startswith(current_confirmed + "_"):
                                                try: os.remove(os.path.join(self.emb_dir, _f))
                                                except OSError: pass
                                    logger.debug(f"👤 Removed generic embedding '{current_confirmed}' (superseded by '{best_name}')")

                        # Not yet confirmed — accumulate votes before promoting
                        if best_name != "Unknown":
                            # Block if another track is already pending with the SAME
                            # real name AND has a higher average score — only the best
                            # face should claim a unique identity.
                            _dominated = False
                            if not _generic_re.match(best_name):
                                for _ot, _ob in self._name_confirm.items():
                                    if _ot != track_id and _ob["name"] == best_name:
                                        _ot_avg = _ob.get("avg_score", 0)
                                        if _ot_avg > best_score:
                                            _dominated = True
                                            logger.debug(f"👤 Pending BLOCKED track={int(track_id)}: "
                                                         f"'{best_name}' (score={best_score:.2f} < track {int(_ot)} avg={_ot_avg:.2f})")
                                            break

                            if _dominated:
                                best_name, best_score = "Unknown", 0.0
                            else:
                                buf = self._name_confirm.get(track_id)
                                if buf and buf["name"] == best_name:
                                    buf["count"] += 1
                                    # Running average score for contention resolution
                                    buf["avg_score"] = (buf["avg_score"] * (buf["count"] - 1) + best_score) / buf["count"]
                                else:
                                    self._name_confirm[track_id] = {"name": best_name, "count": 1, "avg_score": best_score}
                                    buf = self._name_confirm[track_id]

                                if buf["count"] >= self.CONFIRM_FRAMES:
                                    # Before confirming, evict any other track pending
                                    # with the same name (they lost the race)
                                    for _ot in list(self._name_confirm):
                                        if _ot != track_id and self._name_confirm[_ot]["name"] == best_name:
                                            logger.debug(f"👤 Evicted track={int(_ot)}: lost '{best_name}' to track={int(track_id)}")
                                            del self._name_confirm[_ot]
                                    # Confirmed: promote to _track_to_name
                                    claimed_this_frame[best_name] = track_id
                                    self._track_to_name[track_id] = best_name
                                    self._track_buffer.pop(track_id, None)
                                    self._persist_fail_count.pop(track_id, None)
                                    self._name_confirm.pop(track_id, None)
                                    logger.debug(f"👤 Confirmed track={int(track_id)}: '{best_name}' ({self.CONFIRM_FRAMES} frames)")
                                else:
                                    # Still awaiting confirmation — return Unknown for now
                                    logger.debug(f"👤 Pending track={int(track_id)}: '{best_name}' ({buf['count']}/{self.CONFIRM_FRAMES})")
                                    best_name, best_score = "Unknown", 0.0

            if best_name == "Unknown" and track_id in self._track_to_name:
                prev_name = self._track_to_name[track_id]
                # Validate: is the current embedding still compatible with the persisted name?
                # If the persisted name exists in known_embeddings, check minimum similarity.
                # Prevents an incorrectly assigned name from persisting forever.
                persist_ok = True
                if prev_name in self.known_embeddings:
                    persist_sim = float(np.dot(current_emb, self.known_embeddings[prev_name]))
                    if persist_sim < 0.20:
                        fail_count = self._persist_fail_count.get(track_id, 0) + 1
                        self._persist_fail_count[track_id] = fail_count
                        if fail_count >= 3:
                            # 3 consecutive bad frames — drop the identity
                            logger.debug(f"👤 Persist DROPPED track={int(track_id)}: '{prev_name}' sim={persist_sim:.2f} ({fail_count} consecutive fails)")
                            persist_ok = False
                            del self._track_to_name[track_id]
                            del self._persist_fail_count[track_id]
                            self._name_confirm.pop(track_id, None)  # reset pending confirmation
                        else:
                            logger.debug(f"👤 Persist WARNING track={int(track_id)}: '{prev_name}' sim={persist_sim:.2f} ({fail_count}/3)")
                    else:
                        # Good frame — reset fail counter
                        self._persist_fail_count.pop(track_id, None)
                # Only keep the previous name if it's not in use by another track
                if persist_ok and (prev_name not in claimed_this_frame or claimed_this_frame[prev_name] == track_id):
                    best_name = prev_name
                    best_score = 1.0
                    claimed_this_frame[best_name] = track_id

            if best_name == "Unknown":
                # Buffer for auto-enrollment
                buf = self._track_buffer.setdefault(track_id, [])
                buf.append(current_emb)
                if len(buf) >= self.ENROLL_FRAMES:
                    best_name = self._enroll(track_id, buf)
                    best_score = 1.0
                    del self._track_buffer[track_id]
                    claimed_this_frame[best_name] = track_id

            results.append({
                "track_id": track_id,
                "name": best_name,
                "confidence": best_score,
                "bbox": [x1, y1, x2, y2]
            })

        self._log_perf("tracker_update_total", (time.perf_counter() - _t0_frame) * 1000)
        self._perf_frame_count += 1
        if self._perf_frame_count % self._perf_summary_interval == 0:
            self._print_perf_summary()
        return results

    # --- Session metrics API ---

    def get_session_face_metrics(self):
        """Return structured face recognition metrics for the session."""
        _generic_re = re.compile(r'^(Person_\d+|Unknown)$')

        enrollments = [e for e in self._face_events if e["event"] == "enroll"]
        face_fns = [e for e in self._face_events if e["event"] == "face_fn_rename"]

        # Unique tracks that got a real name vs remained generic
        final_names = dict(self._track_to_name)
        real_names = {tid: n for tid, n in final_names.items() if not _generic_re.match(n)}
        generic_names = {tid: n for tid, n in final_names.items() if _generic_re.match(n)}

        # Match score stats (from sampled scores)
        all_scores = [s["best_score"] for s in self._match_scores_sample]
        matched_scores = [s["best_score"] for s in self._match_scores_sample if s["matched"]]
        unmatched_scores = [s["best_score"] for s in self._match_scores_sample if not s["matched"]]

        return {
            "total_tracks": len(final_names),
            "tracks_identified": len(real_names),
            "tracks_generic": len(generic_names),
            "face_fn_count": len(face_fns),
            "face_fn_events": face_fns,
            "enrollments": len(enrollments),
            "match_threshold": self.MATCH_THRESHOLD,
            "merge_threshold": self.MERGE_THRESHOLD,
            "known_embeddings_count": len(self.known_embeddings),
            "total_frames": self._perf_frame_count,
            "match_score_stats": {
                "n_samples": len(all_scores),
                "mean": round(float(np.mean(all_scores)), 4) if all_scores else 0,
                "median": round(float(np.median(all_scores)), 4) if all_scores else 0,
                "max": round(float(max(all_scores)), 4) if all_scores else 0,
                "min": round(float(min(all_scores)), 4) if all_scores else 0,
                "pct_matched": round(len(matched_scores) / len(all_scores), 4) if all_scores else 0,
                "mean_matched": round(float(np.mean(matched_scores)), 4) if matched_scores else 0,
                "mean_unmatched": round(float(np.mean(unmatched_scores)), 4) if unmatched_scores else 0,
            },
            "match_scores_sample": self._match_scores_sample,
            "events": self._face_events,
        }
