import os
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
    MATCH_THRESHOLD = 0.45  # cosine similarity to consider a known person
    MERGE_THRESHOLD = 0.35  # threshold for merging during auto-enrollment
    ENROLL_FRAMES = 50      # frames to accumulate (~5s) for more stable average embedding
    CONFIRM_FRAMES = 4      # consecutive frames needed to confirm a known-name assignment

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

    def load_known_embeddings(self, emb_dir):
        self.emb_dir = emb_dir
        if not os.path.exists(emb_dir):
            return
        self._emb_files = {}  # name -> filename on disk (for consolidation cleanup)
        for file in os.listdir(emb_dir):
            if file.endswith(".npy"):
                base = file[:-4]  # remove .npy
                if base.endswith("_auto"):
                    name = base[:-5]  # "Person_1_auto" -> "Person_1"
                else:
                    # timestamp format: Name_YYYYMMDD_HHMMSS
                    parts = base.split('_')
                    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                        name = '_'.join(parts[:-2])
                    else:
                        name = parts[0]
                emb = np.load(os.path.join(emb_dir, file))
                self.known_embeddings[name] = emb
                self._emb_files[name] = file
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
        """Merge known embeddings that are too similar (same person enrolled multiple times)."""
        names = list(self.known_embeddings.keys())
        merged_into = {}  # name -> canonical name it was merged into

        for i, name_a in enumerate(names):
            if name_a in merged_into:
                continue
            for name_b in names[i + 1:]:
                if name_b in merged_into:
                    continue
                emb_a = self.known_embeddings[name_a]
                emb_b = self.known_embeddings[name_b]
                sim = float(np.dot(emb_a, emb_b))
                if sim >= self.MERGE_THRESHOLD:
                    # Average and re-normalise
                    avg = (emb_a + emb_b) / 2
                    avg = avg / (np.linalg.norm(avg) + 1e-8)
                    self.known_embeddings[name_a] = avg
                    merged_into[name_b] = name_a
                    print(f"[PersonIDTracker] Consolidated '{name_b}' -> '{name_a}' (sim={sim:.3f})")
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
        # Log top matches para cada face (a cada 30 frames para não poluir)
        if hasattr(self, '_match_log_counter'):
            self._match_log_counter += 1
        else:
            self._match_log_counter = 0
        if self._match_log_counter % 30 == 0 and scores:
            top_str = " | ".join(f"{n}: {s:.2f}" for n, s in scores[:3])
            tid_str = f"track={track_id}" if track_id is not None else ""
            logger.debug(f"👤 FaceMatch {tid_str}: [{top_str}] thr={self.MATCH_THRESHOLD}")
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
        if merge_name is not None and merge_score >= self.MERGE_THRESHOLD:
            logger.debug(f"👤 Enroll track={int(track_id)}: MERGED into '{merge_name}' (sim={merge_score:.3f} >= {self.MERGE_THRESHOLD})")
            self._track_to_name[track_id] = merge_name
            return merge_name

        self._person_counter += 1
        name = f"Person_{self._person_counter}"
        logger.debug(f"👤 Enroll track={int(track_id)}: NEW '{name}' (best_merge='{merge_name}' sim={merge_score:.3f} < {self.MERGE_THRESHOLD})")
        self.known_embeddings[name] = avg_emb
        self._track_to_name[track_id] = name

        if self.emb_dir:
            os.makedirs(self.emb_dir, exist_ok=True)
            filename = f"{name}_auto.npy"
            np.save(os.path.join(self.emb_dir, filename), avg_emb)
            if hasattr(self, '_emb_files'):
                self._emb_files[name] = filename
            print(f"[PersonIDTracker] Auto-enrolled: {name} (track {int(track_id)})")

        return name

    def rename_person(self, old_name, new_name, only_track_id=None):
        """Renomeia uma identidade em memória e em disco quando o nome real é detectado.

        Args:
            only_track_id: se fornecido, só renomeia esse track específico
                           (evita renomear rostos de família que foram mergidos).
        """
        if old_name not in self.known_embeddings:
            return
        if only_track_id is not None:
            # Renomeia apenas o track solicitado; mantém o embedding do old_name
            # para os demais tracks que ainda o usam.
            other_tracks_use = any(
                tid != only_track_id and tname == old_name
                for tid, tname in self._track_to_name.items()
            )
            if other_tracks_use:
                # Copia o embedding para o novo nome (não remove o antigo)
                self.known_embeddings[new_name] = self.known_embeddings[old_name].copy()
                self._track_to_name[only_track_id] = new_name
            else:
                # Único track — pode renomear normalmente
                emb = self.known_embeddings.pop(old_name)
                self.known_embeddings[new_name] = emb
                self._track_to_name[only_track_id] = new_name
        else:
            emb = self.known_embeddings.pop(old_name)
            self.known_embeddings[new_name] = emb
            # Atualiza track → name
            for track_id, tname in self._track_to_name.items():
                if tname == old_name:
                    self._track_to_name[track_id] = new_name
        # Renomeia arquivo em disco
        if self.emb_dir:
            old_file = self._emb_files.pop(old_name, None)
            if old_file:
                old_path = os.path.join(self.emb_dir, old_file)
                new_file = f"{new_name}_auto.npy"
                new_path = os.path.join(self.emb_dir, new_file)
                if os.path.exists(old_path):
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(old_path, new_path)
                self._emb_files[new_name] = new_file
        print(f"[PersonIDTracker] Renomeado '{old_name}' -> '{new_name}'")
        # Re-consolidar após rename para mesclar duplicatas criadas na sessão
        self._consolidate_embeddings()
        # Corrigir _track_to_name de tracks que apontavam para entidades mescladas
        for tid in list(self._track_to_name.keys()):
            if self._track_to_name[tid] not in self.known_embeddings:
                self._track_to_name.pop(tid, None)

    def update(self, frame, bboxes):
        if len(bboxes) == 0:
            return []

        tracks = self.tracker.update(np.array(bboxes), frame)
        results = []

        # Rastreia quais nomes já foram atribuídos neste frame para evitar duplicatas
        # (uma pessoa não pode aparecer em dois lugares ao mesmo tempo)
        claimed_this_frame = {}  # name -> track_id

        # Pré-reserva nomes reais já atribuídos a tracks ativos, para que
        # outro track não "roube" o nome via _best_match numa ordem aleatória
        _generic_re = __import__('re').compile(r'^(Person_\d+|Unknown)$')
        active_track_ids = {int(t[4]) for t in tracks}
        for tid, tname in self._track_to_name.items():
            if tid in active_track_ids and not _generic_re.match(tname):
                claimed_this_frame[tname] = tid

        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, *_ = track
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]

            if face_img.size == 0:
                continue

            current_emb = self._extract_face_embedding(face_img)
            best_name, best_score = self._best_match(current_emb, track_id=int(track_id))

            if best_name != "Unknown":
                # Verifica se o nome já foi reivindicado por outro track neste frame
                if best_name in claimed_this_frame and claimed_this_frame[best_name] != track_id:
                    # Mesmo nome em dois rostos — trata este como desconhecido
                    best_name, best_score = "Unknown", 0.0
                else:
                    already_confirmed = self._track_to_name.get(track_id) == best_name
                    if already_confirmed:
                        # Nome já confirmado — mantém normalmente
                        claimed_this_frame[best_name] = track_id
                        self._track_buffer.pop(track_id, None)
                        self._persist_fail_count.pop(track_id, None)
                        self._name_confirm.pop(track_id, None)
                    else:
                        # Bloqueia re-confirmação se o nome atual ainda é plausível,
                        # a não ser que o novo match seja muito mais forte (identidade diferente).
                        current_confirmed = self._track_to_name.get(track_id)
                        if current_confirmed is not None and current_confirmed in self.known_embeddings:
                            persist_sim = float(np.dot(current_emb, self.known_embeddings[current_confirmed]))
                            # Permite transição quando: novo score é alto (≥0.65) E identidade atual está fraca (<0.35)
                            strong_new = best_score >= 0.65 and persist_sim < 0.35
                            if persist_sim >= 0.20 and not strong_new:
                                # Nome atual ainda segura — ignora o novo match
                                logger.debug(f"👤 Re-confirm BLOCKED track={int(track_id)}: keep '{current_confirmed}' (sim={persist_sim:.2f}) over '{best_name}'")
                                best_name, best_score = "Unknown", 0.0
                            elif strong_new:
                                logger.debug(f"👤 Re-confirm OVERRIDE track={int(track_id)}: '{current_confirmed}' (sim={persist_sim:.2f}) → '{best_name}' (score={best_score:.2f})")
                                # Remove a identidade fraca antes de re-confirmar
                                self._track_to_name.pop(track_id, None)

                        # Ainda não confirmado — acumula votos antes de promover
                        if best_name != "Unknown":
                            buf = self._name_confirm.get(track_id)
                            if buf and buf["name"] == best_name:
                                buf["count"] += 1
                            else:
                                self._name_confirm[track_id] = {"name": best_name, "count": 1}
                                buf = self._name_confirm[track_id]

                            if buf["count"] >= self.CONFIRM_FRAMES:
                                # Confirmado: promove ao _track_to_name
                                claimed_this_frame[best_name] = track_id
                                self._track_to_name[track_id] = best_name
                                self._track_buffer.pop(track_id, None)
                                self._persist_fail_count.pop(track_id, None)
                                self._name_confirm.pop(track_id, None)
                                logger.debug(f"👤 Confirmed track={int(track_id)}: '{best_name}' ({self.CONFIRM_FRAMES} frames)")
                            else:
                                # Ainda aguardando confirmação — retorna Unknown por ora
                                logger.debug(f"👤 Pending track={int(track_id)}: '{best_name}' ({buf['count']}/{self.CONFIRM_FRAMES})")
                                best_name, best_score = "Unknown", 0.0

            if best_name == "Unknown" and track_id in self._track_to_name:
                prev_name = self._track_to_name[track_id]
                # Valida: o embedding atual ainda é compatível com o nome persistido?
                # Se o nome persistido existe em known_embeddings, checa similaridade mínima.
                # Evita que um nome incorretamente atribuído persista para sempre.
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
                            self._name_confirm.pop(track_id, None)  # reseta confirmação pendente
                        else:
                            logger.debug(f"👤 Persist WARNING track={int(track_id)}: '{prev_name}' sim={persist_sim:.2f} ({fail_count}/3)")
                    else:
                        # Good frame — reset fail counter
                        self._persist_fail_count.pop(track_id, None)
                # Só mantém o nome anterior se não estiver em uso por outro track
                if persist_ok and (prev_name not in claimed_this_frame or claimed_this_frame[prev_name] == track_id):
                    best_name = prev_name
                    best_score = 1.0
                    claimed_this_frame[best_name] = track_id

            if best_name == "Unknown":
                # Buffer para auto-enrolamento
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

        return results
