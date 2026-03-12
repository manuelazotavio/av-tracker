import torch
import torchaudio
import os
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

class MultiSpeakerVerifier:
    def __init__(self, embedding_directory, threshold=0.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.embedding_directory = embedding_directory
        
        print(f"🧠 Initializing Verifier (Device: {self.device})")
        print(f"   📂 Embeddings: {embedding_directory}")

        # Load the ECAPA-TDNN model (same one used for enrollment)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": self.device}
        )
        
        # Load the databases into memory
        self.embeddings = {}
        self._raw_embeddings = {}     # name → list[tensor] — for incremental centroid
        self._auto_enroll_last = {}   # name → timestamp of last auto-enroll
        self.load_embeddings(embedding_directory)

    @staticmethod
    def _clean_name(base):
        """Extract clean name from filename base, removing _auto and _YYYYMMDD_HHMMSS suffixes."""
        if base.endswith("_auto"):
            base = base[:-5]  # strip _auto, then continue to remove timestamp
        parts = base.split('_')
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            return '_'.join(parts[:-2])
        return base

    def load_embeddings(self, directory):
        if not os.path.exists(directory):
            print("❌ Embeddings directory not found!")
            return

        self.embedding_directory = directory
        # name -> list[tensor]: accumulate embeddings by base name
        accumulated = {}
        count = 0
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            try:
                if filename.endswith(".npy"):
                    name = self._clean_name(os.path.splitext(filename)[0])
                    emb_numpy = np.load(path)
                    emb_tensor = torch.from_numpy(emb_numpy).float().to(self.device)
                    accumulated.setdefault(name, []).append(emb_tensor)
                    count += 1
                elif filename.endswith(".pkl"):
                    import pickle
                    with open(path, "rb") as f:
                        profile = pickle.load(f)
                    raw_name = profile.get("speaker_name") or os.path.splitext(filename)[0]
                    name = self._clean_name(raw_name)
                    emb_numpy = profile.get("embedding")
                    if emb_numpy is None:
                        raise ValueError("Embedding missing from profile")
                    emb_tensor = torch.from_numpy(emb_numpy).float().to(self.device)
                    accumulated.setdefault(name, []).append(emb_tensor)
                    count += 1
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

        # Group embeddings with the same name:
        # - High similarity between them → same speaker → average (multiple recordings)
        # - Low similarity               → homonyms     → unique keys "Name", "Name 2", ...
        SAME_PERSON_SIM = 0.35  # below this, consider different people (actual homonyms)
        # 0.35 is more permissive: same person under different conditions (indoor/outdoor, different mic)
        # keeps in a single cluster instead of generating "Joao 2", "Joao 3" etc.
        for name, tensors in accumulated.items():
            if len(tensors) == 1:
                self.embeddings[name] = tensors[0]
                self._raw_embeddings[name] = list(tensors)
                continue
            # Group by greedy clustering: each tensor joins the group with highest sim
            groups = []  # list of list[tensor]
            for t in tensors:
                placed = False
                for g in groups:
                    rep = g[0]
                    sim = torch.nn.functional.cosine_similarity(t, rep, dim=0).item()
                    if sim >= SAME_PERSON_SIM:
                        g.append(t)
                        placed = True
                        break
                if not placed:
                    groups.append([t])
            for idx, g in enumerate(groups):
                key = name if idx == 0 else f"{name} {idx + 1}"
                avg = torch.stack(g).mean(dim=0)
                norm = torch.norm(avg)
                self.embeddings[key] = avg / norm if norm > 0 else avg
                self._raw_embeddings[key] = list(g)
                if idx > 0:
                    print(f"[Verifier] Homonym detected: '{name}' → '{key}'")

        n_pessoas = len(self.embeddings)
        print(f"   ✅ {count} file(s) → {n_pessoas} person(s) loaded.")

    def auto_enroll(self, name: str, audio_np: np.ndarray,
                    max_total: int = 8, cooldown_s: float = 300.0,
                    min_sim: float = 0.65, max_sim: float = 0.96) -> bool:
        """
        Automatically saves a voice embedding when identification was high confidence.
        - max_total   : max .npy files per person on disk (includes manual ones)
        - cooldown_s  : minimum seconds between auto-enrolls for the same speaker (default: 5min)
        - min_sim     : minimum sim with current centroid — sanity check (is this not the person?)
        - max_sim     : maximum sim — if we already have something identical, it adds no diversity
        Returns True if a new embedding was saved.
        """
        import time
        from datetime import datetime as _dt

        # Cooldown per speaker
        now = time.time()
        if now - self._auto_enroll_last.get(name, 0.0) < cooldown_s:
            return False

        # Total file limit on disk for this name
        if self.embedding_directory and os.path.exists(self.embedding_directory):
            n_on_disk = sum(
                1 for f in os.listdir(self.embedding_directory)
                if f.endswith(".npy") and self._clean_name(os.path.splitext(f)[0]) == name
            )
            if n_on_disk >= max_total:
                return False

        # Extract embedding from audio
        signal = torch.from_numpy(audio_np).float().to(self.device)
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal).squeeze()
        norm = torch.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Sanity check against current centroid
        if name in self.embeddings:
            centroid = self.embeddings[name]
            sim = torch.nn.functional.cosine_similarity(emb, centroid, dim=0).item()
            if sim < min_sim:
                return False  # not this person
            if sim > max_sim:
                return False  # identical to what we already have — adds no diversity

        # Save to disk
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{name}_{ts}_auto.npy"
        fpath = os.path.join(self.embedding_directory, fname)
        np.save(fpath, emb.cpu().numpy())

        # Update centroid in memory with exact incremental average
        raws = self._raw_embeddings.setdefault(name, [])
        raws.append(emb)
        new_avg = torch.stack(raws).mean(dim=0)
        nn = torch.norm(new_avg)
        self.embeddings[name] = new_avg / nn if nn > 0 else new_avg

        self._auto_enroll_last[name] = now
        return True

    def _process_audio_chunk(self, audio_chunk, sample_rate=16000):
        # 1. Prepare the audio (Ensure Tensor [1, Time])
        if isinstance(audio_chunk, np.ndarray):
            signal = torch.from_numpy(audio_chunk).float().to(self.device)
        else:
            signal = audio_chunk.to(self.device)

        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # 2. Extract the embedding of the current voice (Separated)
        with torch.no_grad():
            # The classifier returns [1, 1, 192], we squeeze to [192]
            output = self.classifier.encode_batch(signal)
            current_embedding = output.squeeze()

        # 3. Compare with all speakers in the database
        best_score = -1.0
        best_speaker = "Unknown"

        # Variable for visual debug (only if overlap/separated)
        debug_scores = []

        for speaker, stored_embedding in self.embeddings.items():
            # Cosine Similarity (PyTorch)
            # The stored_embedding also needs to be on the same device
            score = torch.nn.functional.cosine_similarity(current_embedding, stored_embedding, dim=0).item()
            
            debug_scores.append((speaker, score))

            if score > best_score:
                best_score = score
                best_speaker = speaker

        # 4. Decision logic

        # Sort to see top 3 in the log
        debug_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = debug_scores[:3]
        
        # If the score is too low, we print to understand what's going on
        if best_score < 0.25:
            # Build debug string
            top_str = " | ".join([f"{n}: {s:.1%}" for n, s in top_3])
            # Uncomment the line below to see ALL comparisons in the terminal
            # print(f"   📊 Comparison: {top_str}")

        # Return (best, score, all_candidates_above_threshold)
        # Margin check: require minimum margin over the runner-up to avoid
        # ambiguous attribution when two embeddings have similar sim.
        second_score = debug_scores[1][1] if len(debug_scores) >= 2 else -1.0
        margin = best_score - second_score
        candidates = [(name, sc) for name, sc in debug_scores if sc >= self.threshold]
        if best_score >= self.threshold and margin >= 0.07:
            return best_speaker, best_score, candidates
        else:
            # Return actual best name even below threshold so callers can log/diagnose it.
            # Callers should check len(candidates)==0 (or best_score < threshold) to
            # decide whether to trust the result for attribution.
            return best_speaker, best_score, candidates

    def rename_embedding(self, old_name, new_name):
        """Renames voice embeddings in memory and on disk when the real name is detected."""
        # Update key in memory (keys are clean names)
        if old_name in self.embeddings:
            self.embeddings[new_name] = self.embeddings.pop(old_name)
        # Rename files on disk whose base name starts with old_name
        if self.embedding_directory and os.path.exists(self.embedding_directory):
            for fname in os.listdir(self.embedding_directory):
                if not fname.endswith(".npy"):
                    continue
                base = os.path.splitext(fname)[0]
                if base == old_name or base.startswith(old_name + "_"):
                    new_fname = new_name + base[len(old_name):] + ".npy"
                    old_path = os.path.join(self.embedding_directory, fname)
                    new_path = os.path.join(self.embedding_directory, new_fname)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"[Verifier] Renamed '{fname}' -> '{new_fname}'")