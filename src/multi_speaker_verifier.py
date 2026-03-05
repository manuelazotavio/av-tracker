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
        
        print(f"🧠 Inicializando Verificador (Device: {self.device})")
        print(f"   📂 Embeddings: {embedding_directory}")
        
        # Carrega o modelo ECAPA-TDNN (O mesmo do enrollment)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": self.device}
        )
        
        # Carrega os bancos de dados na memória
        self.embeddings = {}
        self.load_embeddings(embedding_directory)

    @staticmethod
    def _clean_name(base):
        """Extrai nome limpo do filename base, removendo sufixos _auto e _YYYYMMDD_HHMMSS."""
        if base.endswith("_auto"):
            return base[:-5]
        parts = base.split('_')
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            return '_'.join(parts[:-2])
        return base

    def load_embeddings(self, directory):
        if not os.path.exists(directory):
            print("❌ Diretório de embeddings não encontrado!")
            return

        self.embedding_directory = directory
        # name -> list[tensor]: acumula embeddings por nome base
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
                        raise ValueError("Embedding ausente no perfil")
                    emb_tensor = torch.from_numpy(emb_numpy).float().to(self.device)
                    accumulated.setdefault(name, []).append(emb_tensor)
                    count += 1
            except Exception as e:
                print(f"❌ Erro ao carregar {filename}: {e}")

        # Agrupa embeddings do mesmo nome:
        # - Alta similaridade entre si → mesmo falante → média (múltiplas gravações)
        # - Baixa similaridade         → homônimos    → chaves únicas "Nome", "Nome 2", …
        SAME_PERSON_SIM = 0.55  # abaixo disto, considera pessoas diferentes
        for name, tensors in accumulated.items():
            if len(tensors) == 1:
                self.embeddings[name] = tensors[0]
                continue
            # Agrupa por clustering guloso: cada tensor entra no grupo de maior sim
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
                if idx > 0:
                    print(f"[Verifier] Homônimo detectado: '{name}' → '{key}'")

        n_pessoas = len(self.embeddings)
        print(f"   ✅ {count} arquivo(s) → {n_pessoas} pessoa(s) carregadas.")

    def _process_audio_chunk(self, audio_chunk, sample_rate=16000):
        # 1. Prepara o áudio (Garante Tensor [1, Time])
        if isinstance(audio_chunk, np.ndarray):
            signal = torch.from_numpy(audio_chunk).float().to(self.device)
        else:
            signal = audio_chunk.to(self.device)

        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # 2. Extrai o embedding da voz atual (Separada)
        with torch.no_grad():
            # O classifier retorna [1, 1, 192], fazemos squeeze para [192]
            output = self.classifier.encode_batch(signal)
            current_embedding = output.squeeze()

        # 3. Compara com todos os atores do banco
        best_score = -1.0
        best_speaker = "Unknown"

        # Variável para debug visual (apenas se for sobreposição/separado)
        debug_scores = []

        for speaker, stored_embedding in self.embeddings.items():
            # Similaridade de Cosseno (PyTorch)
            # A stored_embedding também precisa estar no mesmo device
            score = torch.nn.functional.cosine_similarity(current_embedding, stored_embedding, dim=0).item()
            
            debug_scores.append((speaker, score))

            if score > best_score:
                best_score = score
                best_speaker = speaker

        # 4. Lógica de decisão
        
        # Ordena para vermos os top 3 no log
        debug_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = debug_scores[:3]
        
        # Se a pontuação for muito baixa, imprimimos para entender o drama
        if best_score < 0.25:
            # Monta string de debug
            top_str = " | ".join([f"{n}: {s:.1%}" for n, s in top_3])
            # Descomente a linha abaixo se quiser ver TODOS os comparativos no terminal
            # print(f"   📊 Comparativo: {top_str}")

        # Retorna (best, score, todos_candidatos_acima_do_threshold)
        # O chamador aplica gender check e margin check com contexto completo.
        candidates = [(name, sc) for name, sc in debug_scores if sc >= self.threshold]
        if best_score >= self.threshold:
            return best_speaker, best_score, candidates
        else:
            return "Unknown", best_score, candidates

    def rename_embedding(self, old_name, new_name):
        """Renomeia embeddings de voz em memória e em disco quando o nome real é detectado."""
        # Atualiza chave em memória (chaves são nomes limpos)
        if old_name in self.embeddings:
            self.embeddings[new_name] = self.embeddings.pop(old_name)
        # Renomeia arquivos em disco cujo nome base começa com old_name
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
                        print(f"[Verifier] Renomeado '{fname}' -> '{new_fname}'")