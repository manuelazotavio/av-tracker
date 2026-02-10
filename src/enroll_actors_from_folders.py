import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

RAW_DATA_PATH = "../vox_100_atores/wav" 

EMBEDDINGS_DIR = "../data/embeddings"

def load_encoder(device="cuda"):
    print("🔧 Carregando modelo ECAPA-VoxCeleb...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

def process_actor_folder(actor_path, encoder, device):
    """Lê todos os wavs de um ator, gera embeddings e calcula a média."""
    embeddings = []
    
    
    for root, dirs, files in os.walk(actor_path):
        for file in files:
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)
                try:
                    
                    signal, fs = torchaudio.load(filepath)
                    
                    
                    signal = signal.to(device)

                   
                    with torch.no_grad():
                        emb = encoder.encode_batch(signal)
                      
                        embeddings.append(emb.squeeze().cpu().numpy())
                except Exception as e:
                    print(f"⚠️ Erro ao ler {file}: {e}")

    if not embeddings:
        return None

   
    mean_embedding = np.mean(np.array(embeddings), axis=0)
    
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
        
    return mean_embedding

def main():
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Iniciando cadastro usando {device.upper()}...")
    
    encoder = load_encoder(device)
  
    actor_ids = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
    
    print(f"📂 Encontrados {len(actor_ids)} atores.")

    for actor_id in tqdm(actor_ids):
        actor_path = os.path.join(RAW_DATA_PATH, actor_id)
        
        
        embedding = process_actor_folder(actor_path, encoder, device)
        
        if embedding is not None:
            
            output_path = os.path.join(EMBEDDINGS_DIR, f"{actor_id}.npy")
            np.save(output_path, embedding)
        else:
            print(f"⚠️ Nenhum áudio válido encontrado para {actor_id}")

    print("\n✅ Cadastro finalizado! Embeddings salvos em:", EMBEDDINGS_DIR)

if __name__ == "__main__":
    main()