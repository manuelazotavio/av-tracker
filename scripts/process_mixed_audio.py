import os
import torch
import logging
import numpy as np
import torchaudio
import csv
import json
from pathlib import Path
from scipy.spatial.distance import cosine
import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MixedAudioProcessor:
    def __init__(self, embeddings_dir, meta_path=None, threshold=0.5):
        """
        Processa arquivos de áudio mixados:
        1. Carrega os embeddings pré-computados dos atores
        2. Extrai embeddings do áudio mixado
        3. Identifica quais atores estão presentes
        4. Compara com ground truth
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.threshold = threshold
        self.actor_embeddings = {}
        self.actor_names = {}
        
        # Carregar nomes dos atores do metadata
        if meta_path:
            self._load_actor_names(meta_path)
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Carrega os embeddings pré-computados de todos os atores."""
        logger.info(f"Carregando embeddings de {self.embeddings_dir}")
        
        # Tentar carregar de individual/ directory (arquivos .npy)
        individual_dir = self.embeddings_dir / "individual"
        if individual_dir.exists():
            for npy_file in individual_dir.glob("*.npy"):
                actor_id = npy_file.stem
                try:
                    embedding = np.load(npy_file)
                    self.actor_embeddings[actor_id] = embedding
                    # Tentar carregar o nome do metadata se disponível
                    self.actor_names[actor_id] = actor_id
                except Exception as e:
                    logger.warning(f"Erro ao carregar {npy_file}: {e}")
        
        # Se não encontrou nada, tentar arquivos .pkl no diretório raiz
        if not self.actor_embeddings:
            for pkl_file in self.embeddings_dir.glob("*.pkl"):
                actor_id = pkl_file.stem
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            self.actor_embeddings[actor_id] = data.get('embedding')
                            self.actor_names[actor_id] = data.get('name', actor_id)
                        else:
                            self.actor_embeddings[actor_id] = data
                except Exception as e:
                    logger.warning(f"Erro ao carregar {pkl_file}: {e}")
        
        logger.info(f"Carregados {len(self.actor_embeddings)} atores")
    
    def _load_actor_names(self, meta_path):
        """Carrega os nomes dos atores do arquivo de metadata."""
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    actor_id = row['VoxCeleb1 ID']
                    actor_name = row['VGGFace1 ID']
                    self.actor_names[actor_id] = actor_name
            logger.info(f"Carregados nomes de {len(self.actor_names)} atores")
        except Exception as e:
            logger.warning(f"Erro ao carregar metadata: {e}")
    
    def extract_embeddings_from_audio(self, audio_path, chunk_duration=2.0, sr=16000):
        """
        Extrai embeddings de um arquivo de áudio em chunks.
        Usa SpeechBrain para extrair embeddings.
        """
        try:
            wav, sr_actual = torchaudio.load(audio_path)
            
            if sr_actual != sr:
                resampler = torchaudio.transforms.Resample(sr_actual, sr)
                wav = resampler(wav)
            
            # Converter para mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            chunk_samples = int(chunk_duration * sr)
            embeddings = []
            
            # Importar o extrator de embeddings do SpeechBrain
            try:
                from speechbrain.inference.speaker import SpeakerRecognition
                # Usar o modelo pré-treinado
                recognizer = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                
                # Extrair embeddings para cada chunk
                for i in range(0, wav.shape[1], chunk_samples):
                    chunk = wav[:, i:i+chunk_samples]
                    if chunk.shape[1] < sr * 0.5:  # Mínimo 0.5 segundos
                        continue
                    
                    try:
                        embedding = recognizer.encode_batch(chunk).squeeze().cpu().numpy()
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.warning(f"Erro ao extrair embedding: {e}")
                        continue
                
                return embeddings
            except ImportError:
                logger.error("SpeechBrain não disponível")
                return []
        
        except Exception as e:
            logger.error(f"Erro ao processar áudio {audio_path}: {e}")
            return []
    
    def identify_speakers(self, embeddings, top_n=5):
        """
        Identifica os atores presentes no áudio com base nos embeddings.
        Retorna os top-N atores com maior similaridade.
        """
        if not embeddings:
            return []
        
        # Calcular similaridade média de cada embedding com cada ator
        speaker_scores = {}
        
        for embedding in embeddings:
            for actor_id, actor_embedding in self.actor_embeddings.items():
                if actor_embedding is None:
                    continue
                
                # Normalizar embeddings
                emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
                actor_norm = actor_embedding / (np.linalg.norm(actor_embedding) + 1e-8)
                
                # Calcular similaridade de cosseno
                similarity = 1 - cosine(emb_norm, actor_norm)
                
                if actor_id not in speaker_scores:
                    speaker_scores[actor_id] = []
                speaker_scores[actor_id].append(similarity)
        
        # Calcular score médio para cada ator
        avg_scores = {}
        for actor_id, scores in speaker_scores.items():
            avg_scores[actor_id] = np.mean(scores)
        
        # Retornar top-N atores
        sorted_speakers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_speakers[:top_n]
    
    def load_ground_truth(self, gt_file):
        """Carrega o arquivo ground truth."""
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                speakers = []
                for line in lines:
                    if line.startswith("Speakers:"):
                        speaker_names = line.replace("Speakers:", "").strip().split(", ")
                        speakers = speaker_names
                        break
                return speakers
        except Exception as e:
            logger.warning(f"Erro ao ler ground truth: {e}")
            return []
    
    def process_mix(self, audio_path, gt_path=None):
        """
        Processa um arquivo de áudio mixado.
        Retorna: identificação de atores, score de acurácia, etc.
        """
        result = {
            'audio_file': os.path.basename(audio_path),
            'ground_truth': [],
            'identified_speakers': [],
            'accuracy': 0.0,
            'top_5_speakers': []
        }
        
        # Carregar ground truth se disponível
        if gt_path and os.path.exists(gt_path):
            result['ground_truth'] = self.load_ground_truth(gt_path)
        
        # Extrair embeddings do áudio
        logger.info(f"Processando {os.path.basename(audio_path)}")
        embeddings = self.extract_embeddings_from_audio(audio_path)
        
        if not embeddings:
            logger.warning(f"Sem embeddings extraídos de {audio_path}")
            return result
        
        # Identificar atores
        top_speakers = self.identify_speakers(embeddings, top_n=5)
        result['top_5_speakers'] = [
            {
                'actor_id': actor_id,
                'name': self.actor_names.get(actor_id, actor_id),
                'score': float(score)
            }
            for actor_id, score in top_speakers
        ]
        
        # Calcular acurácia se houver ground truth
        if result['ground_truth']:
            identified_names = [s['name'] for s in result['top_5_speakers'][:len(result['ground_truth'])]]
            matches = sum(1 for gt in result['ground_truth'] if gt in identified_names)
            result['accuracy'] = matches / len(result['ground_truth'])
        
        return result

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    META_PATH = os.path.join(BASE_DIR, "vox1_meta.csv")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    MIXES_DIR = os.path.join(BASE_DIR, "data", "mixed_audio")
    OUTPUT_CSV = os.path.join(BASE_DIR, "mix_processing_results.csv")
    
    # Verificar se os diretórios existem
    if not os.path.exists(EMBEDDINGS_DIR):
        logger.error(f"Pasta de embeddings não encontrada: {EMBEDDINGS_DIR}")
        return
    
    if not os.path.exists(MIXES_DIR):
        logger.error(f"Pasta de mixes não encontrada: {MIXES_DIR}")
        return
    
    # Inicializar processador
    processor = MixedAudioProcessor(EMBEDDINGS_DIR, meta_path=META_PATH)
    
    # Processar todos os mixes
    results = []
    
    for mix_file in sorted(Path(MIXES_DIR).glob("**/mix_*.wav")):
        # Encontrar o arquivo ground truth correspondente
        gt_file = mix_file.with_name(f"{mix_file.stem}_ground_truth.txt")
        
        result = processor.process_mix(str(mix_file), str(gt_file) if gt_file.exists() else None)
        results.append(result)
        
        # Imprimir resultado
        print(f"\n{'='*60}")
        print(f"Arquivo: {result['audio_file']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Top 5 Identificados:")
        for i, speaker in enumerate(result['top_5_speakers'], 1):
            print(f"  {i}. {speaker['name']} (Score: {speaker['score']:.4f})")
        print(f"Acurácia: {result['accuracy']:.1%}")
    
    # Salvar resultados em CSV
    if results:
        logger.info(f"Salvando resultados em {OUTPUT_CSV}")
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'audio_file', 'ground_truth', 'top_1_speaker', 'top_1_score',
                'accuracy', 'all_results_json'
            ])
            writer.writeheader()
            
            for result in results:
                top_speaker = result['top_5_speakers'][0] if result['top_5_speakers'] else None
                writer.writerow({
                    'audio_file': result['audio_file'],
                    'ground_truth': '|'.join(result['ground_truth']),
                    'top_1_speaker': top_speaker['name'] if top_speaker else 'N/A',
                    'top_1_score': top_speaker['score'] if top_speaker else 0,
                    'accuracy': result['accuracy'],
                    'all_results_json': json.dumps(result['top_5_speakers'])
                })
        
        logger.info(f"Processados {len(results)} arquivos")
        print(f"\n✅ Resultados salvos em {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
