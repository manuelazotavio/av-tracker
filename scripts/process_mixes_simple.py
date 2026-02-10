"""
Processamento SIMPLES dos mixes:
- Não usa diarização (muito lento)
- Apenas extrai embeddings dos mixes
- Compara com embeddings dos atores
- Calcula métricas de acurácia
"""
import os
import sys
import logging
import numpy as np
import csv
import json
from pathlib import Path
from scipy.spatial.distance import cosine

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multi_speaker_verifier

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_ground_truth(gt_file):
    """Carrega o arquivo ground truth."""
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            speakers = []
            for line in lines:
                if line.startswith("Speakers:"):
                    speaker_names = [s.strip() for s in line.replace("Speakers:", "").strip().split(",")]
                    speakers = speaker_names
                    break
            return speakers
    except Exception as e:
        logger.warning(f"Erro ao ler ground truth: {e}")
        return []

def process_mix_simple(mix_file, verifier, ground_truth):
    """
    Processa um mix de forma simples:
    - Extrai embeddings do mix inteiro
    - Identifica os speakers mais prováveis
    - Compara com ground truth
    """
    try:
        # Processar o arquivo de áudio
        results = verifier.process_audio_file(str(mix_file), chunk_duration=2.0)
        
        if not results:
            logger.warning(f"Sem resultados extraídos de {mix_file.name}")
            return {
                'identified_speakers': [],
                'accuracy': 0.0,
                'error': 'Sem resultados'
            }
        
        # Identificar speakers únicos
        identified_speakers = set()
        speaker_scores = {}
        
        for chunk_id, speaker, score in results:
            if speaker != "Unknown":
                identified_speakers.add(speaker)
                if speaker not in speaker_scores:
                    speaker_scores[speaker] = []
                speaker_scores[speaker].append(score)
        
        # Calcular score médio para cada speaker
        avg_scores = {s: np.mean(scores) for s, scores in speaker_scores.items()}
        
        # Calcular acurácia
        matches = sum(1 for gt in ground_truth if gt in identified_speakers)
        accuracy = matches / len(ground_truth) if ground_truth else 0.0
        
        return {
            'identified_speakers': list(identified_speakers),
            'speaker_scores': avg_scores,
            'accuracy': accuracy,
            'num_chunks': len(results)
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar {mix_file.name}: {e}")
        return {
            'identified_speakers': [],
            'accuracy': 0.0,
            'error': str(e)
        }

def main():
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
    MIXES_DIR = BASE_DIR / "data" / "mixed_audio"
    OUTPUT_CSV = BASE_DIR / "mix_processing_results_simple.csv"
    
    # Verificar diretórios
    if not EMBEDDINGS_DIR.exists():
        logger.error(f"Pasta de embeddings não encontrada: {EMBEDDINGS_DIR}")
        return
    
    if not MIXES_DIR.exists():
        logger.error(f"Pasta de mixes não encontrada: {MIXES_DIR}")
        return
    
    # Inicializar verifier
    logger.info("Inicializando MultiSpeakerVerifier...")
    verifier = multi_speaker_verifier.MultiSpeakerVerifier(str(EMBEDDINGS_DIR), threshold=0.5)
    
    # Coletar todos os arquivos mixados
    mix_files = sorted(MIXES_DIR.glob("**/mix_*.wav"))
    logger.info(f"Encontrados {len(mix_files)} arquivos para processar\n")
    
    # Resultados
    results = []
    
    # Processar cada arquivo
    for idx, mix_file in enumerate(mix_files, 1):
        # Encontrar ground truth
        gt_file = mix_file.with_name(f"{mix_file.stem}_ground_truth.txt")
        ground_truth = []
        if gt_file.exists():
            ground_truth = load_ground_truth(gt_file)
        
        logger.info(f"[{idx}/{len(mix_files)}] Processando {mix_file.name}")
        logger.info(f"  Ground Truth: {ground_truth}")
        
        # Processar o mix
        result = process_mix_simple(mix_file, verifier, ground_truth)
        
        logger.info(f"  Identificados: {result['identified_speakers']}")
        logger.info(f"  Accuracy: {result['accuracy']:.1%}\n")
        
        # Adicionar ao resultados
        results.append({
            'audio_file': mix_file.name,
            'ground_truth': ground_truth,
            'identified_speakers': result['identified_speakers'],
            'accuracy': result['accuracy'],
            'num_speakers_gt': len(ground_truth),
            'num_speakers_identified': len(result['identified_speakers'])
        })
        
        if idx % 10 == 0:
            logger.info(f"✓ Processados {idx}/{len(mix_files)} arquivos")
    
    # Salvar resultados em CSV
    logger.info(f"\nSalvando resultados em {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'audio_file', 'ground_truth_speakers', 'identified_speakers',
            'num_speakers_gt', 'num_speakers_identified', 'accuracy'
        ])
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'audio_file': result['audio_file'],
                'ground_truth_speakers': '|'.join(result['ground_truth']),
                'identified_speakers': '|'.join(result['identified_speakers']),
                'num_speakers_gt': result['num_speakers_gt'],
                'num_speakers_identified': result['num_speakers_identified'],
                'accuracy': result['accuracy']
            })
    
    # Estatísticas finais
    total = len(results)
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    perfect_accuracy = sum(1 for r in results if r['accuracy'] == 1.0)
    
    print(f"\n{'='*60}")
    print(f"[OK] PROCESSAMENTO COMPLETO!")
    print(f"{'='*60}")
    print(f"Total de arquivos: {total}")
    print(f"Accuracy média: {avg_accuracy:.1%}")
    print(f"Arquivos com 100% accuracy: {perfect_accuracy}/{total} ({perfect_accuracy/total*100:.1f}%)")
    print(f"Resultados CSV: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
