#!/usr/bin/env python3
"""
Processador de áudio simplificado.
Processa arquivos .wav e gera results.csv.
"""

import os
import sys
import csv
import logging
import torch
import torchaudio
from pathlib import Path
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, str(Path(__file__).parent))

from src.multi_speaker_verifier import MultiSpeakerVerifier

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

def main():
    print("\n" + "="*80)
    print("PROCESSAMENTO DE ARQUIVOS DE ÁUDIO")
    print("="*80)
    
    embeddings_dir = "data/embeddings"
    input_dir = "data/demo"
    output_csv = "results.csv"
    threshold = 0.5
    
    print(f"Carregando verificador...")
    verifier = MultiSpeakerVerifier(embeddings_dir, threshold=threshold)
    
    # Find audio files
    input_path = Path(input_dir)
    audio_files = sorted(input_path.glob("**/*.wav"))
    
    print(f"Encontrados {len(audio_files)} arquivos")
    print(f"Iniciando processamento...\n")
    
    results = []
    processed = 0
    errors = 0
    
    for idx, audio_file in enumerate(audio_files, 1):
        filename = audio_file.relative_to(input_path)
        
        try:
            verifications = verifier.process_audio_file(str(audio_file))
            
            if not verifications:
                result = {
                    'file_path': str(filename),
                    'speaker_count': len(verifier.embeddings),
                    'chunks_processed': 0,
                    'identified_count': 0,
                    'unknown_count': 0,
                    'most_common_speaker': 'N/A',
                    'avg_similarity': 0.0,
                    'top_similarity': 0.0,
                    'bottom_similarity': 0.0
                }
                results.append(result)
                processed += 1
                continue
            
            # Analyze results
            speaker_counts = defaultdict(int)
            similarities = []
            unknown_count = 0
            
            for _, speaker, similarity in verifications:
                similarities.append(similarity)
                if speaker == "Unknown":
                    unknown_count += 1
                else:
                    speaker_counts[speaker] += 1
            
            most_common = max(speaker_counts, default="N/A")
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            top_sim = max(similarities) if similarities else 0.0
            bottom_sim = min(similarities) if similarities else 0.0
            identified_count = len(similarities) - unknown_count
            
            result = {
                'file_path': str(filename),
                'speaker_count': len(verifier.embeddings),
                'chunks_processed': len(verifications),
                'identified_count': identified_count,
                'unknown_count': unknown_count,
                'most_common_speaker': str(most_common),
                'avg_similarity': f"{avg_sim:.4f}",
                'top_similarity': f"{top_sim:.4f}",
                'bottom_similarity': f"{bottom_sim:.4f}"
            }
            
            results.append(result)
            processed += 1
            
            # Show progress
            if idx % 10 == 0:
                print(f"[{idx}/{len(audio_files)}] {processed} processados, {errors} erros")
                
        except Exception as e:
            errors += 1
            print(f"[{idx}/{len(audio_files)}] ERRO em {filename}: {str(e)[:80]}")
            
            # Try to continue anyway
            result = {
                'file_path': str(filename),
                'speaker_count': len(verifier.embeddings),
                'chunks_processed': 0,
                'identified_count': 0,
                'unknown_count': 0,
                'most_common_speaker': 'ERROR',
                'avg_similarity': 0.0,
                'top_similarity': 0.0,
                'bottom_similarity': 0.0
            }
            results.append(result)
    
    # Write CSV
    print(f"\nSalvando {len(results)} resultados em {output_csv}...")
    
    fieldnames = [
        'file_path', 'speaker_count', 'chunks_processed', 'identified_count',
        'unknown_count', 'most_common_speaker', 'avg_similarity', 'top_similarity',
        'bottom_similarity'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("="*80)
    print(f"CONCLUIDO!")
    print(f"  Processados: {processed}")
    print(f"  Erros: {errors}")
    print(f"  Arquivo: {output_csv}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
