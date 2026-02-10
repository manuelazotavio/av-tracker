#!/usr/bin/env python3
"""
Processador robusto com salvamento progressivo.
Salva cada resultado conforme processa para evitar perda de dados.
"""

import os
import sys
import csv
import logging
import torch
import gc
from pathlib import Path
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('pyannote').setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from src.multi_speaker_verifier import MultiSpeakerVerifier

# Arquivo de progresso
PROGRESS_FILE = "process_progress.txt"
OUTPUT_CSV = "results.csv"

def read_progress():
    """Lê quais arquivos já foram processados."""
    processed = set()
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, 'r') as f:
            processed = set(line.strip() for line in f)
    return processed

def save_progress(filepath):
    """Salva que um arquivo foi processado."""
    with open(PROGRESS_FILE, 'a') as f:
        f.write(filepath + '\n')

def append_result_to_csv(result):
    """Adiciona um resultado ao CSV."""
    fieldnames = [
        'file_path', 'speaker_count', 'chunks_processed', 'identified_count',
        'unknown_count', 'most_common_speaker', 'avg_similarity', 'top_similarity',
        'bottom_similarity'
    ]
    
    # Verificar se arquivo existe para decidir se escreve header
    file_exists = Path(OUTPUT_CSV).exists()
    
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def main():
    print("\n" + "="*80)
    print("PROCESSADOR DE ÁUDIO COM SALVAMENTO PROGRESSIVO")
    print("="*80)
    
    embeddings_dir = "data/embeddings"
    input_dir = "data/demo"
    
    print("Carregando verificador...")
    try:
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.5)
    except Exception as e:
        print(f"ERRO ao carregar verificador: {e}")
        return
    
    # Find audio files
    input_path = Path(input_dir)
    audio_files = sorted(input_path.glob("**/*.wav"))
    
    print(f"Encontrados {len(audio_files)} arquivos")
    
    # Ler progresso anterior
    processed = read_progress()
    print(f"Já processados: {len(processed)} arquivos\n")
    
    processed_count = 0
    errors = 0
    
    for idx, audio_file in enumerate(audio_files, 1):
        filename = audio_file.relative_to(input_path)
        filename_str = str(filename)
        
        # Skip se já foi processado
        if filename_str in processed:
            processed_count += 1
            continue
        
        try:
            print(f"[{idx}/{len(audio_files)}] {filename_str}...", end=' ', flush=True)
            
            # Process audio
            verifications = verifier.process_audio_file(str(audio_file))
            
            if not verifications:
                result = {
                    'file_path': filename_str,
                    'speaker_count': len(verifier.embeddings),
                    'chunks_processed': 0,
                    'identified_count': 0,
                    'unknown_count': 0,
                    'most_common_speaker': 'N/A',
                    'avg_similarity': 0.0,
                    'top_similarity': 0.0,
                    'bottom_similarity': 0.0
                }
                print("(vazio)")
            else:
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
                    'file_path': filename_str,
                    'speaker_count': len(verifier.embeddings),
                    'chunks_processed': len(verifications),
                    'identified_count': identified_count,
                    'unknown_count': unknown_count,
                    'most_common_speaker': str(most_common),
                    'avg_similarity': f"{avg_sim:.4f}",
                    'top_similarity': f"{top_sim:.4f}",
                    'bottom_similarity': f"{bottom_sim:.4f}"
                }
                print(f"OK ({len(verifications)} chunks)")
            
            # Salvar resultado
            append_result_to_csv(result)
            save_progress(filename_str)
            processed_count += 1
            
            # Liberar memória periodicamente
            if idx % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except KeyboardInterrupt:
            print("\n\nProcessamento interrompido pelo usuário")
            break
        except Exception as e:
            errors += 1
            print(f"ERRO: {str(e)[:60]}")
            save_progress(filename_str)  # Marcar como processado mesmo com erro
    
    print("\n" + "="*80)
    print(f"RESUMO:")
    print(f"  Processados: {processed_count}/{len(audio_files)}")
    print(f"  Erros: {errors}")
    lines = (Path(OUTPUT_CSV).read_text().count('\n')) if Path(OUTPUT_CSV).exists() else 0
    print(f"  Arquivo: {OUTPUT_CSV} ({lines} linhas)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
