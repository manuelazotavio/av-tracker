#!/usr/bin/env python3
"""
Processador otimizado com chunks maiores para processar mais rápido.
Salva progressivamente cada resultado.
"""

import os
import sys
import csv
import torch
import torchaudio
import gc
from pathlib import Path
from collections import defaultdict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, str(Path(__file__).parent))

from src.multi_speaker_verifier import MultiSpeakerVerifier

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
    
    file_exists = Path(OUTPUT_CSV).exists()
    
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def process_audio_fast(verifier, audio_path, chunk_duration=6.0):
    """Processa áudio com chunks maiores para ser mais rápido."""
    try:
        signal, sample_rate = torchaudio.load(audio_path)
        
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            signal = resampler(signal)
            sample_rate = 16000
        
        audio_np = signal.squeeze().numpy()
        chunk_samples = int(chunk_duration * sample_rate)
        total_samples = len(audio_np)
        
        results = []
        
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_np[start:end]
            
            if len(chunk) < sample_rate:
                continue
            
            speaker, score, _ = verifier._process_audio_chunk(chunk, sample_rate)
            results.append((0, speaker, score))
        
        return results
    except Exception as e:
        print(f"\nERRO em {audio_path}: {str(e)[:50]}")
        return None

def main():
    print("\n" + "="*80)
    print("PROCESSADOR OTIMIZADO - CHUNKS MAIORES (6s)")
    print("="*80)
    
    try:
        verifier = MultiSpeakerVerifier("data/embeddings", threshold=0.5)
    except Exception as e:
        print(f"ERRO ao carregar: {e}")
        return
    
    input_path = Path("data/demo")
    audio_files = sorted(input_path.glob("**/*.wav"))
    
    print(f"Encontrados {len(audio_files)} arquivos")
    
    processed = read_progress()
    print(f"Já processados: {len(processed)}\n")
    
    for idx, audio_file in enumerate(audio_files, 1):
        filename_str = str(audio_file.relative_to(input_path))
        
        if filename_str in processed:
            continue
        
        print(f"[{idx:3d}/{len(audio_files)}] {filename_str:50s}", end=' ', flush=True)
        
        verifications = process_audio_fast(verifier, str(audio_file), chunk_duration=6.0)
        
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
            print("VAZIO")
        else:
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
            print(f"OK ({len(verifications):2d} chunks)")
        
        append_result_to_csv(result)
        save_progress(filename_str)
        
        if idx % 30 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    lines = Path(OUTPUT_CSV).read_text().count('\n') if Path(OUTPUT_CSV).exists() else 0
    print("\n" + "="*80)
    print(f"CONCLUÍDO: {lines} arquivos em {OUTPUT_CSV}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
