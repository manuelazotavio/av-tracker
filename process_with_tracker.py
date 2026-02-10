#!/usr/bin/env python3
"""
Processador inteligente que detecta apenas arquivos novos
Integrado com o sistema de rastreamento
"""

import os
import sys
import csv
import logging
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, str(Path(__file__).parent))

from src.multi_speaker_verifier import MultiSpeakerVerifier
from processing_tracker import SmartProcessingTracker
from collections import defaultdict


def process_audio_smart(verifier, audio_path):
    """Processa um arquivo de áudio"""
    try:
        verifications = verifier.process_audio_file(str(audio_path))
        return verifications
    except Exception as e:
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Processador inteligente com rastreamento')
    parser.add_argument('-e', '--embeddings', default='data/embeddings', help='Embeddings dir')
    parser.add_argument('-o', '--output', default='results.csv', help='Output CSV')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold')
    parser.add_argument('--sources', nargs='+', default=['data/demo', 'data/voxceleb_downloaded'],
                        help='Diretórios de áudio')
    parser.add_argument('--full-rescan', action='store_true', help='Rescaneiar todas as fontes')
    
    args = parser.parse_args()
    
    # Inicializar tracker e verificador
    tracker = SmartProcessingTracker()
    
    print("\n" + "="*70)
    print("PROCESSADOR INTELIGENTE COM RASTREAMENTO")
    print("="*70)
    
    # Atualizar contagem de arquivos
    print("\n🔍 Escaneando fontes...")
    for source_dir in args.sources:
        if os.path.exists(source_dir):
            source_name = os.path.basename(source_dir)
            tracker.get_source_stats(source_name, source_dir)
    
    # Mostrar estatísticas
    tracker.print_stats()
    
    # Obter arquivos pendentes
    print("📋 Identificando arquivos novos...")
    unprocessed = tracker.get_unprocessed_files(args.sources)
    
    if not unprocessed:
        print("✅ Todos os arquivos já foram processados!\n")
        return
    
    print(f"⏳ Encontrados {len(unprocessed)} arquivos para processar\n")
    
    # Inicializar verificador
    print("🧠 Carregando modelo...")
    try:
        verifier = MultiSpeakerVerifier(args.embeddings, threshold=args.threshold)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Processar arquivos
    print(f"\n{'='*70}")
    print("PROCESSAMENTO")
    print(f"{'='*70}\n")
    
    results = []
    processed_count = 0
    error_count = 0
    
    for idx, audio_file in enumerate(unprocessed, 1):
        filename = os.path.basename(audio_file)
        
        print(f"[{idx:3d}/{len(unprocessed)}] {filename:50s}", end=' ', flush=True)
        
        try:
            verifications = process_audio_smart(verifier, audio_file)
            
            if not verifications:
                result = {
                    'file_path': filename,
                    'speaker_count': len(verifier.embeddings),
                    'chunks_processed': 0,
                    'identified_count': 0,
                    'unknown_count': 0,
                    'most_common_speaker': 'N/A',
                    'avg_similarity': '0.0000',
                    'top_similarity': '0.0000',
                    'bottom_similarity': '0.0000'
                }
                print("VAZIO")
            else:
                # Analisar resultados
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
                    'file_path': filename,
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
            
            results.append(result)
            processed_count += 1
            
            # Marcar como processado
            tracker.mark_processed(audio_file, results=result)
            
        except Exception as e:
            error_count += 1
            print(f"ERRO: {str(e)[:40]}")
            tracker.mark_processed(audio_file, results={'error': str(e)})
    
    # Salvar resultados
    if results:
        fieldnames = [
            'file_path', 'speaker_count', 'chunks_processed', 'identified_count',
            'unknown_count', 'most_common_speaker', 'avg_similarity', 'top_similarity',
            'bottom_similarity'
        ]
        
        file_exists = os.path.exists(args.output)
        
        print(f"\n📝 Salvando {len(results)} resultados...")
        with open(args.output, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO")
    print("="*70)
    print(f"✅ Processados: {processed_count}")
    print(f"❌ Erros: {error_count}")
    
    lines = Path(args.output).read_text(encoding='utf-8').count('\n') if Path(args.output).exists() else 0
    print(f"📊 Arquivo: {args.output} ({lines} linhas)")
    
    tracker.print_stats()


if __name__ == '__main__':
    main()
