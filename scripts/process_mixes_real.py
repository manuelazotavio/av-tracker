import os
import sys
import torch
import logging
import numpy as np
import torchaudio
import csv
import json
from pathlib import Path
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_speaker_verifier import MultiSpeakerVerifier
from multi_speaker_verification import LargeMeetingTranscriber

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Token HuggingFace (necessário para PyAnnote)
HF_TOKEN = os.getenv('HF_TOKEN', "")

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

def calculate_accuracy(transcript, ground_truth):
    """
    Calcula a acurácia da identificação dos speakers.
    Verifica quantos dos speakers identificados estão no ground truth.
    """
    if not ground_truth:
        return 0.0
    
    # Extrair speakers únicos do transcript
    identified_speakers = set()
    for entry in transcript:
        speaker = entry['speaker']
        # Remover sufixos como " (Voz 1)" ou " (Voz 2)"
        if " (Voz" in speaker:
            speaker = speaker.split(" (Voz")[0]
        identified_speakers.add(speaker)
    
    # Contar quantos speakers do ground truth foram identificados
    matches = 0
    for gt_speaker in ground_truth:
        if gt_speaker in identified_speakers:
            matches += 1
    
    accuracy = matches / len(ground_truth)
    return accuracy

def process_mixed_audio_files(mixes_dir, embeddings_dir, output_dir, hf_token):
    """
    Processa todos os arquivos mixados com diarização, transcrição e speaker verification.
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar o verifier
    logger.info("Inicializando MultiSpeakerVerifier...")
    verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.5)
    
    # Inicializar o transcriber
    logger.info("Inicializando LargeMeetingTranscriber...")
    transcriber = LargeMeetingTranscriber(verifier, hf_token, whisper_size="base", device="cuda")
    
    # Coletar todos os arquivos mixados
    mix_files = sorted(Path(mixes_dir).glob("**/mix_*.wav"))
    logger.info(f"Encontrados {len(mix_files)} arquivos para processar")
    
    # Resultados
    results = []
    
    # Processar cada arquivo
    for idx, mix_file in enumerate(mix_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando [{idx}/{len(mix_files)}]: {mix_file.name}")
        logger.info(f"{'='*60}")
        
        # Encontrar ground truth
        gt_file = mix_file.with_name(f"{mix_file.stem}_ground_truth.txt")
        ground_truth = []
        if gt_file.exists():
            ground_truth = load_ground_truth(gt_file)
            logger.info(f"Ground Truth: {ground_truth}")
        
        try:
            # Processar o áudio com diarização e transcrição
            transcript = transcriber.process_audio(str(mix_file))
            
            # Calcular accuracy
            accuracy = calculate_accuracy(transcript, ground_truth)
            
            # Extrair speakers identificados
            identified_speakers = set()
            for entry in transcript:
                speaker = entry['speaker']
                if " (Voz" in speaker:
                    speaker = speaker.split(" (Voz")[0]
                identified_speakers.add(speaker)
            
            # Salvar transcrição
            output_file = output_dir / f"{mix_file.stem}_transcription.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Arquivo: {mix_file.name}\n")
                f.write(f"Ground Truth: {', '.join(ground_truth)}\n")
                f.write(f"Speakers Identificados: {', '.join(identified_speakers)}\n")
                f.write(f"Accuracy: {accuracy:.1%}\n")
                f.write(f"{'-'*60}\n\n")
                
                for entry in transcript:
                    timestamp = f"[{entry['start']:.1f}s]"
                    speaker = entry['speaker']
                    suffix = entry.get('suffix', '')
                    text = entry['text']
                    f.write(f"{timestamp} {speaker}{suffix}: {text}\n")
            
            logger.info(f"✓ Transcrição salva em: {output_file}")
            logger.info(f"✓ Accuracy: {accuracy:.1%}")
            
            # Adicionar resultado
            results.append({
                'audio_file': mix_file.name,
                'ground_truth': ground_truth,
                'identified_speakers': list(identified_speakers),
                'num_speakers_gt': len(ground_truth),
                'num_speakers_identified': len(identified_speakers),
                'accuracy': accuracy,
                'transcript_lines': len(transcript)
            })
            
        except Exception as e:
            logger.error(f"Erro ao processar {mix_file.name}: {e}")
            results.append({
                'audio_file': mix_file.name,
                'ground_truth': ground_truth,
                'identified_speakers': [],
                'num_speakers_gt': len(ground_truth),
                'num_speakers_identified': 0,
                'accuracy': 0.0,
                'transcript_lines': 0,
                'error': str(e)
            })
    
    return results

def main():
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
    MIXES_DIR = BASE_DIR / "data" / "mixed_audio"
    OUTPUT_DIR = BASE_DIR / "data" / "transcriptions"
    OUTPUT_CSV = BASE_DIR / "mix_processing_results_real.csv"
    
    # Verificar HF_TOKEN
    if not HF_TOKEN:
        logger.warning("⚠️ HF_TOKEN não configurado. Tentando processar sem autenticação...")
        logger.warning("   Se houver erro, configure: $env:HF_TOKEN='seu_token'")
        logger.warning("   Token disponível em: https://huggingface.co/settings/tokens")
    
    # Verificar diretórios
    if not EMBEDDINGS_DIR.exists():
        logger.error(f"Pasta de embeddings não encontrada: {EMBEDDINGS_DIR}")
        return
    
    if not MIXES_DIR.exists():
        logger.error(f"Pasta de mixes não encontrada: {MIXES_DIR}")
        return
    
    # Processar arquivos
    logger.info("Iniciando processamento dos mixes...")
    results = process_mixed_audio_files(MIXES_DIR, EMBEDDINGS_DIR, OUTPUT_DIR, HF_TOKEN)
    
    # Salvar resultados em CSV
    logger.info(f"\nSalvando resultados em {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'audio_file', 'ground_truth_speakers', 'identified_speakers',
            'num_speakers_gt', 'num_speakers_identified', 
            'accuracy', 'transcript_lines'
        ])
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'audio_file': result['audio_file'],
                'ground_truth_speakers': '|'.join(result['ground_truth']),
                'identified_speakers': '|'.join(result['identified_speakers']),
                'num_speakers_gt': result['num_speakers_gt'],
                'num_speakers_identified': result['num_speakers_identified'],
                'accuracy': result['accuracy'],
                'transcript_lines': result['transcript_lines']
            })
    
    # Estatísticas finais
    total = len(results)
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    perfect_accuracy = sum(1 for r in results if r['accuracy'] == 1.0)
    
    print(f"\n{'='*60}")
    print(f"✅ PROCESSAMENTO COMPLETO!")
    print(f"{'='*60}")
    print(f"Total de arquivos: {total}")
    print(f"Accuracy média: {avg_accuracy:.1%}")
    print(f"Arquivos com 100% accuracy: {perfect_accuracy}/{total} ({perfect_accuracy/total*100:.1f}%)")
    print(f"Transcrições salvas em: {OUTPUT_DIR}")
    print(f"Resultados CSV: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
