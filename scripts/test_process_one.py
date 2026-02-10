import os
import sys
import torch
import logging

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multi_speaker_verifier
import multi_speaker_verification

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv('HF_TOKEN', "")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    TEST_FILE = os.path.join(BASE_DIR, "data", "mixed_audio", "mix_2_speakers", "mix_2_speakers_001.wav")
    
    if not os.path.exists(TEST_FILE):
        logger.error(f"Arquivo de teste não encontrado: {TEST_FILE}")
        return
    
    logger.info(f"Teste com arquivo: {TEST_FILE}")
    
    # Inicializar verifier
    logger.info("Carregando embeddings...")
    verifier = multi_speaker_verifier.MultiSpeakerVerifier(EMBEDDINGS_DIR, threshold=0.5)
    
    # Inicializar transcriber
    logger.info("Inicializando transcriber...")
    transcriber = multi_speaker_verification.LargeMeetingTranscriber(verifier, HF_TOKEN, whisper_size="base", device="cuda")
    
    # Processar
    logger.info("Processando áudio...")
    transcript = transcriber.process_audio(TEST_FILE)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Transcrição completa! {len(transcript)} linhas")
    logger.info(f"{'='*60}\n")
    
    for entry in transcript:
        timestamp = f"[{entry['start']:.1f}s]"
        speaker = entry['speaker']
        suffix = entry.get('suffix', '')
        text = entry['text']
        print(f"{timestamp} {speaker}{suffix}: {text}")
    
    logger.info(f"\n✅ Teste concluído!")

if __name__ == "__main__":
    main()
