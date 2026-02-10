#!/usr/bin/env python3
"""
Processa todos os arquivos de áudio e gera o CSV de resultados.

Este script processa todos os arquivos .wav no diretório data/demo
usando o MultiSpeakerVerifier e gera o arquivo results.csv atualizado.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the verifier processing script function
from scripts.run_verifier import process_audio_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Processa todos os arquivos de áudio."""
    
    embeddings_dir = "data/embeddings"
    input_dir = "data/demo"
    output_csv = "results.csv"
    threshold = 0.5
    
    logger.info("=" * 80)
    logger.info("PROCESSAMENTO DE TODOS OS ARQUIVOS DE ÁUDIO")
    logger.info("=" * 80)
    logger.info(f"Embeddings: {embeddings_dir}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Threshold: {threshold}")
    logger.info("=" * 80)
    
    try:
        process_audio_files(
            embeddings_dir=embeddings_dir,
            input_dir=input_dir,
            output_csv=output_csv,
            threshold=threshold,
            recursive=True
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 80)
        logger.info(f"Arquivo gerado: {output_csv}")
        logger.info("\nPróximos passos:")
        logger.info("1. Visualizar o dashboard: python generate_html_report.py")
        logger.info("2. Iniciar servidor web: python web_dashboard.py")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\nERRO durante processamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
