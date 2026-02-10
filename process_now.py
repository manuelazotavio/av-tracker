#!/usr/bin/env python3
"""
Script simples para processar todos os arquivos de áudio.
Sem decorações, sem warnings desnecessários, apenas processa.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_verifier import process_audio_files

if __name__ == "__main__":
    print("Processando todos os arquivos de áudio...")
    print("=" * 80)
    
    process_audio_files(
        embeddings_dir='data/embeddings',
        input_dir='data/demo',
        output_csv='results.csv',
        threshold=0.5,
        recursive=True
    )
    
    print("=" * 80)
    print("Processamento concluído!")
