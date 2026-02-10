#!/usr/bin/env python3
"""
Baixa áudios do VoxCeleb2 usando API direta do HuggingFace
SEM PyTorch, SEM datasets, SEM FFmpeg - APENAS requests!
"""

import os
import json
import requests
from pathlib import Path

class VoxCelebSimplestDownloader:
    def __init__(self, output_dir='data/voxceleb_downloaded'):
        self.output_dir = output_dir
        self.tracker_file = 'download_history.json'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Sua token do HuggingFace (pegue em https://huggingface.co/settings/tokens)
        self.token = os.getenv('HF_TOKEN', None)
        
        self.history = self.load_history()
    
    def load_history(self):
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {'downloaded': [], 'count': 0}
    
    def save_history(self):
        with open(self.tracker_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def download_from_url(self, url, filepath):
        """Baixa arquivo via URL direta"""
        headers = {}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    def download_parquet_info(self):
        """Baixa informações sobre os arquivos parquet do dataset"""
        print("\n📦 Esta é uma abordagem alternativa:")
        print("   O VoxCeleb2 no HuggingFace está em formato Parquet")
        print("   Você tem duas opções:\n")
        
        print("OPÇÃO 1: Use outro dataset mais simples")
        print("   - Common Voice: mozilla-foundation/common_voice")
        print("   - LibriSpeech: librispeech_asr")
        print("   - Google Speech Commands: speech_commands")
        
        print("\nOPÇÃO 2: Baixe o VoxCeleb2 original")
        print("   Site oficial: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html")
        print("   Precisa registrar e aceitar termos")
        
        print("\nOPÇÃO 3: Use os dados de demonstração que você já tem")
        print("   Você já tem 268 arquivos em data/demo/")
        print("   Processe esses primeiro!")
        
        print("\n" + "="*70)


def main():
    print("\n" + "="*70)
    print("DOWNLOAD VOXCELEB2 - DIAGNÓSTICO")
    print("="*70)
    
    downloader = VoxCelebSimplestDownloader()
    
    print("\n🔍 Verificando ambiente...")
    
    # Verificar token
    if downloader.token:
        print(f"✅ Token HuggingFace encontrada")
    else:
        print(f"❌ Token HuggingFace NÃO encontrada")
        print(f"\n   Para configurar:")
        print(f"   1. Crie token em: https://huggingface.co/settings/tokens")
        print(f"   2. Execute: huggingface-cli login")
        print(f"   3. Ou defina: set HF_TOKEN=seu_token")
    
    # Verificar PyTorch
    try:
        import torch
        print(f"✅ PyTorch instalado: {torch.__version__}")
    except:
        print(f"❌ PyTorch com problemas (normal no Windows)")
    
    # Verificar datasets
    try:
        import datasets
        print(f"✅ datasets instalado: {datasets.__version__}")
    except:
        print(f"❌ datasets com problemas")
    
    # Mostrar alternativas
    downloader.download_parquet_info()
    
    # Verificar dados existentes
    demo_dir = Path('data/demo')
    if demo_dir.exists():
        wav_files = list(demo_dir.glob('**/*.wav'))
        print(f"\n📁 VOCÊ JÁ TEM:")
        print(f"   {len(wav_files)} arquivos .wav em data/demo/")
        print(f"   Processe esses primeiro com:")
        print(f"   python process_with_tracker.py")


if __name__ == '__main__':
    main()
