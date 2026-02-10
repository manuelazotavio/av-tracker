#!/usr/bin/env python3
"""
Download SIMPLES do VoxCeleb2
Baixa os áudios diretamente do dataset sem complicações
"""

import os
import json
import io
import soundfile as sf
from pathlib import Path
from datasets import load_dataset

class SimpleVoxCelebDownloader:
    def __init__(self, output_dir='data/voxceleb_downloaded'):
        self.output_dir = output_dir
        self.tracker_file = 'download_history.json'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.history = self.load_history()
    
    def load_history(self):
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {'downloaded': [], 'count': 0}
    
    def save_history(self):
        with open(self.tracker_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def download(self, limit=50):
        """Baixa áudios do VoxCeleb2"""
        print("\n" + "="*70)
        print("DOWNLOAD VOXCELEB2 - VERSÃO SIMPLES")
        print("="*70)
        
        try:
            print(f"\n📥 Carregando dataset (modo streaming)...")
            
            # Carregar sem decode automático de áudio para evitar FFmpeg
            dataset = load_dataset(
                'acul3/voxceleb2',
                split='train',
                streaming=True
            )
            
            print(f"✅ Dataset carregado!")
            print(f"⏬ Baixando {limit} áudios...\n")
            
        except Exception as e:
            print(f"❌ ERRO: {e}")
            print(f"\n💡 SOLUÇÃO:")
            print(f"   1. Instale: pip install datasets soundfile")
            print(f"   2. Login: huggingface-cli login")
            print(f"   3. Token: https://huggingface.co/settings/tokens")
            return
        
        downloaded = 0
        skipped = 0
        errors = 0
        
        for idx, example in enumerate(dataset):
            if downloaded >= limit:
                break
            
            try:
                # Extrair informações
                speaker_id = example.get('speaker_id', f'unknown_{idx}')
                audio_data = example.get('audio')
                
                # Criar diretório
                speaker_dir = os.path.join(self.output_dir, str(speaker_id))
                Path(speaker_dir).mkdir(parents=True, exist_ok=True)
                
                # Salvar áudio
                if audio_data and 'array' in audio_data:
                    filename = f"audio_{idx:05d}.wav"
                    filepath = os.path.join(speaker_dir, filename)
                    
                    # Verificar se já existe
                    if os.path.exists(filepath):
                        skipped += 1
                        continue
                    
                    # Salvar com soundfile
                    sample_rate = audio_data.get('sampling_rate', 16000)
                    audio_array = audio_data['array']
                    
                    sf.write(filepath, audio_array, sample_rate)
                    
                    self.history['downloaded'].append(filepath)
                    downloaded += 1
                    
                    print(f"[{downloaded:3d}/{limit}] ✅ {speaker_id}/{filename}")
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[{idx:3d}] ❌ Erro: {str(e)[:60]}")
        
        # Salvar histórico
        self.history['count'] += downloaded
        self.save_history()
        
        print(f"\n{'='*70}")
        print(f"RESUMO")
        print(f"{'='*70}")
        print(f"✅ Baixados: {downloaded}")
        print(f"⏭️  Pulados: {skipped}")
        print(f"❌ Erros: {errors}")
        print(f"📁 Pasta: {self.output_dir}")
        print(f"📊 Total geral: {self.history['count']}")
        print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download simples VoxCeleb2')
    parser.add_argument('--limit', type=int, default=50,
                        help='Quantos áudios baixar (padrão: 50)')
    parser.add_argument('--output', default='data/voxceleb_downloaded',
                        help='Pasta de saída')
    
    args = parser.parse_args()
    
    downloader = SimpleVoxCelebDownloader(output_dir=args.output)
    downloader.download(limit=args.limit)


if __name__ == '__main__':
    main()
