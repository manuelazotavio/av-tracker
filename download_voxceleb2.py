#!/usr/bin/env python3
"""
Download de áudios do VoxCeleb2 via Hugging Face
Versão alternativa SEM FFmpeg/TorchCodec (mais simples e rápida)
"""

import os
import json
import hashlib
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download, HfApi

class VoxCelebDownloader:
    def __init__(self, output_dir='data/voxceleb_downloaded', tracker_file='download_history.json'):
        self.output_dir = output_dir
        self.tracker_file = tracker_file
        self.history = self.load_history()
        self.api = HfApi()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_history(self):
        """Carrega histórico de downloads"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {
            'downloaded_files': {},
            'failed_files': [],
            'total_downloaded': 0,
            'dataset_version': None
        }
    
    def save_history(self):
        """Salva histórico"""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_file_hash(self, filepath):
        """Calcula hash MD5 de um arquivo"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def is_already_downloaded(self, filename):
        """Verifica se arquivo já foi baixado"""
        return filename in self.history['downloaded_files']
    
    def download_voxceleb2_simple(self, limit=None):
        """
        Baixa áudios WAV diretamente do repositório Hugging Face
        SEM necessidade de FFmpeg ou TorchCodec
        
        Args:
            limit: Máximo de arquivos a baixar (None = todos)
        """
        print("\n" + "="*70)
        print("DOWNLOAD VOXCELEB2 (SEM FFMPEG/TORCHCODEC)")
        print("="*70)
        
        repo_id = "acul3/voxceleb2"
        
        try:
            print(f"\n📥 Listando arquivos do repositório...")
            print(f"   Repo: {repo_id}")
            
            # Listar apenas arquivos .wav
            files = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            wav_files = [f for f in files if f.endswith('.wav')]
            
            print(f"✅ Encontrados {len(wav_files)} arquivos .wav")
            
        except Exception as e:
            print(f"❌ ERRO ao listar arquivos: {e}")
            print(f"\n   Dica: Você pode precisar de:")
            print(f"   - Conta no HuggingFace: https://huggingface.co/join")
            print(f"   - Token: https://huggingface.co/settings/tokens")
            print(f"   - Comando: huggingface-cli login")
            return 0, 0, 0
        
        downloaded = 0
        skipped = 0
        failed = 0
        
        # Baixar arquivos
        for idx, wav_file in enumerate(wav_files, 1):
            if limit and downloaded >= limit:
                break
            
            filename = os.path.basename(wav_file)
            
            # Verificar se já foi baixado
            if self.is_already_downloaded(filename):
                skipped += 1
                continue
            
            try:
                print(f"[{idx:4d}] Baixando: {filename:50s}", end=' ', flush=True)
                
                # Criar diretório do speaker
                speaker_dir = os.path.join(self.output_dir, 'voxceleb2_files')
                Path(speaker_dir).mkdir(parents=True, exist_ok=True)
                
                # Baixar arquivo
                filepath = hf_hub_download(
                    repo_id=repo_id,
                    filename=wav_file,
                    repo_type="dataset",
                    cache_dir=speaker_dir,
                    force_download=False
                )
                
                # Registrar no histórico
                file_hash = self.get_file_hash(filepath)
                self.history['downloaded_files'][filename] = {
                    'path': filepath,
                    'hash': file_hash,
                    'size': os.path.getsize(filepath),
                    'repo_file': wav_file
                }
                
                downloaded += 1
                print("✅")
                
            except Exception as e:
                failed += 1
                self.history['failed_files'].append({
                    'file': wav_file,
                    'error': str(e)[:100]
                })
                print(f"❌ {str(e)[:40]}")
                
                if failed <= 5:
                    pass  # Já mostrado acima
        
        # Atualizar histórico
        self.history['total_downloaded'] += downloaded
        self.history['dataset_version'] = 'voxceleb2_acul3'
        self.save_history()
        
        print(f"\n{'='*70}")
        print(f"RESUMO DO DOWNLOAD")
        print(f"{'='*70}")
        print(f"✅ Baixados: {downloaded}")
        print(f"⏭️  Pulados (já existem): {skipped}")
        print(f"❌ Erros: {failed}")
        print(f"📁 Diretório: {self.output_dir}")
        print(f"📊 Total geral: {self.history['total_downloaded']}")
        print(f"{'='*70}\n")
        
        return downloaded, skipped, failed
    
    def get_speakers(self):
        """Retorna lista de speakers baixados"""
        speakers = {}
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    audio_files = [f for f in os.listdir(item_path) if f.endswith('.wav')]
                    speakers[item] = len(audio_files)
        return speakers
    
    def get_stats(self):
        """Mostra estatísticas de downloads"""
        print(f"\n{'='*70}")
        print(f"ESTATÍSTICAS DE DOWNLOAD")
        print(f"{'='*70}")
        
        if os.path.exists(self.output_dir):
            total_files = len(self.history['downloaded_files'])
            print(f"Total de arquivos baixados: {total_files}")
            print(f"Arquivo de histórico: {self.tracker_file}")
            print(f"Diretório: {self.output_dir}")
        else:
            print(f"Nenhum arquivo baixado ainda")
        
        print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download VoxCeleb2 do Hugging Face (sem FFmpeg)')
    parser.add_argument('--limit', type=int, default=50,
                        help='Máximo de arquivos a baixar (padrão: 50)')
    parser.add_argument('--output-dir', default='data/voxceleb_downloaded',
                        help='Diretório de saída')
    parser.add_argument('--stats', action='store_true',
                        help='Mostrar estatísticas apenas')
    
    args = parser.parse_args()
    
    downloader = VoxCelebDownloader(output_dir=args.output_dir)
    
    if args.stats:
        downloader.get_stats()
    else:
        # Download
        downloader.download_voxceleb2_simple(limit=args.limit)
        
        # Estatísticas finais
        downloader.get_stats()


if __name__ == '__main__':
    main()
