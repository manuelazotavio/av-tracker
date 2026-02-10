#!/usr/bin/env python3
"""
Sistema de rastreamento inteligente para processamento
Detecta automaticamente arquivos já processados e novos
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime


class SmartProcessingTracker:
    def __init__(self, tracker_file='processing_tracker.json'):
        self.tracker_file = tracker_file
        self.data = self.load()
    
    def load(self):
        """Carrega arquivo de rastreamento"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'version': '1.0',
            'processed_files': {},
            'processing_stats': {
                'total_processed': 0,
                'last_processed': None,
                'last_batch_time': None
            },
            'sources': {
                'demo': {'total': 0, 'processed': 0},
                'voxceleb_downloaded': {'total': 0, 'processed': 0}
            }
        }
    
    def save(self):
        """Salva arquivo de rastreamento"""
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def get_file_hash(self, filepath):
        """Calcula hash MD5 do arquivo"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def mark_processed(self, filepath, results=None):
        """Marca um arquivo como processado"""
        rel_path = str(Path(filepath).relative_to(Path.cwd()))
        file_hash = self.get_file_hash(filepath)
        
        self.data['processed_files'][rel_path] = {
            'hash': file_hash,
            'size': os.path.getsize(filepath),
            'date_processed': datetime.now().isoformat(),
            'results': results or {}
        }
        
        self.data['processing_stats']['total_processed'] += 1
        self.data['processing_stats']['last_processed'] = datetime.now().isoformat()
        
        self.save()
    
    def is_already_processed(self, filepath):
        """Verifica se arquivo foi processado (detecta mudanças)"""
        rel_path = str(Path(filepath).relative_to(Path.cwd()))
        
        if rel_path not in self.data['processed_files']:
            return False
        
        # Verificar se arquivo foi modificado
        stored_hash = self.data['processed_files'][rel_path]['hash']
        current_hash = self.get_file_hash(filepath)
        
        return stored_hash == current_hash
    
    def get_unprocessed_files(self, directories):
        """
        Retorna lista de arquivos NÃO processados
        
        Args:
            directories: Lista de diretórios a verificar
        
        Returns:
            Lista de caminhos de arquivos não processados
        """
        unprocessed = []
        
        if isinstance(directories, str):
            directories = [directories]
        
        for directory in directories:
            if not os.path.exists(directory):
                continue
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.wav'):
                        filepath = os.path.join(root, file)
                        
                        if not self.is_already_processed(filepath):
                            unprocessed.append(filepath)
        
        return sorted(unprocessed)
    
    def get_source_stats(self, source_name, directory):
        """Atualiza estatísticas de uma fonte"""
        if not os.path.exists(directory):
            return
        
        total = 0
        processed = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.wav'):
                    total += 1
                    filepath = os.path.join(root, file)
                    if self.is_already_processed(filepath):
                        processed += 1
        
        self.data['sources'][source_name] = {
            'total': total,
            'processed': processed
        }
        self.save()
    
    def get_all_stats(self):
        """Retorna estatísticas completas"""
        stats = {
            'total_processed': self.data['processing_stats']['total_processed'],
            'last_processed': self.data['processing_stats']['last_processed'],
            'sources': self.data['sources'],
            'pending': {}
        }
        
        # Calcular pendentes por fonte
        for source, info in self.data['sources'].items():
            pending = info['total'] - info['processed']
            stats['pending'][source] = pending
        
        return stats
    
    def print_stats(self):
        """Imprime estatísticas formatadas"""
        stats = self.get_all_stats()
        
        print("\n" + "="*70)
        print("ESTATÍSTICAS DE PROCESSAMENTO")
        print("="*70)
        print(f"\n📊 GERAL:")
        print(f"   Total processados: {stats['total_processed']}")
        print(f"   Última atualização: {stats['last_processed']}")
        
        print(f"\n📁 POR FONTE:")
        for source, info in stats['sources'].items():
            pending = stats['pending'][source]
            print(f"\n   {source.upper()}:")
            print(f"      Processados: {info['processed']}")
            print(f"      Pendentes: {pending}")
            print(f"      Total: {info['total']}")
        
        total_pending = sum(stats['pending'].values())
        print(f"\n⏳ RESUMO:")
        print(f"   Pendentes no total: {total_pending}")
        print(f"   Arquivo: {self.tracker_file}")
        print("="*70 + "\n")
        
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de rastreamento de processamento')
    parser.add_argument('--stats', action='store_true', help='Mostrar estatísticas')
    parser.add_argument('--update-sources', action='store_true', 
                       help='Atualizar contagem de arquivos por fonte')
    parser.add_argument('--list-pending', action='store_true',
                       help='Listar arquivos pendentes')
    
    args = parser.parse_args()
    
    tracker = SmartProcessingTracker()
    
    if args.update_sources:
        print("Atualizando contagem de arquivos...")
        tracker.get_source_stats('demo', 'data/demo')
        if os.path.exists('data/voxceleb_downloaded'):
            tracker.get_source_stats('voxceleb_downloaded', 'data/voxceleb_downloaded')
        tracker.print_stats()
    
    elif args.list_pending:
        print("Arquivos pendentes para processar:\n")
        directories = ['data/demo']
        if os.path.exists('data/voxceleb_downloaded'):
            directories.append('data/voxceleb_downloaded')
        
        unprocessed = tracker.get_unprocessed_files(directories)
        
        if not unprocessed:
            print("✅ Todos os arquivos já foram processados!")
        else:
            print(f"⏳ Encontrados {len(unprocessed)} arquivos pendentes:\n")
            for filepath in unprocessed[:20]:  # Mostrar primeiros 20
                print(f"   - {filepath}")
            
            if len(unprocessed) > 20:
                print(f"\n   ... e mais {len(unprocessed) - 20} arquivos")
    
    else:
        tracker.print_stats()


if __name__ == '__main__':
    main()
