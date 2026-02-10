#!/usr/bin/env python3
"""
WORKFLOW COMPLETO AUTOMATIZADO
Executa todo o pipeline do zero até as métricas finais
"""

import os
import sys
import subprocess
import time
from pathlib import Path


class CompleteWorkflow:
    def __init__(self):
        self.steps_completed = []
        self.steps_failed = []
    
    def run_command(self, cmd, description, background=False):
        """Executa comando e monitora resultado"""
        print(f"\n{'='*70}")
        print(f"▶️  {description}")
        print(f"{'='*70}")
        print(f"Comando: {cmd}\n")
        
        start_time = time.time()
        
        try:
            if background:
                result = subprocess.Popen(cmd, shell=True)
                print(f"✅ Processo iniciado em background (PID: {result.pid})")
                self.steps_completed.append(description)
                return True
            else:
                result = subprocess.run(cmd, shell=True, capture_output=False)
                
                elapsed = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"\n✅ {description} - OK ({elapsed:.1f}s)")
                    self.steps_completed.append(description)
                    return True
                else:
                    print(f"\n❌ {description} - ERRO (código {result.returncode})")
                    self.steps_failed.append(description)
                    return False
                    
        except Exception as e:
            print(f"\n❌ {description} - EXCEÇÃO: {e}")
            self.steps_failed.append(description)
            return False
    
    def check_dependencies(self):
        """Verifica se dependências estão instaladas"""
        print("\n" + "="*70)
        print("VERIFICAÇÃO DE DEPENDÊNCIAS")
        print("="*70)
        
        deps = {
            'remotezip': 'pip install remotezip',
            'torch': 'pip install torch',
            'torchaudio': 'pip install torchaudio',
            'speechbrain': 'pip install speechbrain'
        }
        
        missing = []
        
        for dep, install_cmd in deps.items():
            try:
                __import__(dep)
                print(f"✅ {dep}")
            except:
                print(f"❌ {dep} - Instale com: {install_cmd}")
                missing.append(dep)
        
        if missing:
            print(f"\n⚠️  Instale as dependências faltantes primeiro!")
            return False
        
        print(f"\n✅ Todas as dependências OK")
        return True
    
    def run(self):
        """Executa workflow completo"""
        print("\n" + "="*70)
        print("🚀 WORKFLOW COMPLETO: 300 ATORES → MÉTRICAS")
        print("="*70)
        
        # 0. Verificar dependências
        if not self.check_dependencies():
            print("\n❌ Instale as dependências antes de continuar")
            return
        
        # 1. Download dos atores
        if not Path('vox_300_atores').exists() or len(list(Path('vox_300_atores').rglob('*.wav'))) < 100:
            print("\n📥 PASSO 1: Baixando 300 atores ALEATÓRIOS...")
            if not self.run_command('python src/download.py', '1. Download de 300 atores'):
                print("\n⚠️  Falha no download. Continue manualmente ou tente novamente")
                # Não retorna - continua com o que tiver
        else:
            print("\n✅ PASSO 1: Atores já baixados (pulando)")
            self.steps_completed.append('1. Download de 300 atores')
        
        # 1.5. Criar CSV de mapeamento
        print("\n📋 PASSO 1.5: Criando CSV de mapeamento de áudios...")
        if not self.run_command(
            'python create_audio_csv.py',
            '1.5. Criação de CSV de mapeamento'
        ):
            print("\n❌ Falha ao criar CSV. Verifique os logs acima.")
            return
        
        # 2. Criar embeddings
        print("\n🧠 PASSO 2: Criando embeddings...")
        if not self.run_command(
            'python scripts/create_embeddings_from_csv.py --csv data/audio_files.csv --samples 1 --out data/embeddings/all_embeddings.pkl',
            '2. Criação de embeddings'
        ):
            print("\n❌ Falha ao criar embeddings. Verifique os logs acima.")
            return
        
        # 3. Mixar áudios
        print("\n🎵 PASSO 3: Mixando áudios (2, 3, 4, 5 speakers)...")
        if not self.run_command(
            'python scripts/create_and_mix_audio.py --n-speakers 2,3,4,5 --samples-per-mix 50 --output data/mixed_audio',
            '3. Mixagem de áudios'
        ):
            print("\n❌ Falha na mixagem. Verifique os logs acima.")
            return
        
        # 4. Processar áudios mixados
        print("\n⚙️  PASSO 4: Processando áudios mixados...")
        if not self.run_command(
            'python scripts/run_verifier.py -e data/embeddings -i data/demo -o results.csv -t 0.5',
            '4. Processamento de áudios'
        ):
            print("\n❌ Falha no processamento. Verifique os logs acima.")
            return
        
        # 5. Calcular métricas
        print("\n📊 PASSO 5: Calculando métricas detalhadas...")
        if not self.run_command(
            'python src/calculate_metrics.py',
            '5. Cálculo de métricas'
        ):
            print("\n⚠️  Falha nas métricas. Mas processamento foi concluído.")
        
        # 6. Gerar dashboard
        print("\n🎨 PASSO 6: Gerando dashboard...")
        if not self.run_command(
            'python generate_html_report.py',
            '6. Geração de dashboard'
        ):
            print("\n⚠️  Falha no dashboard. Mas dados estão em results.csv")
        
        # Resumo final
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumo final"""
        print("\n" + "="*70)
        print("📋 RESUMO DO WORKFLOW")
        print("="*70)
        
        print(f"\n✅ Passos concluídos: {len(self.steps_completed)}")
        for step in self.steps_completed:
            print(f"   ✓ {step}")
        
        if self.steps_failed:
            print(f"\n❌ Passos com falha: {len(self.steps_failed)}")
            for step in self.steps_failed:
                print(f"   ✗ {step}")
        
        print("\n" + "="*70)
        print("📁 ARQUIVOS GERADOS")
        print("="*70)
        
        files_to_check = [
            ('results.csv', 'Resultados do processamento'),
            ('analysis_results/metrics_summary.txt', 'Resumo de métricas'),
            ('analysis_results/dashboard_report.html', 'Dashboard HTML'),
            ('analysis_results/plots/accuracy_by_speakers.png', 'Gráfico de acurácia')
        ]
        
        for filepath, description in files_to_check:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"✅ {filepath} ({size/1024:.1f} KB)")
            else:
                print(f"❌ {filepath} - Não encontrado")
        
        print("\n" + "="*70)
        print("🎯 PRÓXIMOS PASSOS")
        print("="*70)
        print("1. Abra: analysis_results/dashboard_report.html")
        print("2. Ou inicie: python web_dashboard.py")
        print("3. Verifique: results.csv para dados brutos")
        print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Workflow completo automatizado')
    parser.add_argument('--skip-download', action='store_true',
                        help='Pular download (usar atores já baixados)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Pular criação de embeddings')
    parser.add_argument('--skip-mixing', action='store_true',
                        help='Pular mixagem de áudios')
    
    args = parser.parse_args()
    
    workflow = CompleteWorkflow()
    workflow.run()


if __name__ == '__main__':
    main()
