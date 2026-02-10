#!/usr/bin/env python3
"""
Script de workflow automático:
1. Verificar status
2. Baixar novos áudios
3. Processar só o novo
4. Atualizar dashboard
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Executa comando e mostra resultado"""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - OK")
    else:
        print(f"❌ {description} - ERRO")
    
    return result.returncode == 0


def main():
    print("\n" + "="*70)
    print("WORKFLOW AUTOMÁTICO: DOWNLOAD → PROCESSAMENTO → DASHBOARD")
    print("="*70)
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-limit', type=int, default=50,
                        help='Quantos áudios baixar (padrão: 50)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Pular download de novos áudios')
    parser.add_argument('--skip-process', action='store_true',
                        help='Pular processamento')
    parser.add_argument('--skip-dashboard', action='store_true',
                        help='Pular regeneração do dashboard')
    parser.add_argument('--full-process', action='store_true',
                        help='Reprocessar TUDO (deleta tracking)')
    
    args = parser.parse_args()
    
    success_count = 0
    fail_count = 0
    
    # PASSO 1: Verificar status
    print(f"\n{'='*70}")
    print("📊 PASSO 1: VERIFICAR STATUS")
    print(f"{'='*70}")
    run_command('python processing_tracker.py --stats', 'Verificando status')
    
    # PASSO 2: Baixar novos áudios
    if not args.skip_download:
        print(f"\n{'='*70}")
        print("📥 PASSO 2: BAIXAR NOVOS ÁUDIOS DO VOXCELEB2")
        print(f"{'='*70}")
        
        cmd = f'python download_voxceleb2.py --limit {args.download_limit}'
        if run_command(cmd, f'Baixando até {args.download_limit} áudios'):
            success_count += 1
        else:
            fail_count += 1
        
        # Atualizar contagem de arquivos
        run_command('python processing_tracker.py --update-sources',
                   'Atualizando contagem de arquivos')
    else:
        print("\n⏭️  Pulando download (--skip-download)")
    
    # PASSO 3: Processar arquivos
    if not args.skip_process:
        print(f"\n{'='*70}")
        print("⚙️  PASSO 3: PROCESSAR ARQUIVOS")
        print(f"{'='*70}")
        
        if args.full_process:
            print("🔄 Deletando histórico para reprocessamento completo...")
            if os.path.exists('processing_tracker.json'):
                os.remove('processing_tracker.json')
                print("✅ Histórico deletado")
        
        cmd = 'python process_with_tracker.py'
        if run_command(cmd, 'Processando arquivos'):
            success_count += 1
        else:
            fail_count += 1
    else:
        print("\n⏭️  Pulando processamento (--skip-process)")
    
    # PASSO 4: Regenerar dashboard
    if not args.skip_dashboard:
        print(f"\n{'='*70}")
        print("📊 PASSO 4: REGENERAR DASHBOARD")
        print(f"{'='*70}")
        
        cmd = 'python generate_html_report.py'
        if run_command(cmd, 'Regenerando dashboard HTML'):
            success_count += 1
        else:
            fail_count += 1
    else:
        print("\n⏭️  Pulando dashboard (--skip-dashboard)")
    
    # RESUMO FINAL
    print(f"\n{'='*70}")
    print("📋 RESUMO DO WORKFLOW")
    print(f"{'='*70}")
    print(f"✅ Passos bem-sucedidos: {success_count}")
    print(f"❌ Passos com erro: {fail_count}")
    
    if fail_count == 0:
        print(f"\n🎉 Workflow concluído com sucesso!")
        print(f"\n📊 Dashboard atualizado:")
        print(f"   Arquivo: analysis_results/dashboard_report.html")
        print(f"   Abra no navegador para visualizar")
    else:
        print(f"\n⚠️  Alguns passos falharam. Verifique os erros acima.")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
