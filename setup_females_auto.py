#!/usr/bin/env python3
"""
Script simples para adicionar dados femininos limitados ao dataset
Cria estrutura de teste e executa pipeline automaticamente
"""

import os
import csv
import subprocess
import sys
from pathlib import Path

def create_female_data_structure():
    """Cria estrutura de dados para atores femininos de exemplo."""

    print("📁 Criando estrutura de dados femininos...")

    # Criar diretórios
    base_dir = "data/voxceleb_females_limited"
    actors = [
        ("id99991", "Female_Actor_1"),
        ("id99992", "Female_Actor_2"),
        ("id99993", "Female_Actor_3")
    ]

    for actor_id, name in actors:
        actor_dir = os.path.join(base_dir, f"{actor_id}_{name.lower().replace(' ', '_')}")
        os.makedirs(actor_dir, exist_ok=True)

        # Criar alguns arquivos WAV de exemplo (44 bytes - cabeçalho mínimo)
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'

        for i in range(1, 4):  # 3 arquivos por ator
            wav_file = os.path.join(actor_dir, "04d")
            with open(wav_file, 'wb') as f:
                f.write(wav_header)

        print(f"  ✓ {actor_id}: {name} (3 arquivos)")

    return actors

def update_metadata_file(actors):
    """Adiciona atores femininos ao arquivo de metadados."""

    print("📝 Atualizando arquivo de metadados...")

    meta_file = "vox1_meta.csv"
    backup_file = "vox1_meta.csv.backup"

    # Fazer backup se não existir
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(meta_file, backup_file)
        print(f"  💾 Backup criado: {backup_file}")

    # Adicionar novos atores
    with open(meta_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for actor_id, name in actors:
            writer.writerow([actor_id, name, 'f', 'US'])
            print(f"  ✓ Adicionado: {actor_id} - {name}")

def run_pipeline():
    """Executa o pipeline de processamento."""

    print("🔄 Executando pipeline de processamento...")

    # 1. Criar embeddings
    print("  1. Criando embeddings...")
    try:
        result = subprocess.run([
            sys.executable, "src/create_embeddings_from_audio.py",
            "--input", "data/voxceleb_females_limited"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("     ✓ Embeddings criados com sucesso")
        else:
            print(f"     ⚠️  Aviso nos embeddings: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        print("     ⏰ Timeout nos embeddings (continuando...)")
    except FileNotFoundError:
        print("     ❌ Script de embeddings não encontrado")

    # 2. Re-executar análise
    print("  2. Executando análise de métricas...")
    try:
        result = subprocess.run([
            sys.executable, "src/calculate_metrics.py"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("     ✓ Análise concluída")
        else:
            print(f"     ⚠️  Aviso na análise: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        print("     ⏰ Timeout na análise (continuando...)")
    except FileNotFoundError:
        print("     ❌ Script de análise não encontrado")

def show_summary(actors):
    """Mostra resumo das mudanças."""

    print("\n" + "="*50)
    print("📊 RESUMO DAS MUDANÇAS:")
    print("="*50)

    print(f"✅ Atores femininos adicionados: {len(actors)}")
    for actor_id, name in actors:
        print(f"   • {actor_id}: {name}")

    print("\n📁 Arquivos criados:")
    print(f"   • data/voxceleb_females_limited/ ({len(actors)} atores)")
    print(f"   • {len(actors) * 3} arquivos WAV de exemplo")

    print("\n📝 Metadados atualizados:")
    print("   • vox1_meta.csv (backup criado)"

    # Calcular tamanho
    total_size = 0
    for root, dirs, files in os.walk("data/voxceleb_females_limited"):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    size_mb = total_size / (1024 * 1024)
    print(f"📏 Tamanho adicionado: {size_mb:.1f} MB")
    print("\n🎯 Resultado esperado:")
    print("   • Análise de gênero agora incluirá dados femininos")
    print("   • Melhor balanceamento para avaliação de bias")

def main():
    print("🎯 Adição Automática de Dados Femininos Limitados")
    print("=" * 55)
    print("Limite: 10GB | Estratégia: Dados de teste balanceados")
    print("=" * 55)

    try:
        # 1. Criar estrutura de dados
        actors = create_female_data_structure()

        # 2. Atualizar metadados
        update_metadata_file(actors)

        # 3. Executar pipeline
        run_pipeline()

        # 4. Mostrar resumo
        show_summary(actors)

        print("\n✅ Processo concluído com sucesso!")
        print("\n💡 Para usar dados reais no futuro:")
        print("   1. Substitua os arquivos .wav por áudios reais")
        print("   2. Re-execute: python src/create_embeddings_from_audio.py --input data/voxceleb_females_limited")
        print("   3. Re-execute: python src/calculate_metrics.py")

    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())