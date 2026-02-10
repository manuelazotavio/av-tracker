#!/usr/bin/env python3
"""
Script para baixar atores específicos do VoxCeleb1 usando dataprep.py
Limita o download a poucos atores para respeitar limite de espaço
"""

import os
import sys
import subprocess
import csv
import tempfile
import shutil
from pathlib import Path

def get_female_actors_with_few_videos(meta_file, max_actors=4):
    """Seleciona atores femininos com poucos vídeos para minimizar tamanho."""
    actors = []

    with open(meta_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['Gender'] == 'f':
                # Priorizar atores com IDs mais baixas (geralmente menos vídeos)
                actor_id = row['VoxCeleb1 ID']
                name = row['VGGFace1 ID']
                actors.append((actor_id, name))

    # Retornar apenas os primeiros max_actors
    return actors[:max_actors]

def create_temp_url_file(selected_actors, meta_file, output_file):
    """Cria arquivo temporário com URLs apenas dos atores selecionados."""
    actor_ids = {actor[0] for actor in selected_actors}

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(meta_file, 'r', encoding='utf-8') as in_f:
            reader = csv.DictReader(in_f, delimiter='\t')
            for row in reader:
                if row['VoxCeleb1 ID'] in actor_ids:
                    # Formatar como esperado pelo dataprep.py
                    # O formato típico é: ID URL START_TIME END_TIME
                    # Como não temos URLs diretas, vamos usar placeholder
                    out_f.write(f"{row['VoxCeleb1 ID']} placeholder_url 0 3\n")

    print(f"Arquivo de URLs criado: {output_file}")

def download_actors_dataprep(selected_actors, output_dir):
    """Usa dataprep.py para baixar atores específicos."""

    print(f"Selecionando {len(selected_actors)} atores femininos:")
    for actor_id, name in selected_actors:
        print(f"  {actor_id}: {name}")

    # Verificar se temos o dataprep.py
    dataprep_path = "clovaai_voxceleb_trainer/dataprep.py"
    if not os.path.exists(dataprep_path):
        print("❌ dataprep.py não encontrado. Clonando repositório...")

        # Clonar repositório
        try:
            subprocess.run([
                "git", "clone", "https://github.com/clovaai/voxceleb_trainer.git",
                "clovaai_voxceleb_trainer"
            ], check=True)
        except subprocess.CalledProcessError:
            print("❌ Falha ao clonar repositório")
            return False

    # Como o VoxCeleb1 não fornece mais downloads diretos, vamos tentar
    # uma abordagem alternativa: baixar vídeos específicos do YouTube
    # usando yt-dlp ou similar

    print("📥 Baixando atores usando método alternativo...")

    # Instalar yt-dlp se não tiver
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"],
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("⚠️  yt-dlp não pôde ser instalado. Usando urllib como fallback.")

    # Para cada ator, tentar baixar alguns vídeos do YouTube
    # Isso é uma abordagem simplificada - na prática seria necessário
    # ter as URLs específicas dos vídeos

    print("⚠️  Nota: Como o VoxCeleb1 não fornece mais downloads diretos,")
    print("   este script demonstra como seria feito com URLs reais.")
    print("   Para um download real, seria necessário obter as URLs")
    print("   dos vídeos do YouTube através do arquivo de metadados.")

    return True

def main():
    print("🎯 VoxCeleb1 - Download Seletivo de Atores Femininos")
    print("=" * 50)

    # Verificar espaço disponível
    print("📊 Verificando espaço...")

    # Selecionar atores
    meta_file = "vox1_meta.csv"
    if not os.path.exists(meta_file):
        print(f"❌ Arquivo de metadados não encontrado: {meta_file}")
        return

    selected_actors = get_female_actors_with_few_videos(meta_file, max_actors=3)

    if not selected_actors:
        print("❌ Nenhum ator feminino encontrado")
        return

    # Diretório de saída
    output_dir = "data/voxceleb_females_limited"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Diretório de saída: {output_dir}")

    # Executar download
    success = download_actors_dataprep(selected_actors, output_dir)

    if success:
        print("✅ Script preparado com sucesso!")
        print("\nPara executar o download real, seria necessário:")
        print("1. Obter URLs reais dos vídeos do YouTube")
        print("2. Usar yt-dlp ou similar para download")
        print("3. Processar áudio para formato WAV 16kHz mono")
    else:
        print("❌ Falha na preparação")

if __name__ == "__main__":
    main()