#!/usr/bin/env python3
"""
Script para baixar atores específicos do VoxCeleb1
Uso: python download_specific_actors.py --actors id10006 id10007 --output ../data/voxceleb_selected
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import csv
from pathlib import Path

def load_actor_info(meta_path):
    """Carrega informações dos atores do VoxCeleb1."""
    actors = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            actor_id = row['VoxCeleb1 ID']
            name = row['VGGFace1 ID']
            gender = row['Gender']
            actors[actor_id] = {'name': name, 'gender': gender}
    return actors

def download_file(url, dest_path):
    """Baixa um arquivo com barra de progresso."""
    try:
        print(f"Baixando: {url}")
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgresso: {percent:.1f}%", end='', flush=True)

        print(" ✓ Download concluído")
        return True
    except Exception as e:
        print(f" ✗ Erro no download: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """Extrai arquivo zip ou tar."""
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f" ✓ Extraído para: {extract_to}")
        return True
    except Exception as e:
        print(f" ✗ Erro na extração: {e}")
        return False

def download_voxceleb_actors(actor_ids, output_dir, actors_info):
    """
    Tenta baixar atores específicos do VoxCeleb1.
    Nota: O VoxCeleb1 não permite download individual fácil,
    então este script tenta as partes disponíveis.
    """

    os.makedirs(output_dir, exist_ok=True)

    # URLs das partes do VoxCeleb1 (dev set)
    base_urls = [
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad"
    ]

    print(f"Procurando {len(actor_ids)} atores em {len(base_urls)} partes...")

    found_actors = []

    for part_url in base_urls:
        print(f"\nTentando parte: {part_url}")

        # Baixar a parte
        zip_url = f"{part_url}.zip"
        zip_path = os.path.join(output_dir, f"temp_{os.path.basename(part_url)}.zip")

        if not download_file(zip_url, zip_path):
            continue

        # Extrair temporariamente para verificar conteúdo
        temp_extract = os.path.join(output_dir, "temp_extract")
        os.makedirs(temp_extract, exist_ok=True)

        if not extract_archive(zip_path, temp_extract):
            os.remove(zip_path)
            continue

        # Verificar quais atores estão nesta parte
        wav_dir = os.path.join(temp_extract, "voxceleb1_wav")
        if os.path.exists(wav_dir):
            available_in_part = []
            for item in os.listdir(wav_dir):
                if item in actor_ids:
                    available_in_part.append(item)

            if available_in_part:
                print(f"  Atores encontrados nesta parte: {available_in_part}")

                # Mover atores desejados para o diretório final
                for actor_id in available_in_part:
                    src_dir = os.path.join(wav_dir, actor_id)
                    dst_dir = os.path.join(output_dir, actor_id)

                    if os.path.exists(src_dir):
                        import shutil
                        shutil.move(src_dir, dst_dir)
                        found_actors.append(actor_id)
                        print(f"  ✓ {actor_id} ({actors_info[actor_id]['name']}) movido")

        # Limpar arquivos temporários
        import shutil
        shutil.rmtree(temp_extract)
        os.remove(zip_path)

    # Relatório final
    print(f"\n{'='*50}")
    print("RESUMO DO DOWNLOAD:")
    print(f"Atores solicitados: {len(actor_ids)}")
    print(f"Atores encontrados: {len(found_actors)}")
    print(f"Atores faltando: {len(actor_ids) - len(found_actors)}")

    if found_actors:
        print("\nAtores baixados:")
        for actor_id in found_actors:
            info = actors_info[actor_id]
            print(f"  {actor_id}: {info['name']} ({info['gender']})")

    missing = [aid for aid in actor_ids if aid not in found_actors]
    if missing:
        print("\nAtores não encontrados (podem estar em outras partes ou test set):")
        for actor_id in missing:
            info = actors_info[actor_id]
            print(f"  {actor_id}: {info['name']} ({info['gender']})")

    return found_actors

def main():
    parser = argparse.ArgumentParser(description='Baixar atores específicos do VoxCeleb1')
    parser.add_argument('--actors', nargs='+', required=True,
                       help='IDs dos atores para baixar (ex: id10006 id10007)')
    parser.add_argument('--output', default='../data/voxceleb_selected',
                       help='Diretório de saída')
    parser.add_argument('--meta', default='../vox1_meta.csv',
                       help='Arquivo de metadados')

    args = parser.parse_args()

    # Carregar informações dos atores
    if not os.path.exists(args.meta):
        print(f"Arquivo de metadados não encontrado: {args.meta}")
        sys.exit(1)

    actors_info = load_actor_info(args.meta)

    # Verificar se os atores existem
    invalid_actors = [aid for aid in args.actors if aid not in actors_info]
    if invalid_actors:
        print(f"Atores inválidos: {invalid_actors}")
        sys.exit(1)

    # Mostrar informações dos atores
    print("Atores selecionados:")
    for actor_id in args.actors:
        info = actors_info[actor_id]
        print(f"  {actor_id}: {info['name']} ({info['gender']})")

    # Confirmar download
    response = input(f"\nBaixar {len(args.actors)} atores para {args.output}? (s/N): ")
    if response.lower() != 's':
        print("Download cancelado.")
        return

    # Executar download
    found = download_voxceleb_actors(args.actors, args.output, actors_info)

    if found:
        print(f"\n✓ {len(found)} atores baixados com sucesso!")
        print(f"Localização: {args.output}")
    else:
        print("\n✗ Nenhum ator foi encontrado nas partes disponíveis.")

if __name__ == "__main__":
    main()