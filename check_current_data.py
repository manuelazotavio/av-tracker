#!/usr/bin/env python3
import os
import csv

# Verificar atores locais
local_actors = []
actor_dir = 'vox_100_atores/wav'
if os.path.exists(actor_dir):
    for item in os.listdir(actor_dir):
        if item.startswith('id'):
            local_actors.append(item)

print(f'Atores locais encontrados: {len(local_actors)}')
print('Primeiros 10:', local_actors[:10])

# Verificar gêneros dos atores locais
female_actors = []
male_actors = []

with open('vox1_meta.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        actor_id = row['VoxCeleb1 ID']
        if actor_id in local_actors:
            if row['Gender'] == 'f':
                female_actors.append(actor_id)
            else:
                male_actors.append(actor_id)

print(f'\nDistribuição atual:')
print(f'  Mulheres: {len(female_actors)}')
print(f'  Homens: {len(male_actors)}')
print(f'  Total: {len(local_actors)}')

# Calcular tamanho dos dados atuais
def get_folder_size(folder):
    total_size = 0
    for path, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size

data_size = get_folder_size('data')
vox_size = get_folder_size('vox_100_atores')
total_size = data_size + vox_size
total_gb = total_size / (1024**3)

print(f'\nTamanho dos dados:')
print(f'  data/: {data_size / (1024**3):.2f} GB')
print(f'  vox_100_atores/: {vox_size / (1024**3):.2f} GB')
print(f'  Total: {total_gb:.2f} GB')
print(f'  Espaço restante: {10 - total_gb:.2f} GB')