import csv
import os

# Verificar distribuição de gêneros nos dados
gender_count = {'m': 0, 'f': 0, 'unknown': 0}

if os.path.exists('analysis_results/detailed_results_with_females.csv'):
    with open('analysis_results/detailed_results_with_females.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row['true_label']
            # Verificar se é ID ou nome
            if true_label.startswith('id'):
                # É um ID, vamos buscar o gênero
                gender_count['unknown'] += 1
            else:
                # É um nome, pode ser feminino
                if any(female_name in true_label.lower() for female_name in ['danielle', 'alba', 'abbie', 'alison', 'amanda', 'ana', 'angela', 'caterina', 'catherine', 'cristin', 'dakota', 'dana', 'abbie']):
                    gender_count['f'] += 1
                else:
                    gender_count['m'] += 1

print('Distribuição nos resultados:')
print(f'  Masculino: {gender_count["m"]}')
print(f'  Feminino: {gender_count["f"]}')
print(f'  Desconhecido (IDs): {gender_count["unknown"]}')

# Também verificar o mapeamento de gêneros
gender_map = {}
with open('vox1_meta.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        actor_id = row['VoxCeleb1 ID']
        gender = row['Gender']
        gender_map[actor_id] = gender

print(f'\nTotal de atores no mapeamento: {len(gender_map)}')

# Contar gêneros no mapeamento
male_count = sum(1 for g in gender_map.values() if g == 'm')
female_count = sum(1 for g in gender_map.values() if g == 'f')

print(f'Masculino no mapeamento: {male_count}')
print(f'Feminino no mapeamento: {female_count}')