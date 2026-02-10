import csv
import os

# Carregar mapeamento correto de nomes para gêneros
name_to_gender = {}
id_to_gender = {}

with open('vox1_meta.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        vox_id = row['VoxCeleb1 ID']
        vgg_id = row['VGGFace1 ID']
        name = row.get('VGGFace1 ID', '').strip()  # O nome está na coluna VGGFace1 ID
        gender = row['Gender']

        # Mapear ID para gênero
        id_to_gender[vox_id] = gender

        # Mapear nome para gênero (se existir)
        if name and name != vox_id:  # Se o nome for diferente do ID
            name_to_gender[name] = gender

print(f"Total de IDs mapeados: {len(id_to_gender)}")
print(f"Total de nomes mapeados: {len(name_to_gender)}")

# Verificar Danielle_Panabaker
print(f"Gênero de Danielle_Panabaker: {name_to_gender.get('Danielle_Panabaker', 'NÃO ENCONTRADO')}")
print(f"Gênero de id10196: {id_to_gender.get('id10196', 'NÃO ENCONTRADO')}")

# Verificar alguns IDs masculinos
male_ids = ['id10210', 'id10213', 'id10194']
for mid in male_ids:
    print(f"Gênero de {mid}: {id_to_gender.get(mid, 'NÃO ENCONTRADO')}")

# Agora analisar o arquivo de resultados
gender_count = {'m': 0, 'f': 0, 'unknown': 0}
male_confs = []
female_confs = []

if os.path.exists('analysis_results/detailed_results_with_females.csv'):
    with open('analysis_results/detailed_results_with_females.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row['true_label']
            confidence = float(row['confidence'])

            # Tentar mapear o gênero
            gender = None
            if true_label.startswith('id'):
                gender = id_to_gender.get(true_label)
            else:
                gender = name_to_gender.get(true_label)

            if gender == 'm':
                gender_count['m'] += 1
                male_confs.append(confidence)
            elif gender == 'f':
                gender_count['f'] += 1
                female_confs.append(confidence)
            else:
                gender_count['unknown'] += 1

print("
Análise corrigida dos resultados:")
print(f"  Masculino: {gender_count['m']} amostras")
print(f"  Feminino: {gender_count['f']} amostras")
print(f"  Desconhecido: {gender_count['unknown']} amostras")

# Calcular acurácias
if male_confs:
    male_correct = sum(1 for c in male_confs if c >= 0.5)
    male_accuracy = male_correct / len(male_confs)
    print(f"Acurácia masculina: {male_accuracy:.3f} ({male_correct}/{len(male_confs)})")

if female_confs:
    female_correct = sum(1 for c in female_confs if c >= 0.5)
    female_accuracy = female_correct / len(female_confs)
    print(f"Acurácia feminina: {female_accuracy:.3f} ({female_correct}/{len(female_confs)})")

if male_confs and female_confs:
    print(f"Diferença: {abs(male_accuracy - female_accuracy):.3f}")