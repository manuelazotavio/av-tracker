import csv
import os

# Contar linhas no arquivo
if os.path.exists('analysis_results/detailed_results_with_females.csv'):
    with open('analysis_results/detailed_results_with_females.csv', 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)
        print(f'Total de linhas no arquivo: {total_lines}')

    # Analisar distribuição de gêneros
    gender_count = {'m': 0, 'f': 0, 'unknown': 0}
    male_samples = []
    female_samples = []

    with open('analysis_results/detailed_results_with_females.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row['true_label']
            confidence = float(row['confidence'])

            # Verificar se é ID ou nome
            if true_label.startswith('id'):
                # É um ID, vamos buscar o gênero
                gender_count['unknown'] += 1
            else:
                # É um nome, pode ser feminino
                if any(female_name in true_label.lower() for female_name in ['danielle', 'alba', 'abbie', 'alison', 'amanda', 'ana', 'angela', 'caterina', 'catherine', 'cristin', 'dakota', 'dana', 'abbie']):
                    gender_count['f'] += 1
                    female_samples.append(confidence)
                else:
                    gender_count['m'] += 1
                    male_samples.append(confidence)

    print('Distribuição nos resultados:')
    print(f'  Masculino: {gender_count["m"]} amostras')
    print(f'  Feminino: {gender_count["f"]} amostras')
    print(f'  Desconhecido (IDs): {gender_count["unknown"]} amostras')

    # Calcular acurácias
    if male_samples:
        male_accuracy = sum(1 for c in male_samples if c >= 0.5) / len(male_samples)
        print(f'Acurácia masculina: {male_accuracy:.3f}')
    else:
        print('Nenhuma amostra masculina encontrada')

    if female_samples:
        female_accuracy = sum(1 for c in female_samples if c >= 0.5) / len(female_samples)
        print(f'Acurácia feminina: {female_accuracy:.3f}')
    else:
        print('Nenhuma amostra feminina encontrada')

    print(f'Diferença: {abs(male_accuracy - female_accuracy):.3f}' if male_samples and female_samples else 'Não foi possível calcular diferença')