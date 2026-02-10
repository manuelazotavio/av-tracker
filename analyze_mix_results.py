import os
import pandas as pd
import numpy as np

BASE_DIR = os.getcwd()
CSV_FILE = os.path.join(BASE_DIR, "mix_processing_results.csv")

# Carregar CSV
df = pd.read_csv(CSV_FILE)

# Estatísticas por número de speakers
print(f"\n{'='*60}")
print("📊 ESTATÍSTICAS DE PROCESSAMENTO DOS MIXES")
print(f"{'='*60}")

print(f"\nTotal de arquivos: {len(df)}")
print(f"Accuracy média geral: {df['accuracy'].mean():.1%}")
print(f"Arquivos com 100% accuracy: {(df['accuracy'] == 1.0).sum()}/{len(df)}")

print(f"\n{'RESUMO POR NÚMERO DE SPEAKERS':^60}")
print(f"{'-'*60}")
print(f"{'Speakers':<15} {'Arquivos':<15} {'Accuracy':<15} {'Total Atores':<15}")
print(f"{'-'*60}")

for num_speakers in sorted(df['num_speakers'].unique()):
    subset = df[df['num_speakers'] == num_speakers]
    count = len(subset)
    accuracy = subset['accuracy'].mean()
    
    # Contar total de atores únicos
    all_speakers = []
    for speakers_str in subset['ground_truth_speakers']:
        all_speakers.extend(speakers_str.split('|'))
    unique_speakers = len(set(all_speakers))
    
    print(f"{num_speakers:<15} {count:<15} {accuracy:.1%}{'  ':<11} {unique_speakers:<15}")

print(f"{'-'*60}")

print(f"\n{'DETALHE POR GRUPO':^60}")
print(f"{'-'*60}")

for num_speakers in sorted(df['num_speakers'].unique()):
    subset = df[df['num_speakers'] == num_speakers]
    print(f"\n{num_speakers} Speakers ({len(subset)} arquivos):")
    
    # Mostrar alguns exemplos
    for idx, (audio_file, speakers, accuracy) in enumerate(subset[['audio_file', 'ground_truth_speakers', 'accuracy']].values[:3]):
        speaker_list = speakers.replace('|', ', ')
        print(f"  • {audio_file}: {speaker_list}")
    
    if len(subset) > 3:
        print(f"  ... e mais {len(subset) - 3} arquivos")

print(f"\n{'='*60}")
print(f"✅ CSV salvo em: {CSV_FILE}")
print(f"{'='*60}\n")
