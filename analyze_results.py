#!/usr/bin/env python3
"""
Analyze the detailed results CSV and generate statistics by speaker count.
"""

import pandas as pd
import os
from collections import defaultdict

def analyze_results(csv_path):
    """Analyze the verification results and generate statistics."""

    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Carregados {len(df)} registros do CSV")

    # Categorize by speaker count in filename
    categories = defaultdict(list)

    for _, row in df.iterrows():
        file_path = row['file_path'].lower()

        if 'mix_2_speakers' in file_path or '2actors' in file_path:
            categories[2].append(row)
        elif 'mix_3_speakers' in file_path or '3actors' in file_path:
            categories[3].append(row)
        elif 'mix_4_speakers' in file_path or '4actors' in file_path:
            categories[4].append(row)
        elif 'mix_5_speakers' in file_path or '5actors' in file_path:
            categories[5].append(row)
        else:
            categories['other'].append(row)

    print("\n=== ANÁLISE POR NÚMERO DE SPEAKERS ===")

    for num_speakers in sorted([k for k in categories.keys() if k != 'other'], key=lambda x: (isinstance(x, str), x)):
        rows = categories[num_speakers]
        if not rows:
            continue

        print(f"\n{num_speakers} SPEAKERS:")
        print(f"  Arquivos analisados: {len(rows)}")

        # Calculate averages
        avg_chunks = sum(r['chunks_processed'] for r in rows) / len(rows)
        avg_identified = sum(r['identified_count'] for r in rows) / len(rows)
        avg_unknown = sum(r['unknown_count'] for r in rows) / len(rows)
        avg_similarity = sum(r['avg_similarity'] for r in rows) / len(rows)

        # Calculate identification rate
        total_chunks = sum(r['chunks_processed'] for r in rows)
        total_identified = sum(r['identified_count'] for r in rows)
        identification_rate = total_identified / total_chunks if total_chunks > 0 else 0

        print(".2f")
        print(".2f")
        print(".2f")
        print(".3f")
        print(".3f")

    print("\n=== ESTATÍSTICAS GERAIS ===")
    total_files = len(df)
    total_chunks = df['chunks_processed'].sum()
    total_identified = df['identified_count'].sum()
    overall_rate = total_identified / total_chunks if total_chunks > 0 else 0

    print(f"Total de arquivos: {total_files}")
    print(f"Total de chunks processados: {total_chunks}")
    print(f"Total de identificações: {total_identified}")
    print(".3f")

    # Files with some identification
    files_with_id = len(df[df['identified_count'] > 0])
    print(f"Arquivos com pelo menos uma identificação: {files_with_id} ({files_with_id/total_files*100:.1f}%)")

    # High confidence identifications (avg_similarity > 0.5)
    high_conf_files = len(df[df['avg_similarity'] > 0.5])
    print(f"Arquivos com alta confiança (>0.5): {high_conf_files} ({high_conf_files/total_files*100:.1f}%)")

if __name__ == "__main__":
    csv_path = "analysis_results/detailed_results_with_females.csv"
    analyze_results(csv_path)