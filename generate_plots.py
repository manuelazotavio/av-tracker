#!/usr/bin/env python3
"""
Generate plots for the updated verification results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

def create_plots(csv_path, output_dir):
    """Create plots from the verification results CSV."""

    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Carregados {len(df)} registros do CSV")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Box plot of average similarity by speaker count
    plt.figure(figsize=(10, 6))
    speaker_counts = []
    similarities = []

    for _, row in df.iterrows():
        file_path = row['file_path'].lower()
        if 'mix_2_speakers' in file_path or '2actors' in file_path:
            speaker_counts.append('2 Speakers')
            similarities.append(row['avg_similarity'])
        elif 'mix_3_speakers' in file_path or '3actors' in file_path:
            speaker_counts.append('3 Speakers')
            similarities.append(row['avg_similarity'])
        elif 'mix_4_speakers' in file_path or '4actors' in file_path:
            speaker_counts.append('4 Speakers')
            similarities.append(row['avg_similarity'])
        elif 'mix_5_speakers' in file_path or '5actors' in file_path:
            speaker_counts.append('5 Speakers')
            similarities.append(row['avg_similarity'])

    if speaker_counts and similarities:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=speaker_counts, y=similarities)
        plt.title('Distribuição da Similaridade Média por Número de Speakers')
        plt.xlabel('Número de Speakers')
        plt.ylabel('Similaridade Média')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similaridade_por_speakers_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Bar plot of identification rates by speaker count
    categories = defaultdict(list)
    for _, row in df.iterrows():
        file_path = row['file_path'].lower()
        if 'mix_2_speakers' in file_path or '2actors' in file_path:
            categories['2 Speakers'].append(row)
        elif 'mix_3_speakers' in file_path or '3actors' in file_path:
            categories['3 Speakers'].append(row)
        elif 'mix_4_speakers' in file_path or '4actors' in file_path:
            categories['4 Speakers'].append(row)
        elif 'mix_5_speakers' in file_path or '5actors' in file_path:
            categories['5 Speakers'].append(row)

    speaker_labels = []
    identification_rates = []

    for label in ['2 Speakers', '3 Speakers', '4 Speakers', '5 Speakers']:
        if label in categories:
            rows = categories[label]
            total_chunks = sum(r['chunks_processed'] for r in rows)
            total_identified = sum(r['identified_count'] for r in rows)
            rate = total_identified / total_chunks if total_chunks > 0 else 0
            speaker_labels.append(label)
            identification_rates.append(rate)

    if speaker_labels and identification_rates:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(speaker_labels, identification_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Taxa de Identificação por Número de Speakers')
        plt.xlabel('Número de Speakers')
        plt.ylabel('Taxa de Identificação')
        plt.ylim(0, 0.3)  # Ajustado baseado nos dados
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, rate in zip(bars, identification_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'taxa_identificacao_por_speakers.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Scatter plot: chunks processed vs identification rate
    plt.figure(figsize=(10, 6))

    chunks_list = []
    rates_list = []
    colors = []

    for _, row in df.iterrows():
        chunks = row['chunks_processed']
        identified = row['identified_count']
        rate = identified / chunks if chunks > 0 else 0

        chunks_list.append(chunks)
        rates_list.append(rate)

        file_path = row['file_path'].lower()
        if 'mix_2_speakers' in file_path or '2actors' in file_path:
            colors.append('#1f77b4')  # blue
        elif 'mix_3_speakers' in file_path or '3actors' in file_path:
            colors.append('#ff7f0e')  # orange
        elif 'mix_4_speakers' in file_path or '4actors' in file_path:
            colors.append('#2ca02c')  # green
        elif 'mix_5_speakers' in file_path or '5actors' in file_path:
            colors.append('#d62728')  # red
        else:
            colors.append('#9467bd')  # purple

    plt.scatter(chunks_list, rates_list, c=colors, alpha=0.7, s=50)
    plt.title('Taxa de Identificação vs Número de Chunks Processados')
    plt.xlabel('Chunks Processados')
    plt.ylabel('Taxa de Identificação')
    plt.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='2 Speakers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label='3 Speakers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label='4 Speakers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='5 Speakers')
    ]
    plt.legend(handles=legend_elements, title='Número de Speakers')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_chunks_vs_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Histogram of average similarities
    plt.figure(figsize=(10, 6))
    plt.hist(df['avg_similarity'], bins=20, alpha=0.7, color='#4c72b0', edgecolor='black')
    plt.title('Distribuição das Similaridades Médias')
    plt.xlabel('Similaridade Média')
    plt.ylabel('Frequência')
    plt.grid(True, alpha=0.3, axis='y')

    # Add mean line
    mean_sim = df['avg_similarity'].mean()
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_sim:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histograma_similaridades.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Pie chart of identification success
    total_identified = df['identified_count'].sum()
    total_unknown = df['unknown_count'].sum()

    if total_identified + total_unknown > 0:
        plt.figure(figsize=(8, 8))
        labels = ['Identificados', 'Não Identificados']
        sizes = [total_identified, total_unknown]
        colors = ['#66c2a5', '#fc8d62']
        explode = (0.1, 0)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.title('Proporção Total de Identificações')
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pizza_identificacoes.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Gráficos salvos em: {output_dir}")
    print("Gráficos gerados:")
    print("1. similaridade_por_speakers_boxplot.png - Box plot da similaridade por speakers")
    print("2. taxa_identificacao_por_speakers.png - Barras da taxa de identificação")
    print("3. scatter_chunks_vs_rate.png - Scatter plot chunks vs taxa")
    print("4. histograma_similaridades.png - Histograma das similaridades")
    print("5. pizza_identificacoes.png - Pizza chart das identificações")

if __name__ == "__main__":
    csv_path = "analysis_results/detailed_results_with_females.csv"
    output_dir = "analysis_results/plots"
    create_plots(csv_path, output_dir)