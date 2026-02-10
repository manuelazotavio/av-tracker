"""
Análise detalhada dos resultados completos com transcrições e métricas
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(csv_file):
    """Analisa os resultados completos"""
    
    if not Path(csv_file).exists():
        print(f"Arquivo não encontrado: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    print("="*70)
    print("ANÁLISE COMPLETA DE RESULTADOS")
    print("="*70)
    print(f"\nTotal de mixes processados: {len(df)}")
    
    # Estatísticas gerais
    print("\n" + "="*70)
    print("MÉTRICAS GERAIS")
    print("="*70)
    print(f"Accuracy média:   {df['accuracy'].mean():.1%} ± {df['accuracy'].std():.1%}")
    print(f"Precision média:  {df['precision'].mean():.1%} ± {df['precision'].std():.1%}")
    print(f"Recall média:     {df['recall'].mean():.1%} ± {df['recall'].std():.1%}")
    print(f"F1-Score média:   {df['f1_score'].mean():.1%} ± {df['f1_score'].std():.1%}")
    print(f"Segmentos médios: {df['total_segments'].mean():.1f} ± {df['total_segments'].std():.1f}")
    
    # Por número de speakers
    df['num_speakers'] = df['audio_file'].str.extract(r'mix_(\d+)_speakers')[0].astype(int)
    
    print("\n" + "="*70)
    print("MÉTRICAS POR NÚMERO DE SPEAKERS")
    print("="*70)
    
    for n_speakers in sorted(df['num_speakers'].unique()):
        subset = df[df['num_speakers'] == n_speakers]
        print(f"\n{n_speakers} Speakers ({len(subset)} mixes):")
        print(f"  Accuracy:  {subset['accuracy'].mean():.1%} ± {subset['accuracy'].std():.1%}")
        print(f"  Precision: {subset['precision'].mean():.1%} ± {subset['precision'].std():.1%}")
        print(f"  Recall:    {subset['recall'].mean():.1%} ± {subset['recall'].std():.1%}")
        print(f"  F1-Score:  {subset['f1_score'].mean():.1%} ± {subset['f1_score'].std():.1%}")
        print(f"  Segmentos: {subset['total_segments'].mean():.1f} ± {subset['total_segments'].std():.1f}")
    
    # Distribuição de accuracy
    print("\n" + "="*70)
    print("DISTRIBUIÇÃO DE ACCURACY")
    print("="*70)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df['accuracy_bin'] = pd.cut(df['accuracy'], bins=bins, labels=labels)
    print(df['accuracy_bin'].value_counts().sort_index())
    
    # Casos de sucesso vs falha
    print("\n" + "="*70)
    print("CASOS EXTREMOS")
    print("="*70)
    
    print("\n5 MELHORES (maior accuracy):")
    best = df.nlargest(5, 'accuracy')[['audio_file', 'accuracy', 'precision', 'recall', 'f1_score']]
    for idx, row in best.iterrows():
        print(f"  {row['audio_file']:40s} | Acc: {row['accuracy']:.1%} | F1: {row['f1_score']:.1%}")
    
    print("\n5 PIORES (menor accuracy):")
    worst = df.nsmallest(5, 'accuracy')[['audio_file', 'accuracy', 'precision', 'recall', 'f1_score']]
    for idx, row in worst.iterrows():
        print(f"  {row['audio_file']:40s} | Acc: {row['accuracy']:.1%} | F1: {row['f1_score']:.1%}")
    
    # Speakers mais comuns
    print("\n" + "="*70)
    print("SPEAKERS MAIS IDENTIFICADOS")
    print("="*70)
    
    all_speakers = []
    for speakers_str in df['identified_speakers'].dropna():
        if speakers_str and speakers_str != '':
            all_speakers.extend(speakers_str.split('|'))
    
    from collections import Counter
    speaker_counts = Counter(all_speakers)
    
    print("\nTop 10 speakers mais frequentemente identificados:")
    for speaker, count in speaker_counts.most_common(10):
        print(f"  {speaker:30s}: {count} vezes")
    
    # Correlações
    print("\n" + "="*70)
    print("CORRELAÇÕES")
    print("="*70)
    corr = df[['accuracy', 'precision', 'recall', 'f1_score', 'total_segments', 'num_speakers']].corr()
    print("\nCorrelação entre métricas:")
    print(corr.round(3))
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)
    
    # Gerar gráficos
    try:
        generate_plots(df)
    except Exception as e:
        print(f"\nErro ao gerar gráficos: {e}")

def generate_plots(df):
    """Gera gráficos de análise"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análise de Resultados - Mix Processing', fontsize=16, fontweight='bold')
    
    # 1. Distribuição de Accuracy
    ax = axes[0, 0]
    ax.hist(df['accuracy'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(df['accuracy'].mean(), color='red', linestyle='--', linewidth=2, label=f"Média: {df['accuracy'].mean():.1%}")
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Métricas por número de speakers
    ax = axes[0, 1]
    metrics_by_speakers = df.groupby('num_speakers')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    metrics_by_speakers.plot(kind='bar', ax=ax)
    ax.set_xlabel('Número de Speakers')
    ax.set_ylabel('Score')
    ax.set_title('Métricas por Número de Speakers')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # 3. Boxplot de accuracy por speakers
    ax = axes[0, 2]
    df.boxplot(column='accuracy', by='num_speakers', ax=ax)
    ax.set_xlabel('Número de Speakers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Distribuição de Accuracy por Speakers')
    plt.sca(ax)
    plt.xticks(rotation=0)
    
    # 4. F1-Score vs Accuracy
    ax = axes[1, 0]
    scatter = ax.scatter(df['accuracy'], df['f1_score'], c=df['num_speakers'], 
                         cmap='viridis', alpha=0.6, s=50)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score vs Accuracy')
    plt.colorbar(scatter, ax=ax, label='Num Speakers')
    ax.grid(alpha=0.3)
    
    # 5. Número de segmentos por speakers
    ax = axes[1, 1]
    df.groupby('num_speakers')['total_segments'].mean().plot(kind='bar', ax=ax, color='coral')
    ax.set_xlabel('Número de Speakers')
    ax.set_ylabel('Número Médio de Segmentos')
    ax.set_title('Segmentos Médios por Número de Speakers')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # 6. Tabela de métricas
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = []
    for n_speakers in sorted(df['num_speakers'].unique()):
        subset = df[df['num_speakers'] == n_speakers]
        summary_data.append([
            f"{n_speakers} speakers",
            f"{subset['accuracy'].mean():.1%}",
            f"{subset['precision'].mean():.1%}",
            f"{subset['recall'].mean():.1%}",
            f"{subset['f1_score'].mean():.1%}"
        ])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Tipo', 'Accuracy', 'Precision', 'Recall', 'F1'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilizar header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
    
    ax.set_title('Resumo de Métricas', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Salvar gráficos
    output_file = Path('data') / 'mix_analysis_complete.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nGráficos salvos em: {output_file}")
    
    plt.close()

if __name__ == "__main__":
    csv_file = "mix_processing_results_complete.csv"
    analyze_results(csv_file)
