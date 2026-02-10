import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def load_speaker_mapping(meta_path):
    """Load mapping from VoxCeleb1 ID to VGGFace1 ID and gender."""
    mapping = {}
    gender_map = {}
    name_to_gender = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            vox_id = row['VoxCeleb1 ID']
            vgg_id = row['VGGFace1 ID']
            gender = row['Gender']
            mapping[vox_id] = vgg_id
            gender_map[vgg_id] = gender
            # Also map VoxCeleb1 ID to gender
            gender_map[vox_id] = gender
            # If VGGFace1 ID is actually a name, map it too
            if vgg_id and vgg_id != vox_id:
                name_to_gender[vgg_id] = gender
    return mapping, gender_map

def calculate_metrics(predictions, true_labels, confidences):
    """
    Calcula várias métricas de performance.
    """
    metrics = {}
    
    # Acurácia Global
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    metrics['global_accuracy'] = correct / len(predictions) if predictions else 0
    
    # FAR e FRR (simplificado - assumindo que Unknown é rejeição)
    far_count = sum(1 for p, t in zip(predictions, true_labels) if p != "Unknown" and p != t)
    frr_count = sum(1 for p, t in zip(predictions, true_labels) if p == "Unknown" and t != "Unknown")
    total_impostor_attempts = sum(1 for p in predictions if p != "Unknown")
    total_legitimate_attempts = len(predictions)
    
    metrics['far'] = far_count / total_impostor_attempts if total_impostor_attempts > 0 else 0
    metrics['frr'] = frr_count / total_legitimate_attempts if total_legitimate_attempts > 0 else 0
    
    # Score médio de confiança
    correct_confs = [c for p, t, c in zip(predictions, true_labels, confidences) if p == t and not np.isnan(c)]
    incorrect_confs = [c for p, t, c in zip(predictions, true_labels, confidences) if p != t and not np.isnan(c)]
    
    metrics['avg_confidence_correct'] = np.mean(correct_confs) if correct_confs else 0
    metrics['avg_confidence_incorrect'] = np.mean(incorrect_confs) if incorrect_confs else 0
    
    # Top-N acurácia (simplificado - aqui seria 1, mas podemos calcular precision/recall)
    unique_labels = list(set(true_labels))
    report = classification_report(true_labels, predictions, labels=unique_labels, output_dict=True, zero_division=0)
    metrics['precision_macro'] = report['macro avg']['precision']
    metrics['recall_macro'] = report['macro avg']['recall']
    metrics['f1_macro'] = report['macro avg']['f1-score']
    
    return metrics

def analyze_by_threshold(predictions, true_labels, confidences, thresholds):
    """Analisa performance com diferentes thresholds."""
    results = []
    for thresh in thresholds:
        # Simular threshold: rejeitar se confiança < thresh
        filtered_predictions = []
        filtered_true = []
        for p, t, c in zip(predictions, true_labels, confidences):
            if c >= thresh:
                filtered_predictions.append(p)
                filtered_true.append(t)
            else:
                filtered_predictions.append("Unknown")
                filtered_true.append(t)
        
        metrics = calculate_metrics(filtered_predictions, filtered_true, [c for c in confidences if c >= thresh] + [0] * (len(filtered_predictions) - sum(1 for c in confidences if c >= thresh)))
        metrics['threshold'] = thresh
        results.append(metrics)
    return results

def analyze_by_num_speakers(df, speaker_mapping):
    """Analisa performance por número de speakers."""
    results = {}
    
    # Mapear arquivos para num_speakers
    file_to_num = {
        'mix_2_speakers.wav': 2,
        'mix_3_speakers.wav': 3,
        'mix_4_speakers.wav': 4,
        'mix_5_speakers.wav': 5
    }
    
    # Adicionar coluna num_speakers baseada no arquivo
    df['num_speakers'] = df['file'].map(lambda x: file_to_num.get(os.path.basename(x), 0))
    
    for num in [2, 3, 4, 5]:
        subset = df[df['num_speakers'] == num]
        if not subset.empty:
            preds = subset['prediction_mapped'].tolist()
            trues = subset['true_label'].tolist()
            confs = subset['confidence'].tolist()
            results[num] = calculate_metrics(preds, trues, confs)
    
    return results

def analyze_gender_bias(predictions, true_labels, confidences, gender_map):
    """Analisa viés de gênero."""
    results = {'m': {'predictions': [], 'true_labels': [], 'confidences': []},
               'f': {'predictions': [], 'true_labels': [], 'confidences': []}}
    
    for p, t, c in zip(predictions, true_labels, confidences):
        gender = gender_map.get(t, 'unknown')
        if gender in ['m', 'f']:
            results[gender]['predictions'].append(p)
            results[gender]['true_labels'].append(t)
            results[gender]['confidences'].append(c)
    
    gender_metrics = {}
    for gender in ['m', 'f']:
        if results[gender]['predictions']:
            gender_metrics[gender] = calculate_metrics(
                results[gender]['predictions'],
                results[gender]['true_labels'],
                results[gender]['confidences']
            )
    
    return gender_metrics

def plot_metrics(metrics, output_dir):
    """
    Plota gráficos das métricas usando matplotlib em português.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Gráfico de barras para acurácia, FAR, FRR
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = ['Acurácia Global', 'Taxa de Falsa Aceitação (FAR)', 'Taxa de Falsa Rejeição (FRR)', 
              'Precisão (Macro)', 'Revocação (Macro)', 'F1-Score (Macro)']
    values = [metrics['global_accuracy'], metrics['far'], metrics['frr'], 
              metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']]
    colors = ['green', 'red', 'orange', 'blue', 'purple', 'brown']
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Valor')
    ax.set_title('Métricas de Verificação de Voz')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metricas_barra.png'))
    plt.close()
    
    # Gráfico de confiança
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Corretas', 'Incorretas']
    values = [metrics['avg_confidence_correct'], metrics['avg_confidence_incorrect']]
    bars = ax.bar(labels, values, color=['blue', 'red'])
    ax.set_ylabel('Confiança Média')
    ax.set_title('Scores de Confiança Média')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'confianca_scores.png'))
    plt.close()

def plot_confusion_matrix(predictions, true_labels, output_dir):
    """
    Plota matriz de confusão.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    unique_labels = sorted(list(set(true_labels + predictions)))
    cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'))
    plt.close()

def plot_threshold_analysis(threshold_results, output_dir):
    """Plota análise de diferentes thresholds."""
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = [r['threshold'] for r in threshold_results]
    accuracies = [r['global_accuracy'] for r in threshold_results]
    fars = [r['far'] for r in threshold_results]
    frrs = [r['frr'] for r in threshold_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, accuracies, 'o-', label='Acurácia Global', color='green')
    ax.plot(thresholds, fars, 's-', label='FAR', color='red')
    ax.plot(thresholds, frrs, '^-', label='FRR', color='orange')
    ax.set_xlabel('Threshold de Confiança')
    ax.set_ylabel('Valor')
    ax.set_title('Análise de Performance por Threshold')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(output_dir, 'analise_threshold.png'))
    plt.close()

def plot_by_num_speakers(speaker_results, output_dir):
    """Plota performance por número de speakers."""
    os.makedirs(output_dir, exist_ok=True)
    
    nums = list(speaker_results.keys())
    accuracies = [speaker_results[n]['global_accuracy'] for n in nums]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(nums, accuracies, color='skyblue')
    ax.set_xlabel('Número de Speakers')
    ax.set_ylabel('Acurácia Global')
    ax.set_title('Acurácia por Número de Speakers')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'acuracia_por_speakers.png'))
    plt.close()

def plot_gender_bias(gender_results, output_dir):
    """Plota análise de viés de gênero."""
    os.makedirs(output_dir, exist_ok=True)
    
    genders = ['Masculino', 'Feminino']
    keys = ['m', 'f']
    accuracies = [gender_results[k]['global_accuracy'] for k in keys if k in gender_results]
    genders = genders[:len(accuracies)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(genders, accuracies, color=['lightblue', 'pink'])
    ax.set_ylabel('Acurácia Global')
    ax.set_title('Viés de Gênero - Acurácia')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'vies_genero.png'))
    plt.close()

def main():
    results_file = "../analysis_results/detailed_results_with_females.csv"
    meta_path = "../vox1_meta.csv"
    output_dir = "../analysis_results"
    
    if not os.path.exists(results_file):
        print("Arquivo de resultados não encontrado")
        return
    
    print("Carregando mapeamento...")
    speaker_mapping, gender_map = load_speaker_mapping(meta_path)
    
    print("Carregando resultados...")
    df = pd.read_csv(results_file)
    
    # Aplicar mapeamento às predições
    df['prediction_mapped'] = df['prediction'].map(lambda x: speaker_mapping.get(x, x))
    
    # Adicionar coluna file simulada (5 segmentos por arquivo)
    files = []
    for i in range(len(df)):
        if i < 5:
            files.append('../data/demo/mix_2_speakers.wav')
        elif i < 10:
            files.append('../data/demo/mix_3_speakers.wav')
        elif i < 15:
            files.append('../data/demo/mix_4_speakers.wav')
        else:
            files.append('../data/demo/mix_5_speakers.wav')
    df['file'] = files
    
    predictions = df['prediction_mapped'].tolist()
    true_labels = df['true_label'].tolist()
    confidences = df['confidence'].tolist()
    
    print("Calculando métricas...")
    metrics = calculate_metrics(predictions, true_labels, confidences)
    
    print("\n=== Métricas Calculadas ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Análises adicionais
    print("\nAnalisando por threshold...")
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = analyze_by_threshold(predictions, true_labels, confidences, thresholds)
    
    print("Analisando por número de speakers...")
    speaker_results = analyze_by_num_speakers(df, speaker_mapping)
    
    print("Analisando viés de gênero...")
    gender_results = analyze_gender_bias(predictions, true_labels, confidences, gender_map)
    
    # Plotar
    plot_metrics(metrics, output_dir)
    plot_confusion_matrix(predictions, true_labels, output_dir)
    plot_threshold_analysis(threshold_results, output_dir)
    plot_by_num_speakers(speaker_results, output_dir)
    plot_gender_bias(gender_results, output_dir)
    
    # Salvar resultados mapeados
    df.to_csv(os.path.join(output_dir, 'detailed_results_mapped.csv'), index=False)
    
    # Salvar métricas detalhadas
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Análise Estatística Completa do Sistema de Re-ID de Voz ===\n\n")
        f.write("MÉTRICAS GERAIS:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write("\nANÁLISE POR THRESHOLD:\n")
        for r in threshold_results:
            f.write(f"Threshold {r['threshold']:.1f}: Acurácia={r['global_accuracy']:.3f}, FAR={r['far']:.3f}, FRR={r['frr']:.3f}\n")
        
        f.write("\nANÁLISE POR NÚMERO DE SPEAKERS:\n")
        for num, m in speaker_results.items():
            f.write(f"{num} speakers: Acurácia={m['global_accuracy']:.3f}\n")
        
        f.write("\nANÁLISE DE VIÉS DE GÊNERO:\n")
        for gender, m in gender_results.items():
            gender_name = 'Masculino' if gender == 'm' else 'Feminino'
            f.write(f"{gender_name}: Acurácia={m['global_accuracy']:.3f}\n")
    
    print(f"\nResultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()