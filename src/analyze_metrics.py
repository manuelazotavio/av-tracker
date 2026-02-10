import os
import torch
import torchaudio
import numpy as np
import logging
import csv
from multi_speaker_verifier import MultiSpeakerVerifier
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_speaker_mapping(meta_path):
    """Load mapping from VoxCeleb1 ID to VGGFace1 ID."""
    mapping = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            vox_id = row['VoxCeleb1 ID']
            vgg_id = row['VGGFace1 ID']
            mapping[vox_id] = vgg_id
    return mapping

class SimpleMetricsAnalyzer:
    def __init__(self, verifier, speaker_mapping, device="cpu"):
        self.verifier = verifier
        self.speaker_mapping = speaker_mapping
        self.device = device

    def process_mixed_audio_simple(self, audio_path, ground_truth_speakers, segment_length=3.0):
        """
        Processa áudio misturado dividindo em segmentos fixos e verificando cada um.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        logger.info(f"Processando: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Converter para mono se necessário
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Parâmetros
        segment_samples = int(segment_length * sample_rate)
        total_samples = waveform.shape[1]
        
        predictions = []
        true_labels = []
        confidences = []
        
        # Dividir em segmentos
        for start_sample in range(0, total_samples, segment_samples):
            end_sample = min(start_sample + segment_samples, total_samples)
            segment = waveform[:, start_sample:end_sample]
            
            if segment.shape[1] < segment_samples * 0.5:  # Pular segmentos muito curtos
                continue
            
            # Converter para numpy
            segment_np = segment.squeeze().numpy()
            
            # Normalizar
            mx = np.abs(segment_np).max()
            if np.isnan(mx) or mx == 0:
                continue
            segment_np = segment_np / mx
            
            # Verificar
            pred_speaker, conf = self.verifier._process_audio_chunk(segment_np, sample_rate=sample_rate)
            
            # Mapear predição para VGGFace1 ID se possível
            mapped_pred = self.speaker_mapping.get(pred_speaker, pred_speaker)
            
            # Ground truth: assumimos que qualquer speaker do ground truth é válido
            true_speaker = ground_truth_speakers[0] if ground_truth_speakers else "Unknown"
            
            predictions.append(mapped_pred)
            true_labels.append(true_speaker)
            confidences.append(conf)
        
        return predictions, true_labels, confidences

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
    
    return metrics

def plot_metrics(metrics, output_dir):
    """
    Plota gráficos das métricas usando matplotlib.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Gráfico de barras para acurácia, FAR, FRR
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Global Accuracy', 'FAR', 'FRR']
    values = [metrics['global_accuracy'], metrics['far'], metrics['frr']]
    bars = ax.bar(labels, values, color=['green', 'red', 'orange'])
    ax.set_ylabel('Value')
    ax.set_title('Speaker Verification Metrics')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'metrics_bar.png'))
    plt.close()
    
    # Gráfico de confiança
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Correct', 'Incorrect']
    values = [metrics['avg_confidence_correct'], metrics['avg_confidence_incorrect']]
    bars = ax.bar(labels, values, color=['blue', 'red'])
    ax.set_ylabel('Average Confidence')
    ax.set_title('Average Confidence Scores')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'confidence_scores.png'))
    plt.close()

def main():
    embeddings_dir = "../data/embeddings"
    demo_dir = "../data/demo"
    output_dir = "../analysis_results"
    meta_path = "../vox1_meta.csv"
    
    if not os.path.exists(embeddings_dir):
        print("Pasta de embeddings não encontrada")
        return

    print("Carregando mapeamento de speakers...")
    speaker_mapping = load_speaker_mapping(meta_path)

    print("Inicializando...")
    verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.0)
    analyzer = SimpleMetricsAnalyzer(verifier, speaker_mapping)
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    # Processar cada áudio misturado
    for num_speakers in [2, 3, 4, 5]:
        audio_file = os.path.join(demo_dir, f"mix_{num_speakers}_speakers.wav")
        gt_file = os.path.join(demo_dir, f"mix_{num_speakers}_ground_truth.txt")
        
        if not os.path.exists(audio_file) or not os.path.exists(gt_file):
            print(f"Arquivos para {num_speakers} speakers não encontrados, pulando...")
            continue
        
        # Ler ground truth
        with open(gt_file, 'r') as f:
            lines = f.readlines()
            speakers_line = lines[0].strip()
            ground_truth_speakers = [s.strip() for s in speakers_line.replace("Speakers: ", "").split(", ")]
        
        print(f"\nProcessando mix com {num_speakers} speakers: {ground_truth_speakers}")
        
        predictions, true_labels, confidences = analyzer.process_mixed_audio_simple(audio_file, ground_truth_speakers)
        
        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)
        all_confidences.extend(confidences)
        
        print(f"  Segmentos processados: {len(predictions)}")
    
    if not all_predictions:
        print("Nenhum segmento processado. Verifique os arquivos de áudio.")
        return
    
    # Calcular métricas globais
    metrics = calculate_metrics(all_predictions, all_true_labels, all_confidences)
    
    print("\n=== Métricas Calculadas ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plotar
    plot_metrics(metrics, output_dir)
    
    # Salvar resultados
    results_df = pd.DataFrame({
        'prediction': all_predictions,
        'true_label': all_true_labels,
        'confidence': all_confidences
    })
    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    print(f"\nResultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()