"""Teste rápido das métricas - processa apenas 1 arquivo"""
import os
import sys
from batch_process_multi_speaker import BatchProcessor

if __name__ == "__main__":
    embeddings_dir = "../data/embeddings/individual"
    output_dir = "../logs"
    
    print("Inicializando...")
    processor = BatchProcessor(embeddings_dir, output_dir)
    
    # Processa apenas 1 arquivo de teste
    test_file = "../data/mixed_audio/mix_2_speakers/mix_2_speakers_001.wav"
    gt_file = "../data/mixed_audio/mix_2_speakers/mix_2_speakers_001_ground_truth.txt"
    
    print(f"\nProcessando: {test_file}")
    result = processor.process_audio_file(test_file, gt_file)
    
    if result['success']:
        m = result['metrics']
        print("\n=== RESULTADO ===")
        print(f"Speaker Accuracy: {m['speaker_accuracy']:.2f}%")
        print(f"Precision: {m['speaker_precision']:.2f}%")
        print(f"Recall: {m['speaker_recall']:.2f}%")
        print(f"F1-Score: {m['speaker_f1_score']:.2f}%")
        print(f"DER: {m['diarization_error_rate']:.2f}%")
        print(f"\nSpeakers GT: {m['num_speakers_ground_truth']}")
        print(f"Speakers Detectados: {m['num_speakers_predicted']}")
        print(f"Correct IDs: {m['correct_speaker_ids']}/{m['total_compared']}")
        print(f"\nGender metrics:")
        for gender in ['m', 'f', 'unknown']:
            gm = m['gender_metrics'][gender]
            if gm['total_gt'] > 0 or gm['total_pred'] > 0:
                print(f"  {gender}: GT={gm['total_gt']}, Pred={gm['total_pred']}, Correct={gm['correct']}, Acc={gm['accuracy']:.1f}%")
    else:
        print("ERRO:", result.get('error'))
