import os
import torch
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from multi_speaker_verification import LargeMeetingTranscriber
from multi_speaker_verifier import MultiSpeakerVerifier
from collections import defaultdict, Counter
import re
import torchaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = "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"

class GenderAnalyzer:
    """Analisa gênero dos speakers baseado em metadados VoxCeleb"""
    def __init__(self, vox_meta_path="../vox1_meta.csv"):
        self.gender_map = {}
        if os.path.exists(vox_meta_path):
            try:
                df = pd.read_csv(vox_meta_path, delimiter='\t')
                for _, row in df.iterrows():
                    vox_id = row['VoxCeleb1 ID'].strip()
                    gender = row['Gender'].strip().lower()
                    self.gender_map[vox_id] = gender
                logger.info(f"✅ Carregados dados de gênero para {len(self.gender_map)} speakers")
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível carregar metadados de gênero: {e}")
        else:
            logger.warning(f"⚠️ Arquivo de metadados não encontrado: {vox_meta_path}")
    
    def get_gender(self, speaker_id):
        """Retorna o gênero do speaker (m/f) ou 'unknown'"""
        return self.gender_map.get(speaker_id, 'unknown')

class BatchProcessor:
    def __init__(self, embeddings_dir, output_dir):
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.results = []
        self.gender_analyzer = GenderAnalyzer()
        
        logger.info("Inicializando Batch Processor...")
        self.verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.0)
        self.transcriber = LargeMeetingTranscriber(self.verifier, HF_TOKEN)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def parse_ground_truth(self, ground_truth_path):
        """Lê o arquivo ground_truth e retorna informações dos speakers"""
        speakers = set()
        segments = []
        speaker_map = {}  # Mapeia nome legível para ID VoxCeleb
        
        if not os.path.exists(ground_truth_path):
            logger.warning(f"Ground truth não encontrado: {ground_truth_path}")
            return speakers, segments
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse formato novo: extrai speakers e seus IDs VoxCeleb
        # Formato: "Speaker X: Name\n  WAV file: .../idXXXXX/..."
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Procura linhas "Speakers: name1, name2, ..."
            if line.startswith('Speakers:'):
                speaker_names = line.replace('Speakers:', '').strip().split(',')
                speaker_names = [s.strip() for s in speaker_names]
            
            # Procura linhas "Speaker X (name) enters at Ys"
            if 'enters at' in line:
                match = re.search(r'Speaker \d+ \(([^)]+)\) enters at ([\d.]+)s', line)
                if match:
                    speaker_name = match.group(1)
                    timestamp = float(match.group(2))
                    
                    # Cria entrada inicial para este speaker
                    segments.append({
                        'timestamp': timestamp,
                        'speaker': speaker_name,
                        'text': f'[Speaker {speaker_name} enters]'
                    })
            
            # Procura linhas "Speaker X: Name" seguido de "WAV file: .../idXXXXX/..."
            if line.startswith('Speaker ') and ':' in line and 'WAV file' not in line:
                speaker_name = line.split(':', 1)[1].strip()
                # Próxima linha deve ter o WAV file com o ID
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if 'WAV file:' in next_line:
                        # Extrai ID VoxCeleb do caminho
                        wav_match = re.search(r'[/\\](id\d+)[/\\]', next_line)
                        if wav_match:
                            vox_id = wav_match.group(1)
                            speaker_map[speaker_name] = vox_id
                            speakers.add(vox_id)
        
        # Atualiza os segmentos com os IDs corretos
        for seg in segments:
            name = seg['speaker']
            if name in speaker_map:
                seg['speaker'] = speaker_map[name]
        
        logger.debug(f"Ground truth parsed: {len(speakers)} speakers, {len(segments)} segments")
        logger.debug(f"Speaker map: {speaker_map}")
        
        return speakers, segments
    
    def calculate_metrics(self, predicted_transcript, ground_truth_segments, ground_truth_speakers, audio_duration, processing_time, waveform, sample_rate):
        """Calcula métricas detalhadas de acurácia e performance"""
        
        # Calcula métricas de áudio (volume/energia)
        audio_np = waveform.numpy()
        rms_energy = np.sqrt(np.mean(audio_np ** 2))
        peak_amplitude = np.max(np.abs(audio_np))
        db_level = 20 * np.log10(rms_energy + 1e-10)
        
        # Classifica volume
        if db_level < -30:
            volume_category = "muito_baixo"
        elif db_level < -20:
            volume_category = "baixo"
        elif db_level < -10:
            volume_category = "medio"
        else:
            volume_category = "alto"
        
        metrics = {
            # Métricas básicas
            'total_segments_predicted': len(predicted_transcript),
            'total_segments_ground_truth': len(ground_truth_segments),
            'audio_duration_seconds': audio_duration,
            'processing_time_seconds': processing_time,
            'processing_speed_ratio': audio_duration / processing_time if processing_time > 0 else 0,
            
            # Métricas de áudio
            'audio_metrics': {
                'rms_energy': float(rms_energy),
                'peak_amplitude': float(peak_amplitude),
                'db_level': float(db_level),
                'volume_category': volume_category,
                'sample_rate': sample_rate
            },
            
            # Métricas de identificação
            'correct_speaker_ids': 0,
            'total_compared': 0,
            'speaker_accuracy': 0.0,
            'speaker_precision': 0.0,
            'speaker_recall': 0.0,
            'speaker_f1_score': 0.0,
            
            # Informação de speakers
            'ground_truth_speakers': sorted(list(ground_truth_speakers)),
            'predicted_speakers': [],
            'num_speakers_predicted': 0,
            'num_speakers_ground_truth': len(ground_truth_speakers),
            'speaker_detection_accuracy': 0.0,
            
            # Confusion matrix e métricas por speaker
            'confusion_matrix': {},
            'per_speaker_metrics': {},
            
            # Métricas por gênero
            'gender_metrics': {
                'm': {'total_gt': 0, 'total_pred': 0, 'correct': 0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0},
                'f': {'total_gt': 0, 'total_pred': 0, 'correct': 0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0},
                'unknown': {'total_gt': 0, 'total_pred': 0, 'correct': 0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0},
                'gender_distribution_gt': {},
                'gender_distribution_pred': {}
            },
            
            # Métricas de confiança
            'average_confidence': 0.0,
            'confidence_distribution': {},
            'low_confidence_segments': 0,
            
            # Métricas de overlap/separação
            'overlapping_segments': 0,
            'overlap_accuracy': 0.0,
            'separation_attempts': 0,
            
            # Métricas temporais
            'avg_segment_duration': 0.0,
            'total_speech_time_gt': 0.0,
            'total_speech_time_pred': 0.0,
            
            # Erros detalhados
            'false_alarms': 0,
            'missed_detections': 0,
            'speaker_confusion_errors': 0,
        }
        
        # Extrai speakers únicos preditos e calcula confiança
        predicted_speakers = set()
        confidences = []
        overlap_count = 0
        
        for seg in predicted_transcript:
            speaker = seg['speaker']
            # Remove sufixo (Voz 1), (Voz 2) se existir
            clean_speaker = re.sub(r'\s*\(Voz \d+\)$', '', speaker)
            predicted_speakers.add(clean_speaker)
            
            # Conta overlaps
            if '(Voz' in seg.get('suffix', ''):
                overlap_count += 1
        
        metrics['predicted_speakers'] = sorted(list(predicted_speakers))
        metrics['num_speakers_predicted'] = len(predicted_speakers)
        metrics['overlapping_segments'] = overlap_count
        
        # Acurácia de detecção de número de speakers
        if metrics['num_speakers_ground_truth'] > 0:
            speaker_diff = abs(metrics['num_speakers_predicted'] - metrics['num_speakers_ground_truth'])
            metrics['speaker_detection_accuracy'] = max(0, (1 - speaker_diff / metrics['num_speakers_ground_truth']) * 100)
        
        # Inicializa confusion matrix
        all_speakers = ground_truth_speakers.union(predicted_speakers)
        for gt_spk in all_speakers:
            metrics['confusion_matrix'][gt_spk] = {}
            for pred_spk in all_speakers:
                metrics['confusion_matrix'][gt_spk][pred_spk] = 0
        
        # Inicializa métricas por speaker (incluindo gênero)
        for spk in ground_truth_speakers:
            gender = self.gender_analyzer.get_gender(spk)
            metrics['per_speaker_metrics'][spk] = {
                'gender': gender,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'total_segments_gt': 0,
                'total_segments_pred': 0
            }
            metrics['gender_metrics'][gender]['total_gt'] += 1
            metrics['gender_metrics']['gender_distribution_gt'][spk] = gender
        
        # Conta segmentos ground truth por speaker
        for gt_seg in ground_truth_segments:
            spk = gt_seg['speaker']
            if spk in metrics['per_speaker_metrics']:
                metrics['per_speaker_metrics'][spk]['total_segments_gt'] += 1
        
        # Adiciona gênero para speakers preditos
        for spk in predicted_speakers:
            gender = self.gender_analyzer.get_gender(spk)
            metrics['gender_metrics']['gender_distribution_pred'][spk] = gender
            if spk not in metrics['per_speaker_metrics']:
                metrics['per_speaker_metrics'][spk] = {
                    'gender': gender,
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'total_segments_gt': 0,
                    'total_segments_pred': 0
                }
        
        # Nova abordagem: comparar cada segmento de transcrição com os speakers esperados
        # Já que o ground truth só diz quais speakers estão presentes (e quando entram),
        # vamos avaliar se cada segmento detectado corresponde a um speaker esperado
        
        for i, pred_seg in enumerate(predicted_transcript):
            pred_time = pred_seg['start']
            pred_speaker = re.sub(r'\s*\(Voz \d+\)$', '', pred_seg['speaker'])
            
            # Determina quais speakers deveriam estar ativos neste timestamp
            active_speakers = set()
            for gt_seg in ground_truth_segments:
                if pred_time >= gt_seg['timestamp']:
                    active_speakers.add(gt_seg['speaker'])
            
            # Se não há speakers ativos ainda, usa todos os speakers conhecidos
            if not active_speakers:
                active_speakers = ground_truth_speakers
            
            metrics['total_compared'] += 1
            
            # Verifica se o speaker predito está entre os esperados
            is_correct = (pred_speaker in active_speakers)
            
            # Atualiza confusion matrix
            for gt_spk in active_speakers:
                if gt_spk in metrics['confusion_matrix'] and pred_speaker in metrics['confusion_matrix'][gt_spk]:
                    if pred_speaker == gt_spk:
                        metrics['confusion_matrix'][gt_spk][pred_speaker] += 1
            
            pred_gender = self.gender_analyzer.get_gender(pred_speaker)
            
            if is_correct:
                metrics['correct_speaker_ids'] += 1
                # Incrementa correct para o gênero do speaker correto
                gt_gender = self.gender_analyzer.get_gender(pred_speaker)
                metrics['gender_metrics'][gt_gender]['correct'] += 1
                metrics['gender_metrics'][gt_gender]['total_pred'] += 1
                
                if pred_speaker in metrics['per_speaker_metrics']:
                    metrics['per_speaker_metrics'][pred_speaker]['true_positives'] += 1
                    metrics['per_speaker_metrics'][pred_speaker]['total_segments_pred'] += 1
            else:
                # Speaker incorreto - é confusão ou false alarm
                if pred_speaker in ground_truth_speakers:
                    # É um speaker válido mas no momento errado = confusão
                    metrics['speaker_confusion_errors'] += 1
                    if pred_speaker in metrics['per_speaker_metrics']:
                        metrics['per_speaker_metrics'][pred_speaker]['false_positives'] += 1
                else:
                    # Speaker completamente desconhecido = false alarm
                    metrics['false_alarms'] += 1
                
                # Incrementa total_pred para o gênero detectado
                metrics['gender_metrics'][pred_gender]['total_pred'] += 1
        
        # Calcula missed detections baseado em speakers não detectados
        for gt_spk in ground_truth_speakers:
            if gt_spk not in predicted_speakers:
                metrics['missed_detections'] += 1
                if gt_spk in metrics['per_speaker_metrics']:
                    metrics['per_speaker_metrics'][gt_spk]['false_negatives'] += 1
        
        # Calcula acurácia geral
        if metrics['total_compared'] > 0:
            metrics['speaker_accuracy'] = (metrics['correct_speaker_ids'] / metrics['total_compared']) * 100
            metrics['speaker_precision'] = (metrics['correct_speaker_ids'] / len(predicted_transcript)) * 100 if len(predicted_transcript) > 0 else 0
            # Recall baseado nos speakers esperados aparecendo na transcrição
            expected_total = len(ground_truth_speakers) * max(1, len(predicted_transcript) // len(ground_truth_speakers))
            metrics['speaker_recall'] = (metrics['correct_speaker_ids'] / expected_total) * 100 if expected_total > 0 else 0
            
            if metrics['speaker_precision'] + metrics['speaker_recall'] > 0:
                metrics['speaker_f1_score'] = 2 * (metrics['speaker_precision'] * metrics['speaker_recall']) / (metrics['speaker_precision'] + metrics['speaker_recall'])
        
        # Calcula métricas por speaker
        for spk in metrics['per_speaker_metrics']:
            m = metrics['per_speaker_metrics'][spk]
            tp = m['true_positives']
            fp = m['false_positives']
            fn = m['false_negatives']
            
            if tp + fp > 0:
                m['precision'] = (tp / (tp + fp)) * 100
            if tp + fn > 0:
                m['recall'] = (tp / (tp + fn)) * 100
            if m['precision'] + m['recall'] > 0:
                m['f1_score'] = 2 * (m['precision'] * m['recall']) / (m['precision'] + m['recall'])
        
        # Calcula acurácia de overlap
        if overlap_count > 0:
            # Estima acurácia baseada em overlaps identificados corretamente
            overlap_correct = 0
            for seg in predicted_transcript:
                if '(Voz' in seg.get('suffix', ''):
                    # Considera correto se identificou o speaker certo em overlap
                    speaker = re.sub(r'\s*\(Voz \d+\)$', '', seg['speaker'])
                    if speaker in ground_truth_speakers:
                        overlap_correct += 1
            metrics['overlap_accuracy'] = (overlap_correct / overlap_count) * 100 if overlap_count > 0 else 0
        
        # Calcula DER (Diarization Error Rate) simplificado
        # DER = (FA + Miss + Confusion) / Total Reference
        # Como nosso ground truth é simples, usamos número de segmentos como referência
        total_errors = metrics['false_alarms'] + metrics['missed_detections'] + metrics['speaker_confusion_errors']
        total_reference = max(len(predicted_transcript), len(ground_truth_speakers))  # Usa o maior como referência
        metrics['diarization_error_rate'] = (total_errors / total_reference) * 100 if total_reference > 0 else 0
        metrics['diarization_error_rate'] = (total_errors / total_reference) * 100 if total_reference > 0 else 0
        
        # Calcula métricas por gênero
        for gender in ['m', 'f', 'unknown']:
            gm = metrics['gender_metrics'][gender]
            if gm['total_gt'] > 0:
                gm['recall'] = (gm['correct'] / gm['total_gt']) * 100
            if gm['total_pred'] > 0:
                gm['precision'] = (gm['correct'] / gm['total_pred']) * 100
            if gm['correct'] > 0:
                gm['accuracy'] = (gm['correct'] / max(gm['total_gt'], gm['total_pred'])) * 100
        
        return metrics
    
    def process_audio_file(self, audio_path, ground_truth_path):
        """Processa um único arquivo de áudio"""
        logger.info(f"Processando: {audio_path}")
        
        try:
            # Carrega áudio completo
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_duration = waveform.shape[1] / sample_rate
            
            # Processa o áudio
            start_time = time.time()
            transcript = self.transcriber.process_audio(audio_path)
            processing_time = time.time() - start_time
            
            # Lê ground truth
            gt_speakers, gt_segments = self.parse_ground_truth(ground_truth_path)
            
            # Calcula métricas (passa waveform e sample_rate)
            metrics = self.calculate_metrics(transcript, gt_segments, gt_speakers, audio_duration, processing_time, waveform, sample_rate)
            
            result = {
                'audio_file': os.path.basename(audio_path),
                'success': True,
                'metrics': metrics,
                'transcript': transcript
            }
            
            logger.info(f"✅ Concluído: {os.path.basename(audio_path)}")
            logger.info(f"   Duração: {audio_duration:.2f}s | Proc: {processing_time:.2f}s | Ratio: {metrics['processing_speed_ratio']:.2f}x | Vol: {metrics['audio_metrics']['volume_category']} ({metrics['audio_metrics']['db_level']:.1f}dB)")
            logger.info(f"   Accuracy: {metrics['speaker_accuracy']:.2f}% | Prec: {metrics['speaker_precision']:.2f}% | Rec: {metrics['speaker_recall']:.2f}%")
            logger.info(f"   F1: {metrics['speaker_f1_score']:.2f}% | DER: {metrics['diarization_error_rate']:.2f}%")
            logger.info(f"   Speakers GT: {metrics['num_speakers_ground_truth']} | Pred: {metrics['num_speakers_predicted']} | Det Acc: {metrics['speaker_detection_accuracy']:.2f}%")
            
            # Log métricas de gênero
            gm = metrics['gender_metrics']
            if gm['m']['total_gt'] > 0 or gm['f']['total_gt'] > 0:
                male_acc = gm['m']['accuracy'] if gm['m']['total_gt'] > 0 else 0
                female_acc = gm['f']['accuracy'] if gm['f']['total_gt'] > 0 else 0
                logger.info(f"   Gênero: M={gm['m']['total_gt']} (Acc: {male_acc:.1f}%) | F={gm['f']['total_gt']} (Acc: {female_acc:.1f}%)")
            
            logger.info(f"   Erros: FA={metrics['false_alarms']}, Miss={metrics['missed_detections']}, Conf={metrics['speaker_confusion_errors']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'audio_file': os.path.basename(audio_path),
                'success': False,
                'error': str(e)
            }
    
    def process_directory(self, mixed_audio_dir):
        """Processa todos os arquivos em um diretório"""
        audio_dir = Path(mixed_audio_dir)
        
        # Encontra todos os arquivos .wav nas subpastas
        wav_files = []
        for subdir in ['mix_2_speakers', 'mix_3_speakers', 'mix_4_speakers', 'mix_5_speakers']:
            subdir_path = audio_dir / subdir
            if subdir_path.exists():
                wav_files.extend(sorted(subdir_path.glob('*.wav')))
        
        logger.info(f"Encontrados {len(wav_files)} arquivos de áudio")
        
        # Define quantos arquivos pular (já processados)
        SKIP_FIRST_N = 86
        if SKIP_FIRST_N > 0:
            logger.info(f"⚠️  Pulando os primeiros {SKIP_FIRST_N} arquivos já processados")
        
        # Processa cada arquivo
        for i, wav_file in enumerate(wav_files, 1):
            # Ignora arquivos que não são os mix principais (evita duplicados)
            if '_ground_truth' in wav_file.name or '_transcricao' in wav_file.name:
                continue
            
            # Pula os primeiros N arquivos já processados
            if i <= SKIP_FIRST_N:
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Arquivo {i}/{len(wav_files)}")
            
            # Encontra o arquivo ground_truth correspondente
            gt_file = wav_file.with_name(wav_file.stem + '_ground_truth.txt')
            
            result = self.process_audio_file(str(wav_file), str(gt_file))
            self.results.append(result)
            
            # Salva resultados parciais
            self.save_results()
    
    def save_results(self):
        """Salva resultados em arquivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"batch_results_{timestamp}.json")
        
        # Calcula estatísticas gerais
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]
        
        # Estatísticas agregadas
        if successful:
            accuracies = [r['metrics']['speaker_accuracy'] for r in successful]
            precisions = [r['metrics']['speaker_precision'] for r in successful]
            recalls = [r['metrics']['speaker_recall'] for r in successful]
            f1_scores = [r['metrics']['speaker_f1_score'] for r in successful]
            ders = [r['metrics']['diarization_error_rate'] for r in successful]
            processing_ratios = [r['metrics']['processing_speed_ratio'] for r in successful]
            
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_der = sum(ders) / len(ders)
            avg_processing_ratio = sum(processing_ratios) / len(processing_ratios)
            
            # Agrupa por número de speakers
            by_num_speakers = defaultdict(list)
            for r in successful:
                num_spk = r['metrics']['num_speakers_ground_truth']
                by_num_speakers[num_spk].append(r)
            
            speaker_group_stats = {}
            for num_spk, results in by_num_speakers.items():
                accs = [r['metrics']['speaker_accuracy'] for r in results]
                speaker_group_stats[f'{num_spk}_speakers'] = {
                    'count': len(results),
                    'avg_accuracy': sum(accs) / len(accs),
                    'min_accuracy': min(accs),
                    'max_accuracy': max(accs)
                }
            
            # Agrupa por volume
            by_volume = defaultdict(list)
            for r in successful:
                vol_cat = r['metrics']['audio_metrics']['volume_category']
                by_volume[vol_cat].append(r)
            
            volume_stats = {}
            for vol_cat, results in by_volume.items():
                accs = [r['metrics']['speaker_accuracy'] for r in results]
                dbs = [r['metrics']['audio_metrics']['db_level'] for r in results]
                volume_stats[vol_cat] = {
                    'count': len(results),
                    'avg_accuracy': sum(accs) / len(accs),
                    'min_accuracy': min(accs),
                    'max_accuracy': max(accs),
                    'avg_db_level': sum(dbs) / len(dbs)
                }
            
            # Métricas agregadas por gênero
            gender_stats = {
                'm': {'total_speakers': 0, 'total_correct': 0, 'total_segments': 0, 'avg_accuracy': 0.0},
                'f': {'total_speakers': 0, 'total_correct': 0, 'total_segments': 0, 'avg_accuracy': 0.0},
                'unknown': {'total_speakers': 0, 'total_correct': 0, 'total_segments': 0, 'avg_accuracy': 0.0}
            }
            
            for r in successful:
                gm = r['metrics']['gender_metrics']
                for gender in ['m', 'f', 'unknown']:
                    gender_stats[gender]['total_speakers'] += gm[gender]['total_gt']
                    gender_stats[gender]['total_correct'] += gm[gender]['correct']
                    gender_stats[gender]['total_segments'] += gm[gender]['total_gt']
            
            for gender in ['m', 'f', 'unknown']:
                if gender_stats[gender]['total_segments'] > 0:
                    gender_stats[gender]['avg_accuracy'] = (gender_stats[gender]['total_correct'] / gender_stats[gender]['total_segments']) * 100
                    
        else:
            avg_accuracy = avg_precision = avg_recall = avg_f1 = avg_der = avg_processing_ratio = 0.0
            speaker_group_stats = {}
            volume_stats = {}
            gender_stats = {}
        
        summary = {
            'timestamp': timestamp,
            'total_processed': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            
            # Métricas médias gerais
            'overall_metrics': {
                'average_speaker_accuracy': avg_accuracy,
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1,
                'average_diarization_error_rate': avg_der,
                'average_processing_speed_ratio': avg_processing_ratio,
            },
            
            # Métricas por grupo de speakers
            'by_number_of_speakers': speaker_group_stats,
            
            # Métricas por volume/categoria de áudio
            'by_audio_volume': volume_stats,
            
            # Métricas por gênero
            'by_gender': gender_stats,
            
            # Resultados detalhados
            'results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTADOS FINAIS")
        logger.info(f"{'='*80}")
        logger.info(f"Resultados salvos em: {output_file}")
        logger.info(f"\nESTATÍSTICAS GERAIS:")
        logger.info(f"  Total processados: {len(self.results)}")
        logger.info(f"  Sucesso: {len(successful)}")
        logger.info(f"  Falhas: {len(failed)}")
        logger.info(f"\nMÉTRICAS MÉDIAS:")
        logger.info(f"  Speaker Accuracy: {avg_accuracy:.2f}%")
        logger.info(f"  Precision: {avg_precision:.2f}%")
        logger.info(f"  Recall: {avg_recall:.2f}%")
        logger.info(f"  F1-Score: {avg_f1:.2f}%")
        logger.info(f"  DER (Diarization Error Rate): {avg_der:.2f}%")
        logger.info(f"  Processing Speed: {avg_processing_ratio:.2f}x realtime")
        
        if speaker_group_stats:
            logger.info(f"\nPOR NÚMERO DE SPEAKERS:")
            for group, stats in sorted(speaker_group_stats.items()):
                logger.info(f"  {group}: {stats['count']} arquivos, Acc: {stats['avg_accuracy']:.2f}% (min: {stats['min_accuracy']:.2f}%, max: {stats['max_accuracy']:.2f}%)")
        
        if volume_stats:
            logger.info(f"\nPOR VOLUME DO ÁUDIO:")
            for vol_cat, stats in volume_stats.items():
                logger.info(f"  {vol_cat}: {stats['count']} arquivos, Acc: {stats['avg_accuracy']:.2f}% ({stats['avg_db_level']:.1f} dB)")
        
        if gender_stats:
            logger.info(f"\nPOR GÊNERO:")
            for gender, stats in gender_stats.items():
                if stats['total_speakers'] > 0:
                    logger.info(f"  {gender.capitalize()}: {stats['total_speakers']} speakers, Acc: {stats['avg_accuracy']:.2f}%")
        
        # Salva também um resumo detalhado em texto
        summary_file = os.path.join(self.output_dir, f"batch_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RESUMO DETALHADO DO PROCESSAMENTO EM LOTE\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"ESTATÍSTICAS GERAIS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Total processados: {len(self.results)}\n")
            f.write(f"Sucesso: {len(successful)}\n")
            f.write(f"Falhas: {len(failed)}\n\n")
            
            f.write(f"MÉTRICAS MÉDIAS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Speaker Accuracy:              {avg_accuracy:.2f}%\n")
            f.write(f"Precision:                     {avg_precision:.2f}%\n")
            f.write(f"Recall:                        {avg_recall:.2f}%\n")
            f.write(f"F1-Score:                      {avg_f1:.2f}%\n")
            f.write(f"DER (Diarization Error Rate):  {avg_der:.2f}%\n")
            f.write(f"Processing Speed:              {avg_processing_ratio:.2f}x realtime\n\n")
            
            if speaker_group_stats:
                f.write(f"MÉTRICAS POR NÚMERO DE SPEAKERS\n")
                f.write(f"{'-'*80}\n")
                for group, stats in sorted(speaker_group_stats.items()):
                    f.write(f"{group}:\n")
                    f.write(f"  Arquivos processados: {stats['count']}\n")
                    f.write(f"  Acurácia média: {stats['avg_accuracy']:.2f}%\n")
                    f.write(f"  Acurácia mínima: {stats['min_accuracy']:.2f}%\n")
                    f.write(f"  Acurácia máxima: {stats['max_accuracy']:.2f}%\n\n")
            
            if volume_stats:
                f.write(f"MÉTRICAS POR VOLUME DO ÁUDIO\n")
                f.write(f"{'-'*80}\n")
                for vol_cat, stats in volume_stats.items():
                    f.write(f"{vol_cat.upper()}:\n")
                    f.write(f"  Arquivos processados: {stats['count']}\n")
                    f.write(f"  Acurácia média: {stats['avg_accuracy']:.2f}%\n")
                    f.write(f"  Acurácia mínima: {stats['min_accuracy']:.2f}%\n")
                    f.write(f"  Acurácia máxima: {stats['max_accuracy']:.2f}%\n")
                    f.write(f"  Nível médio (dB): {stats['avg_db_level']:.1f}\n\n")
            
            if gender_stats:
                f.write(f"MÉTRICAS POR GÊNERO\n")
                f.write(f"{'-'*80}\n")
                for gender, stats in gender_stats.items():
                    if stats['total_speakers'] > 0:
                        f.write(f"{gender.upper()}:\n")
                        f.write(f"  Total de speakers: {stats['total_speakers']}\n")
                        f.write(f"  Total de segmentos: {stats['total_segments']}\n")
                        f.write(f"  Segmentos corretos: {stats['total_correct']}\n")
                        f.write(f"  Acurácia: {stats['avg_accuracy']:.2f}%\n\n")
            
            f.write(f"\nRESULTADOS DETALHADOS POR ARQUIVO\n")
            f.write(f"{'='*80}\n\n")
            
            for result in self.results:
                f.write(f"Arquivo: {result['audio_file']}\n")
                f.write(f"{'-'*80}\n")
                
                if result['success']:
                    m = result['metrics']
                    f.write(f"✅ SUCESSO\n\n")
                    
                    f.write(f"Métricas Principais:\n")
                    f.write(f"  Speaker Accuracy:          {m['speaker_accuracy']:.2f}%\n")
                    f.write(f"  Precision:                 {m['speaker_precision']:.2f}%\n")
                    f.write(f"  Recall:                    {m['speaker_recall']:.2f}%\n")
                    f.write(f"  F1-Score:                  {m['speaker_f1_score']:.2f}%\n")
                    f.write(f"  DER:                       {m['diarization_error_rate']:.2f}%\n\n")
                    
                    f.write(f"Informação de Speakers:\n")
                    f.write(f"  Ground Truth:              {m['ground_truth_speakers']}\n")
                    f.write(f"  Preditos:                  {m['predicted_speakers']}\n")
                    f.write(f"  Número GT:                 {m['num_speakers_ground_truth']}\n")
                    f.write(f"  Número Predito:            {m['num_speakers_predicted']}\n")
                    f.write(f"  Detection Accuracy:        {m['speaker_detection_accuracy']:.2f}%\n\n")
                    
                    f.write(f"Contagens:\n")
                    f.write(f"  Segmentos GT:              {m['total_segments_ground_truth']}\n")
                    f.write(f"  Segmentos Preditos:        {m['total_segments_predicted']}\n")
                    f.write(f"  Segmentos Comparados:      {m['total_compared']}\n")
                    f.write(f"  Identificações Corretas:   {m['correct_speaker_ids']}\n")
                    f.write(f"  False Alarms:              {m['false_alarms']}\n")
                    f.write(f"  Missed Detections:         {m['missed_detections']}\n")
                    f.write(f"  Speaker Confusions:        {m['speaker_confusion_errors']}\n\n")
                    
                    f.write(f"Overlaps:\n")
                    f.write(f"  Segmentos com Overlap:     {m['overlapping_segments']}\n")
                    f.write(f"  Overlap Accuracy:          {m['overlap_accuracy']:.2f}%\n\n")
                    
                    f.write(f"Performance:\n")
                    f.write(f"  Duração do Áudio:          {m['audio_duration_seconds']:.2f}s\n")
                    f.write(f"  Tempo de Processamento:    {m['processing_time_seconds']:.2f}s\n")
                    f.write(f"  Velocidade:                {m['processing_speed_ratio']:.2f}x realtime\n\n")
                    
                    f.write(f"Informações do Áudio:\n")
                    f.write(f"  RMS Energy:                {m['audio_metrics']['rms_energy']:.6f}\n")
                    f.write(f"  Peak Amplitude:            {m['audio_metrics']['peak_amplitude']:.6f}\n")
                    f.write(f"  Nível (dB):                {m['audio_metrics']['db_level']:.2f}\n")
                    f.write(f"  Categoria de Volume:       {m['audio_metrics']['volume_category']}\n")
                    f.write(f"  Sample Rate:               {m['audio_metrics']['sample_rate']} Hz\n\n")
                    
                    f.write(f"Análise de Gênero:\n")
                    gm = m['gender_metrics']
                    for gender in ['m', 'f', 'unknown']:
                        if gm[gender]['total_gt'] > 0:
                            gender_label = {'m': 'Male', 'f': 'Female', 'unknown': 'Unknown'}[gender]
                            f.write(f"  {gender_label}:\n")
                            f.write(f"    Speakers (GT):           {gm[gender]['total_gt']}\n")
                            f.write(f"    Accuracy:                {gm[gender]['accuracy']:.2f}%\n")
                            f.write(f"    Precision:               {gm[gender]['precision']:.2f}%\n")
                            f.write(f"    Recall:                  {gm[gender]['recall']:.2f}%\n")
                    f.write(f"\n")
                    
                    if m['per_speaker_metrics']:
                        f.write(f"Métricas por Speaker:\n")
                        for spk, spk_metrics in m['per_speaker_metrics'].items():
                            f.write(f"  {spk} ({spk_metrics.get('gender', 'unknown')}):\n")
                            f.write(f"    Precision: {spk_metrics['precision']:.2f}%\n")
                            f.write(f"    Recall:    {spk_metrics['recall']:.2f}%\n")
                            f.write(f"    F1-Score:  {spk_metrics['f1_score']:.2f}%\n")
                            f.write(f"    TP: {spk_metrics['true_positives']}, FP: {spk_metrics['false_positives']}, FN: {spk_metrics['false_negatives']}\n")
                        f.write(f"\n")
                    
                    if m.get('confusion_matrix'):
                        f.write(f"Confusion Matrix:\n")
                        # Exibe matriz de confusão
                        speakers = sorted(m['confusion_matrix'].keys())
                        if speakers:
                            header = "GT\\Pred"
                            f.write(f"  {header:<15}")
                            for spk in speakers:
                                f.write(f"{spk:<15}")
                            f.write(f"\n")
                            for gt_spk in speakers:
                                f.write(f"  {gt_spk:<15}")
                                for pred_spk in speakers:
                                    count = m['confusion_matrix'].get(gt_spk, {}).get(pred_spk, 0)
                                    f.write(f"{count:<15}")
                                f.write(f"\n")
                        f.write(f"\n")
                else:
                    f.write(f"❌ FALHA\n")
                    f.write(f"Erro: {result.get('error', 'Unknown error')}\n\n")
                
                f.write(f"\n")

if __name__ == "__main__":
    embeddings_dir = "../data/embeddings/individual"
    mixed_audio_dir = "../data/mixed_audio"
    output_dir = "../logs"
    
    if not os.path.exists(embeddings_dir):
        logger.error(f"❌ Pasta de embeddings não encontrada: {embeddings_dir}")
        exit(1)
    
    if not os.path.exists(mixed_audio_dir):
        logger.error(f"❌ Pasta de áudio não encontrada: {mixed_audio_dir}")
        exit(1)
    
    processor = BatchProcessor(embeddings_dir, output_dir)
    processor.process_directory(mixed_audio_dir)
    
    logger.info("\n✅ Processamento em lote concluído!")
