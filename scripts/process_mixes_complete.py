"""
Processamento COMPLETO dos mixes com:
- Diarização (PyAnnote)
- Separação de vozes sobrepostas (SepFormer)
- Transcrição (Whisper)
- Speaker Verification (ECAPA-TDNN)
- Métricas detalhadas
"""
import os
import sys
import torch
import logging
import csv
import json
from pathlib import Path
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importar módulos necessários
from src.multi_speaker_verifier import MultiSpeakerVerifier

# Importar as classes necessárias diretamente
import torchaudio
import torchaudio.transforms as T
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from speechbrain.inference.separation import SepformerSeparation
from huggingface_hub import login
import numpy as np
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv('HF_TOKEN', "")

class SimpleLargeMeetingTranscriber:
    """Versão simplificada do LargeMeetingTranscriber para processar os mixes"""
    
    def __init__(self, verifier, hf_token, whisper_size="base", device="cuda"):
        self.verifier = verifier
        self.device = device
        
        if hf_token:
            logger.info("Autenticando no HuggingFace...")
            login(token=hf_token)
        
        logger.info("Carregando PyAnnote Diarization...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token if hf_token else None
            ).to(torch.device(device))
        except Exception as e:
            logger.warning(f"Erro ao carregar PyAnnote: {e}")
            logger.warning("Continuando sem diarização...")
            self.pipeline = None
        
        logger.info(f"Carregando Whisper ({whisper_size})...")
        self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="int8")
        
        logger.info("Sistema inicializado!")
    
    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def process_audio(self, audio_path):
        """Processa o áudio com diarização e transcrição"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        logger.info(f"Processando: {os.path.basename(audio_path)}")
        
        # Se não tem PyAnnote, fazer processamento simples
        if not self.pipeline:
            return self._process_simple(audio_path)
        
        # Fazer diarização
        diarization = self.pipeline(audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        resample_orig_to_16k = T.Resample(sample_rate, 16000)
        
        full_transcript = []
        speaker_history = {}
        
        logger.info("Transcrevendo segmentos...")
        
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            if (turn.end - turn.start) < 0.5:
                continue
            
            start_frame = int(turn.start * sample_rate)
            end_frame = int(turn.end * sample_rate)
            if start_frame >= waveform.shape[1]:
                continue
            
            audio_segment = waveform[:, start_frame:end_frame]
            audio_16k = resample_orig_to_16k(audio_segment)
            audio_np = audio_16k.squeeze().numpy()
            
            # Normalizar
            mx = np.abs(audio_np).max()
            if mx > 0:
                audio_np = audio_np / mx
            
            # Identificar speaker
            real_name, conf, _ = self.verifier._process_audio_chunk(audio_np, sample_rate=16000)
            disp = real_name if real_name != "Unknown" else speaker_id
            
            # Transcrever
            try:
                segs, _ = self.whisper.transcribe(audio_np, language="en", beam_size=1, vad_filter=True)
                text = " ".join([s.text for s in segs]).strip()
                
                if text and "Amara.org" not in text:
                    clean_new = self.normalize_text(text)
                    last_entry = speaker_history.get(disp)
                    is_duplicate = False
                    
                    if last_entry and clean_new:
                        clean_old = last_entry['clean_text']
                        if clean_new in clean_old or clean_old in clean_new:
                            is_duplicate = True
                        else:
                            ratio = SequenceMatcher(None, clean_new, clean_old).ratio()
                            if ratio > 0.5:
                                is_duplicate = True
                    
                    if not is_duplicate:
                        speaker_history[disp] = {'clean_text': clean_new, 'time': turn.end}
                        
                        full_transcript.append({
                            "speaker": disp,
                            "text": text,
                            "start": turn.start,
                            "end": turn.end,
                            "confidence": conf
                        })
            except Exception as e:
                logger.error(f"Erro Whisper: {e}")
        
        return full_transcript
    
    def _process_simple(self, audio_path):
        """Processamento simples sem diarização"""
        logger.info("Processando sem diarização...")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        audio_np = waveform.squeeze().numpy()
        
        # Transcrever o áudio inteiro
        segs, _ = self.whisper.transcribe(audio_np, language="en", beam_size=1, vad_filter=True)
        text = " ".join([s.text for s in segs]).strip()
        
        # Identificar speaker do áudio inteiro
        speaker, conf, _ = self.verifier._process_audio_chunk(audio_np, sample_rate=16000)
        
        return [{
            "speaker": speaker,
            "text": text,
            "start": 0.0,
            "end": len(audio_np) / 16000,
            "confidence": conf
        }]

def load_ground_truth(gt_file):
    """Carrega o arquivo ground truth."""
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            speakers = []
            for line in lines:
                if line.startswith("Speakers:"):
                    speaker_names = [s.strip() for s in line.replace("Speakers:", "").strip().split(",")]
                    speakers = speaker_names
                    break
            return speakers
    except Exception as e:
        logger.warning(f"Erro ao ler ground truth: {e}")
        return []

def calculate_metrics(transcript, ground_truth):
    """Calcula métricas detalhadas"""
    # Extrair speakers identificados
    identified_speakers = set()
    speaker_segments = {}
    total_duration = 0
    
    for entry in transcript:
        speaker = entry['speaker']
        if " (Voz" in speaker:
            speaker = speaker.split(" (Voz")[0]
        
        identified_speakers.add(speaker)
        
        duration = entry.get('end', 0) - entry.get('start', 0)
        if speaker not in speaker_segments:
            speaker_segments[speaker] = {'count': 0, 'duration': 0, 'texts': []}
        
        speaker_segments[speaker]['count'] += 1
        speaker_segments[speaker]['duration'] += duration
        speaker_segments[speaker]['texts'].append(entry['text'])
        total_duration += duration
    
    # Calcular accuracy
    matches = sum(1 for gt in ground_truth if gt in identified_speakers)
    accuracy = matches / len(ground_truth) if ground_truth else 0.0
    
    # Precision e Recall
    precision = matches / len(identified_speakers) if identified_speakers else 0.0
    recall = accuracy  # Same as accuracy in this context
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'identified_speakers': list(identified_speakers),
        'num_speakers_identified': len(identified_speakers),
        'num_speakers_gt': len(ground_truth),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_segments': len(transcript),
        'total_duration': total_duration,
        'speaker_segments': speaker_segments
    }

def main():
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
    MIXES_DIR = BASE_DIR / "data" / "mixed_audio"
    OUTPUT_DIR = BASE_DIR / "data" / "transcriptions"
    OUTPUT_CSV = BASE_DIR / "mix_processing_results_complete.csv"
    
    # Criar diretório de saída
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar diretórios
    if not EMBEDDINGS_DIR.exists():
        logger.error(f"Pasta de embeddings não encontrada: {EMBEDDINGS_DIR}")
        return
    
    if not MIXES_DIR.exists():
        logger.error(f"Pasta de mixes não encontrada: {MIXES_DIR}")
        return
    
    if not HF_TOKEN:
        logger.warning("HF_TOKEN não configurado. Tentando sem autenticação...")
    
    # Inicializar verifier
    logger.info("Inicializando MultiSpeakerVerifier...")
    verifier = MultiSpeakerVerifier(str(EMBEDDINGS_DIR), threshold=0.5)
    
    # Inicializar transcriber
    logger.info("Inicializando Transcriber...")
    transcriber = SimpleLargeMeetingTranscriber(verifier, HF_TOKEN, whisper_size="base", device="cuda")
    
    # Coletar arquivos
    mix_files = sorted(MIXES_DIR.glob("**/mix_*.wav"))
    logger.info(f"Encontrados {len(mix_files)} arquivos para processar\n")
    
    results = []
    
    for idx, mix_file in enumerate(mix_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{len(mix_files)}] Processando {mix_file.name}")
        logger.info(f"{'='*60}")
        
        # Carregar ground truth
        gt_file = mix_file.with_name(f"{mix_file.stem}_ground_truth.txt")
        ground_truth = []
        if gt_file.exists():
            ground_truth = load_ground_truth(gt_file)
            logger.info(f"Ground Truth: {ground_truth}")
        
        try:
            # Processar o áudio
            transcript = transcriber.process_audio(str(mix_file))
            
            # Calcular métricas
            metrics = calculate_metrics(transcript, ground_truth)
            
            # Salvar transcrição
            output_file = OUTPUT_DIR / f"{mix_file.stem}_transcription.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Arquivo: {mix_file.name}\n")
                f.write(f"Ground Truth: {', '.join(ground_truth)}\n")
                f.write(f"Speakers Identificados: {', '.join(metrics['identified_speakers'])}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.1%}\n")
                f.write(f"Precision: {metrics['precision']:.1%}\n")
                f.write(f"Recall: {metrics['recall']:.1%}\n")
                f.write(f"F1-Score: {metrics['f1_score']:.1%}\n")
                f.write(f"Total Segmentos: {metrics['total_segments']}\n")
                f.write(f"{'-'*60}\n\n")
                
                for entry in transcript:
                    timestamp = f"[{entry['start']:.1f}s-{entry.get('end', 0):.1f}s]"
                    speaker = entry['speaker']
                    text = entry['text']
                    conf = entry.get('confidence', 0)
                    f.write(f"{timestamp} {speaker} ({conf:.2f}): {text}\n")
            
            logger.info(f"[OK] Transcrição salva: {output_file.name}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.1%} | F1: {metrics['f1_score']:.1%}")
            
            # Adicionar resultado
            results.append({
                'audio_file': mix_file.name,
                'ground_truth': ground_truth,
                **metrics
            })
            
        except Exception as e:
            logger.error(f"Erro ao processar {mix_file.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'audio_file': mix_file.name,
                'ground_truth': ground_truth,
                'identified_speakers': [],
                'num_speakers_identified': 0,
                'num_speakers_gt': len(ground_truth),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'total_segments': 0,
                'error': str(e)
            })
    
    # Salvar CSV
    logger.info(f"\nSalvando resultados em {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'audio_file', 'ground_truth_speakers', 'identified_speakers',
            'num_speakers_gt', 'num_speakers_identified',
            'accuracy', 'precision', 'recall', 'f1_score', 'total_segments'
        ])
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'audio_file': result['audio_file'],
                'ground_truth_speakers': '|'.join(result['ground_truth']),
                'identified_speakers': '|'.join(result.get('identified_speakers', [])),
                'num_speakers_gt': result['num_speakers_gt'],
                'num_speakers_identified': result.get('num_speakers_identified', 0),
                'accuracy': result.get('accuracy', 0.0),
                'precision': result.get('precision', 0.0),
                'recall': result.get('recall', 0.0),
                'f1_score': result.get('f1_score', 0.0),
                'total_segments': result.get('total_segments', 0)
            })
    
    # Estatísticas
    total = len(results)
    avg_accuracy = np.mean([r.get('accuracy', 0) for r in results])
    avg_precision = np.mean([r.get('precision', 0) for r in results])
    avg_recall = np.mean([r.get('recall', 0) for r in results])
    avg_f1 = np.mean([r.get('f1_score', 0) for r in results])
    
    print(f"\n{'='*60}")
    print(f"[OK] PROCESSAMENTO COMPLETO!")
    print(f"{'='*60}")
    print(f"Total de arquivos: {total}")
    print(f"Accuracy média: {avg_accuracy:.1%}")
    print(f"Precision média: {avg_precision:.1%}")
    print(f"Recall média: {avg_recall:.1%}")
    print(f"F1-Score média: {avg_f1:.1%}")
    print(f"\nTranscrições: {OUTPUT_DIR}")
    print(f"Métricas CSV: {OUTPUT_CSV}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
