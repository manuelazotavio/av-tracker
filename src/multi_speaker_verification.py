import sys
from types import ModuleType

sys.modules['k2'] = ModuleType('k2')
sys.modules['_k2'] = ModuleType('_k2')
sys.modules['speechbrain.integrations.k2_fsa'] = ModuleType('speechbrain.integrations.k2_fsa')

import os
import torch
from pyannote.audio import Pipeline
from speechbrain.inference.separation import SepformerSeparation
import logging
import numpy as np
import torchaudio
import re
import torchaudio.transforms as T
from faster_whisper import WhisperModel
from .multi_speaker_verifier import MultiSpeakerVerifier
from huggingface_hub import login, snapshot_download
from difflib import SequenceMatcher
from collections import defaultdict
from datetime import datetime
import time
import spacy
from transformers import pipeline as hf_pipeline

from pyannote.audio.core.task import Specifications, Problem, Resolution

torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Specifications,
    Problem,
    Resolution
])

HF_TOKEN = os.environ.get("HF_TOKEN", "")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("pyannote").setLevel(logging.WARNING)

class LargeMeetingTranscriber:
    def __init__(
        self,
        verifier,
        hf_token,
        whisper_size="small",
        device="cuda",
        use_ai_analysis=True,
        diarization_clustering_threshold=0.6,
        verifier_confidence_min=0.8,
        min_segment_duration=0.8,
    ):
        self.verifier = verifier
        self.device = device
        self.use_ai_analysis = use_ai_analysis
        self.diarization_clustering_threshold = diarization_clustering_threshold
        self.verifier_confidence_min = verifier_confidence_min
        self.min_segment_duration = min_segment_duration
        
        logger.info("Authenticating...")
        login(token=hf_token)
        
        if use_ai_analysis:
            logger.info("Loading NER model (spaCy)...")
            try:
                self.nlp = spacy.load("pt_core_news_lg")
            except OSError:
                logger.warning("Model pt_core_news_lg not found. Trying pt_core_news_sm...")
                try:
                    self.nlp = spacy.load("pt_core_news_sm")
                except OSError:
                    logger.warning("spaCy not available. Install with: python -m spacy download pt_core_news_sm")
                    self.nlp = None
            
            logger.info("Loading LLM for contextual analysis...")
            try:
                self.llm = hf_pipeline(
                    model="google/flan-t5-small",
                    device=0 if device == "cuda" else -1,
                    max_length=512
                )
            except Exception as e:
                logger.warning(f"LLM not available: {e}")
                self.llm = None
        else:
            self.nlp = None
            self.llm = None

        logger.info("Loading PyAnnote...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(torch.device(device))

        if self.diarization_clustering_threshold is not None:
            try:
                self.pipeline.instantiate({
                    "clustering": {"threshold": self.diarization_clustering_threshold},
                    "segmentation": {"threshold": 0.4} 
                })
                logger.info(
                    f"Clustering threshold set: {self.diarization_clustering_threshold}"
                )
                logger.info("Segmentation threshold set: 0.4")
            except Exception as e:
                logger.warning(f"Failed to set thresholds: {e}")

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        logger.info("Loading SepFormer...")
        local_path = "pretrained_models/sepformer"
        sepformer_source = local_path
        if not os.path.exists(os.path.join(local_path, "hyperparams.yaml")):
            logger.warning("Local SepFormer not found. Downloading model...")
            snapshot_download(
                repo_id="speechbrain/sepformer-whamr",
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            sepformer_source = local_path

        self.separator = SepformerSeparation.from_hparams(
            source=sepformer_source,
            savedir=local_path,
            run_opts={"device": device}
        )

        logger.info(f"Loading Whisper ({whisper_size})...")
        try:
            self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="float16")
            logger.info("Using compute_type: float16")
        except Exception as e:
            logger.warning(f"float16 not supported ({e}), using int8...")
            self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="int8")
            logger.info("Using compute_type: int8")

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def normalize_audio(self, audio, target_db=-20.0):
        if len(audio) == 0:
            return audio
        
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            return audio
        
        target_amplitude = 10**(target_db/20.0)
        scaling_factor = target_amplitude / rms
        normalized = (audio * scaling_factor).astype(np.float32)
        
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val
        
        return normalized
    
    def extract_names_with_ner(self, text):
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        names = []
        
        for ent in doc.ents:
            if ent.label_ == "PER": 
                name = ent.text.strip()
                name = re.sub(r'^(Sr\.|Sra\.|Dr\.|Dra\.)\s+', '', name, flags=re.IGNORECASE)
                if len(name) > 2:
                    names.append(name)
        
        return names
    
    def analyze_conversation_context_with_llm(self, transcript):
        if not self.llm or len(transcript) == 0:
            return {}
        
        conversation_text = ""
        for entry in transcript[:20]: 
            speaker = entry['speaker']
            text = entry['text']
            conversation_text += f"{speaker}: {text}\n"
        
        prompt = f"""Analyze this English conversation and identify the REAL HUMAN NAME of each speaker.

IMPORTANT RULES:
- Only extract actual person names 
- IGNORE programming languages (Kotlin, Java, Python, etc.)
- IGNORE technical terms, frameworks, or companies
- IGNORE product names or technologies
- When someone says "PersonName, ..." at start of sentence, they are ADDRESSING that person
- When someone says "my name is PersonName" they are introducing themselves

Conversation:
{conversation_text}

For each speaker that you can identify a real human name, write:
SPEAKER_XX=ActualHumanName

Answer (only list speakers with confirmed human names):"""
        
        try:
            result = self.llm(prompt, max_length=200, do_sample=False)
            response = result[0]['generated_text']
            
            speaker_names = {}
            invalid_names = {'name', 'another', 'person', 'speaker', 'someone', 'user', 'kotlin', 'java', 'python'}
            
            matches = re.findall(r'(SPEAKER_\d+)\s*=\s*([A-ZÀ-Ú][a-zà-ú]+)', response)
            for speaker_id, name in matches:
                if name.lower() not in invalid_names and len(name) >= 3:
                    speaker_names[speaker_id] = name
                    logger.info(f"LLM detected: {speaker_id} = {name}")
                else:
                    logger.debug(f"LLM returned invalid name: {name}")
            
            return speaker_names
        except Exception as e:
            logger.warning(f"LLM analysis error: {e}")
            return {}
    
    def associate_names_with_speakers(self, transcript):
        speaker_names = {}
        
        if self.llm:
            llm_names = self.analyze_conversation_context_with_llm(transcript)
            speaker_names.update(llm_names)
        
        speaker_mentions = defaultdict(lambda: defaultdict(int)) 
        vocative_mentions = defaultdict(lambda: defaultdict(int)) 
        self_introductions = {} 
        
        for i, entry in enumerate(transcript):
            speaker = entry['speaker']
            text = entry['text']
            
            # Portuguese self-introduction patterns:
            # "meu nome é" = "my name is", "me chamo" = "I'm called",
            # "eu sou" = "I am", "sou o/sou a" = "I am (masc/fem)"
            # "aqui é" = "this is", "aqui quem fala é" = "the one speaking here is"
            self_intro_patterns = [
                r'(?:meu nome é|me chamo|eu sou|sou o|sou a)\s+([A-ZÀ-Ú][a-zà-ú]+)',
                r'(?:aqui é|aqui quem fala é)\s+(?:o|a)?\s*([A-ZÀ-Ú][a-zà-ú]+)',
            ]
            
            for pattern in self_intro_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    name = match.group(1).capitalize()
                    self_introductions[speaker] = name
                    logger.info(f"Self-introduction: {speaker} = {name}")
            
            detected_names = []
            if self.nlp:
                detected_names = self.extract_names_with_ner(text)
                if detected_names:
                    logger.debug(f"NER extracted from '{text[:50]}...': {detected_names}")
            
            regex_name_pattern = r'^\s*([A-ZÀ-Ú][a-zà-úç]{2,15})\s*[,!]'
            regex_match = re.search(regex_name_pattern, text)
            if regex_match:
                potential_name = regex_match.group(1)
                common_names = {
                    'lucas', 'mateus', 'pedro', 'joão', 'gabriel', 'rafael', 'bruno', 'carlos', 'andré',
                    'fernando', 'ricardo', 'rodrigo', 'marcelo', 'paulo', 'thiago', 'vitor', 'daniel',
                    'maria', 'ana', 'julia', 'beatriz', 'leticia', 'amanda', 'carolina', 'fernanda',
                    'mariana', 'juliana', 'patricia', 'camila', 'bruna', 'aline', 'bianca', 'renata',
                    'augusto', 'leonardo', 'eduardo', 'henrique', 'diego', 'felipe', 'guilherme', 'manuela', 'kauan'
                }
                tech_terms = {'kotlin', 'java', 'python', 'javascript', 'react', 'typescript', 'node'}
                
                if potential_name.lower() in common_names and potential_name.lower() not in tech_terms:
                    if potential_name not in detected_names:
                        detected_names.append(potential_name)
                        logger.debug(f"Regex detected name: '{potential_name}' in '{text[:50]}...'")

            for name in detected_names:
                vocative_pattern = rf'^\s*{re.escape(name)}\s*[,!]'
                if re.search(vocative_pattern, text, re.IGNORECASE):
                    target_speaker = None
                    
                    if i + 1 < len(transcript):
                        target_speaker = transcript[i + 1]['speaker']
                        logger.info(f"Vocative detected: '{name}' in '{text[:60]}...' -> next speaker is {target_speaker}")
                    else:
                        all_speakers = set(e['speaker'] for e in transcript)
                        other_speakers = all_speakers - {speaker}
                        if len(other_speakers) == 1:
                            target_speaker = list(other_speakers)[0]
                            logger.info(f"Vocative detected (last turn): '{name}' in '{text[:60]}...' -> other speaker is {target_speaker}")
                        elif len(other_speakers) > 1:
                            for j in range(i - 1, -1, -1):
                                if transcript[j]['speaker'] != speaker:
                                    target_speaker = transcript[j]['speaker']
                                    logger.info(f"Vocative detected: '{name}' in '{text[:60]}...' -> previous speaker is {target_speaker}")
                                    break
                    
                    if target_speaker:
                        vocative_mentions[name][target_speaker] += 2 
                else:
                    if i + 1 < len(transcript):
                        next_speaker = transcript[i + 1]['speaker']
                        speaker_mentions[name][next_speaker] += 1
        
        speaker_names.update(self_introductions)
        
        for name, responding_speakers in vocative_mentions.items():
            if responding_speakers:
                most_likely_speaker = max(responding_speakers.items(), key=lambda x: x[1])[0]
                if most_likely_speaker not in speaker_names:
                    speaker_names[most_likely_speaker] = name
                    logger.info(f"Vocative: {most_likely_speaker} = {name} (called {responding_speakers[most_likely_speaker]}x)")
        
        invalid_terms = {'kotlin', 'java', 'python', 'javascript', 'react', 'angular', 'node'}
        for name, responding_speakers in speaker_mentions.items():
            if responding_speakers and name.lower() not in invalid_terms:
                most_likely_speaker = max(responding_speakers.items(), key=lambda x: x[1])[0]
                if most_likely_speaker not in speaker_names:
                    if responding_speakers[most_likely_speaker] >= 2:
                        speaker_names[most_likely_speaker] = name
                        logger.info(f"Contextual: {most_likely_speaker} = {name} (mentioned {responding_speakers[most_likely_speaker]}x before responding)")
        
        return speaker_names
    
    def save_new_embeddings(self, unknown_speakers_audio, speaker_names, embeddings_dir):
        from speechbrain.inference.speaker import EncoderClassifier
        
        if not unknown_speakers_audio:
            return
        
        logger.info("\nCreating embeddings for new speakers...")
        
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        
        saved_count = 0
        
        for speaker_id, audio_chunks in unknown_speakers_audio.items():
            actual_name = speaker_names.get(speaker_id, None)
            
            if not actual_name:
                logger.info(f"{speaker_id}: No name detected, skipping...")
                continue
            
            if len(audio_chunks) == 0:
                continue
                
            combined_audio = np.concatenate(audio_chunks)
            
            mx = np.abs(combined_audio).max()
            if mx > 0:
                combined_audio = combined_audio / mx
            
            signal = torch.from_numpy(combined_audio).float().to(self.device)
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            
            with torch.no_grad():
                embedding = classifier.encode_batch(signal).squeeze().cpu().numpy()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{actual_name}_{timestamp}.npy"
            filepath = os.path.join(embeddings_dir, filename)
            
            np.save(filepath, embedding)
            logger.info(f"Saved: {filepath}")
            saved_count += 1
        
        if saved_count > 0:
            logger.info(f"\n{saved_count} new embedding(s) created successfully!")
            logger.info("Restart the system to load new embeddings.")

    def process_audio(self, audio_path, embeddings_dir=None, num_speakers=None):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
        
        start_total = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {audio_path}")
        if num_speakers:
            logger.info(f"Limiting to MAXIMUM {num_speakers} speaker(s)")
        logger.info(f"{'='*60}")
        
        start = time.time()
        waveform, sample_rate = torchaudio.load(audio_path)
        elapsed = time.time() - start
        logger.info(f"Audio loading: {elapsed:.2f}s")
        
        start = time.time()
        diarization_params = {}
        if num_speakers:
            diarization_params['min_speakers'] = num_speakers
            diarization_params['max_speakers'] = num_speakers
        
        diarization = self.pipeline(audio_path, **diarization_params)
        overlap_timeline = diarization.get_overlap()
        elapsed = time.time() - start
        logger.info(f"Diarization (speaker identification): {elapsed:.2f}s")
        
        resample_to_8k = T.Resample(sample_rate, 8000)
        resample_to_16k = T.Resample(8000, 16000)
        resample_orig_to_16k = T.Resample(sample_rate, 16000)
        
        full_transcript = []
        speaker_history = {}
        unknown_speakers_audio = defaultdict(list) 
        
        total_segments = 0
        transcribed_segments = 0
        
        start_transcription = time.time()
        logger.info("\nTranscribing segments...")

        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            
            if duration < self.min_segment_duration:
                logger.debug(
                    f"[{turn.start:.1f}s] Segment too short ({duration:.2f}s), skipping..."
                )
                continue
            
            effective_end = turn.end
            if duration > 30.0:
                logger.debug(f"[{turn.start:.1f}s] Segment too long ({duration:.2f}s), limiting to 30s...")
                effective_end = turn.start + 30.0
            
            total_segments += 1
            logger.debug(f"[{turn.start:.1f}s - {effective_end:.1f}s] Processing {speaker_id} ({effective_end - turn.start:.2f}s)")
            
            start_frame = int(turn.start * sample_rate)
            end_frame = int(effective_end * sample_rate)
            if start_frame >= waveform.shape[1]:
                logger.debug(f"[{turn.start:.1f}s] Frame out of bounds, skipping...")
                continue
            
            audio_segment = waveform[:, start_frame:end_frame]
            
            overlap_duration = 0.0
            for ov in overlap_timeline:
                inter = turn & ov
                if inter:
                    overlap_duration += inter.duration
            is_overlap = (overlap_duration / (turn.end - turn.start)) > 0.30

            sources_to_transcribe = []

            if is_overlap:
                start_sep = time.time()
                logger.info(f"Overlap at {turn.start:.1f}s. Separating...")
                mono = torch.mean(audio_segment, dim=0) if audio_segment.shape[0] > 1 else audio_segment.squeeze(0)
                mono_8k = resample_to_8k(mono)
                mono_8k_input = mono_8k.unsqueeze(0)

                try:
                    est_sources = self.separator.separate_batch(mono_8k_input)
                    src1_8k = est_sources[0, :, 0].detach().cpu()
                    src2_8k = est_sources[0, :, 1].detach().cpu()

                    src1_16k = resample_to_16k(src1_8k).numpy()
                    src2_16k = resample_to_16k(src2_8k).numpy()
                    
                    sources_to_transcribe = [src1_16k, src2_16k]
                    elapsed_sep = time.time() - start_sep
                    logger.info(f"Separation: {elapsed_sep:.2f}s")
                except Exception as e:
                    logger.error(f"Separation failed: {e}. Using original audio.")
                    sources_to_transcribe = [resample_orig_to_16k(audio_segment).squeeze().numpy()]
            else:
                audio_16k = resample_orig_to_16k(audio_segment)
                sources_to_transcribe = [audio_16k.squeeze().numpy()]

            for idx, source_np in enumerate(sources_to_transcribe):
                source_np = source_np.squeeze()
                
                if source_np.ndim == 2:
                    logger.debug(f"Converting stereo to mono...")
                    source_np = np.mean(source_np, axis=0)
                
                if source_np.ndim != 1 or source_np.size == 0:
                    logger.debug(f"[{turn.start:.1f}s] Voice {idx+1}: Invalid audio (ndim={source_np.ndim}, size={source_np.size}), skipping...")
                    continue
                
                mx = np.abs(source_np).max()
                if np.isnan(mx) or mx == 0:
                    logger.debug(f"[{turn.start:.1f}s] Voice {idx+1}: Silent audio (max={mx}), skipping...")
                    continue
                source_np = source_np / mx
                
                real_name, conf = self.verifier._process_audio_chunk(source_np, sample_rate=16000)
                
                if real_name == "Unknown":
                    unknown_speakers_audio[speaker_id].append(source_np.copy())
                
                if is_overlap:
                    print(f"[DEBUG] Voice {idx+1}: {real_name} ({conf:.1%})")

                verified_name = None
                if real_name != "Unknown" and conf >= self.verifier_confidence_min:
                    verified_name = real_name

                disp = speaker_id
                display_label = disp if not verified_name else f"{disp} ({verified_name})"
                
                try:
                    source_normalized = self.normalize_audio(source_np)
                    
                    if isinstance(source_normalized, np.ndarray):
                        audio_for_whisper = source_normalized
                    else:
                        audio_for_whisper = source_normalized.cpu().numpy()
                    
                    logger.debug(f"Audio shape: {audio_for_whisper.shape}, dtype: {audio_for_whisper.dtype}")
                    
                    start_whisper = time.time()
                    segs, _ = self.whisper.transcribe(
                        audio_for_whisper, 
                        language="pt", 
                        beam_size=1,
                        vad_filter=False,
                        without_timestamps=True
                    )
                    text = " ".join([s.text for s in segs]).strip()
                    elapsed_whisper = time.time() - start_whisper
                    
                    logger.debug(f"Whisper: {elapsed_whisper:.2f}s, result: '{text[:50]}...'")
                    
                    if not text:
                        logger.debug(f"[{turn.start:.1f}s] {speaker_id}: Whisper returned empty, skipping...")
                        continue
                    
                    if "Amara.org" in text:
                        logger.debug(f"[{turn.start:.1f}s] {speaker_id}: Watermark detected, skipping...")
                        continue
                    
                    clean_new = self.normalize_text(text)
                    last_entry = speaker_history.get(disp)
                    
                    is_duplicate = False
                    if last_entry and clean_new:
                        time_diff = turn.start - last_entry['time']
                        if time_diff < 1.0: 
                            clean_old = last_entry['clean_text']
                            ratio = SequenceMatcher(None, clean_new, clean_old).ratio()
                            if ratio > 0.8: 
                                is_duplicate = True
                    
                    if is_duplicate:
                        logger.debug(f"[{turn.start:.1f}s] {disp}: Duplicate (similar>{ratio:.0%}), skipping...")
                        continue
                    
                    transcribed_segments += 1
                    speaker_history[disp] = {'clean_text': clean_new, 'time': turn.start}

                    suffix = f" (Voice {idx+1})" if is_overlap else ""
                    logger.debug(f"[{turn.start:.1f}s] {display_label}{suffix}: TRANSCRIBED")
                    print(f" [{turn.start:.1f}s] {display_label}{suffix}: {text}")
                    
                    full_transcript.append({
                        "speaker": disp,
                        "verified_name": verified_name,
                        "text": text,
                        "start": turn.start,
                        "suffix": suffix
                    })
                except Exception as e:
                    logger.error(f"Whisper error: {e}")
        
        elapsed_transcription = time.time() - start_transcription
        logger.info(f"\nTotal transcription: {elapsed_transcription:.2f}s")
        
        coverage_rate = (transcribed_segments / total_segments * 100) if total_segments > 0 else 0
        unique_speakers = len(set(line['speaker'] for line in full_transcript))
        logger.info(f"\nTRANSCRIPTION STATISTICS:")
        logger.info(f"Segments processed: {total_segments}")
        logger.info(f"Segments transcribed: {transcribed_segments}")
        logger.info(f"Coverage rate: {coverage_rate:.1f}%")
        logger.info(f"Unique speakers: {unique_speakers}")
        
        if unknown_speakers_audio:
            start_names = time.time()
            logger.info(f"\nDetected {len(unknown_speakers_audio)} unknown speaker(s)")
            logger.info("Analyzing transcript to identify names...")
            
            speaker_names = self.associate_names_with_speakers(full_transcript)
            
            if speaker_names:
                elapsed_names = time.time() - start_names
                logger.info(f"\nNames identified: {speaker_names}")
                logger.info(f"Name detection: {elapsed_names:.2f}s")
                
                embeddings_dir = embeddings_dir or "../data/embeddings"
                if not os.path.exists(embeddings_dir):
                    os.makedirs(embeddings_dir)
                
                start_embeddings = time.time()
                self.save_new_embeddings(unknown_speakers_audio, speaker_names, embeddings_dir)
                elapsed_embeddings = time.time() - start_embeddings
                logger.info(f"Embedding creation: {elapsed_embeddings:.2f}s")
            else:
                logger.info("No name was identified in the conversation")
        
        elapsed_total = time.time() - start_total
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL TIME: {elapsed_total:.2f}s")
        logger.info(f"{'='*60}")
        
        return full_transcript

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_dirs = [
        os.path.join(base_dir, "data", "embeddings"),
        os.path.join(base_dir, "data", "embeddings", "individual"),
    ]
    embeddings_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
    if not embeddings_dir:
        print("Embeddings directory not found")
        exit()

    print("Initializing...")
    try:
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.65)
        system = LargeMeetingTranscriber(
            verifier,
            HF_TOKEN,
            diarization_clustering_threshold=0.45,
            verifier_confidence_min=0.9,
            min_segment_duration=0.6,
        )

        while True:
            f = input("\nFile: ").strip()
            if f in ['sair', 'exit']:
                break
            path = f if os.path.exists(f) else f"../data/testes/{f}"
            
            if os.path.exists(path):
                num_speakers_input = input("MAXIMUM number of speakers (Enter for automatic detection): ").strip()
                num_speakers = None
                if num_speakers_input.isdigit():
                    num_speakers = int(num_speakers_input)
                    print(f"Limiting to maximum {num_speakers} speaker(s)")
                else:
                    print("Automatic speaker detection")
                
                transcript_result = system.process_audio(path, embeddings_dir, num_speakers=num_speakers)
                
                base_name = os.path.splitext(os.path.basename(path))[0]
                output_filename = f"{base_name}_transcript.txt"
                
                if transcript_result:
                    print(f"\nSaving transcript to: {output_filename} ...")
                    with open(output_filename, "w", encoding="utf-8") as txt_file:
                        txt_file.write(f"FILE: {f}\n")
                        txt_file.write("-" * 50 + "\n\n")
                        
                        for line in transcript_result:
                            timestamp = f"[{line['start']:.1f}s]"
                            speaker = line['speaker']
                            verified_name = line.get('verified_name')
                            suffix = line.get('suffix', '')
                            text = line['text']
                            if verified_name:
                                txt_file.write(f"{timestamp} {speaker} ({verified_name}){suffix}: {text}\n")
                            else:
                                txt_file.write(f"{timestamp} {speaker}{suffix}: {text}\n")
                    
                    print(f"File '{output_filename}' saved successfully!")
                else:
                    print("No speech detected.")
            else:
                print("File does not exist.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")