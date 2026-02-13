import os
import torch
import logging
import numpy as np
import torchaudio
import re
import torchaudio.transforms as T
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from speechbrain.inference.separation import SepformerSeparation
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

HF_TOKEN = "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"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silencia logs verbosos de bibliotecas externas
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
        
        logger.info("Autenticando...")
        login(token=hf_token)
        
        # Inicializa NER com spaCy para português
        if use_ai_analysis:
            logger.info("Carregando modelo NER (spaCy)...")
            try:
                self.nlp = spacy.load("pt_core_news_lg")
            except OSError:
                logger.warning("Modelo pt_core_news_lg não encontrado. Tentando pt_core_news_sm...")
                try:
                    self.nlp = spacy.load("pt_core_news_sm")
                except OSError:
                    logger.warning("⚠️ spaCy não disponível. Instale com: python -m spacy download pt_core_news_sm")
                    self.nlp = None
            
            # Carrega LLM para análise contextual (modelo pequeno e rápido)
            logger.info("Carregando LLM para análise contextual...")
            try:
                self.llm = hf_pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    device=0 if device == "cuda" else -1,
                    max_length=512
                )
            except Exception as e:
                logger.warning(f"⚠️ LLM não disponível: {e}")
                self.llm = None
        else:
            self.nlp = None
            self.llm = None

        logger.info("Carregando PyAnnote...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(torch.device(device))

        if self.diarization_clustering_threshold is not None:
            try:
                self.pipeline.instantiate({
                    "clustering": {"threshold": self.diarization_clustering_threshold},
                    "segmentation": {"threshold": 0.4}  # Reduz threshold de segmentação
                })
                logger.info(
                    f"✅ Clustering threshold ajustado: {self.diarization_clustering_threshold}"
                )
                logger.info("✅ Segmentation threshold ajustado: 0.4")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao ajustar thresholds: {e}")

        # Evita erro de symlink no Windows durante download do modelo
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        logger.info("Carregando SepFormer...")
        local_path = "pretrained_models/sepformer"
        sepformer_source = local_path
        if not os.path.exists(os.path.join(local_path, "hyperparams.yaml")):
            logger.warning("SepFormer local nao encontrado. Baixando modelo...")
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

        logger.info(f"Carregando Whisper ({whisper_size})...")
        try:
            self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="float16")
            logger.info("✅ Usando compute_type: float16")
        except Exception as e:
            logger.warning(f"⚠️ float16 não suportado ({e}), usando int8...")
            self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="int8")
            logger.info("✅ Usando compute_type: int8")

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def normalize_audio(self, audio, target_db=-20.0):
        """Normaliza áudio para nível padrão melhorando qualidade"""
        if len(audio) == 0:
            return audio
        
        # Calcula RMS (root mean square)
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6:
            return audio
        
        # Normaliza para target_db
        target_amplitude = 10**(target_db/20.0)
        scaling_factor = target_amplitude / rms
        normalized = (audio * scaling_factor).astype(np.float32)
        
        # Evita clipping
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val
        
        return normalized
    
    def extract_names_with_ner(self, text):
        """Extrai nomes próprios usando NER (Named Entity Recognition)"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        names = []
        
        for ent in doc.ents:
            if ent.label_ == "PER":  # Pessoa
                # Limpa o nome
                name = ent.text.strip()
                # Remove títulos comuns
                name = re.sub(r'^(Sr\.|Sra\.|Dr\.|Dra\.)\s+', '', name, flags=re.IGNORECASE)
                if len(name) > 2:
                    names.append(name)
        
        return names
    
    def analyze_conversation_context_with_llm(self, transcript):
        """Usa LLM para analisar o contexto e identificar nomes"""
        if not self.llm or len(transcript) == 0:
            return {}
        
        # Cria um resumo da conversa com contexto
        conversation_text = ""
        for entry in transcript[:20]:  # Analisa primeiros 20 turnos (geralmente apresentações)
            speaker = entry['speaker']
            text = entry['text']
            conversation_text += f"{speaker}: {text}\n"
        
        # Prompt para o LLM
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
            
            # Parse a resposta
            speaker_names = {}
            invalid_names = {'name', 'another', 'person', 'speaker', 'someone', 'user', 'kotlin', 'java', 'python'}
            
            matches = re.findall(r'(SPEAKER_\d+)\s*=\s*([A-ZÀ-Ú][a-zà-ú]+)', response)
            for speaker_id, name in matches:
                # Valida se é um nome real
                if name.lower() not in invalid_names and len(name) >= 3:
                    speaker_names[speaker_id] = name
                    logger.info(f"🤖 LLM detectou: {speaker_id} = {name}")
                else:
                    logger.debug(f"⚠️ LLM retornou nome inválido: {name}")
            
            return speaker_names
        except Exception as e:
            logger.warning(f"⚠️ Erro na análise LLM: {e}")
            return {}
    
    def associate_names_with_speakers(self, transcript):
        """Analisa transcrição para associar nomes a speakers usando IA"""
        speaker_names = {}
        
        # Estratégia 1: Análise com LLM (mais inteligente)
        if self.llm:
            llm_names = self.analyze_conversation_context_with_llm(transcript)
            speaker_names.update(llm_names)
        
        # Estratégia 2: NER + Análise de contexto
        speaker_mentions = defaultdict(lambda: defaultdict(int))  # {mentioned_name: {next_speaker: count}}
        vocative_mentions = defaultdict(lambda: defaultdict(int))  # {mentioned_name: {current_speaker: count}}
        self_introductions = {}  # {speaker: name}
        
        for i, entry in enumerate(transcript):
            speaker = entry['speaker']
            text = entry['text']
            
            # Detecta auto-apresentações com regex (ainda útil)
            self_intro_patterns = [
                r'(?:meu nome é|me chamo|eu sou|sou o|sou a)\s+([A-ZÀ-Ú][a-zà-ú]+)',
                r'(?:aqui é|aqui quem fala é)\s+(?:o|a)?\s*([A-ZÀ-Ú][a-zà-ú]+)',
            ]
            
            for pattern in self_intro_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    name = match.group(1).capitalize()
                    self_introductions[speaker] = name
                    logger.info(f"✅ Auto-apresentação: {speaker} = {name}")
            
            # Extrai nomes com NER
            detected_names = []
            if self.nlp:
                detected_names = self.extract_names_with_ner(text)
                if detected_names:
                    logger.debug(f"📝 NER extraiu de '{text[:50]}...': {detected_names}")
            
            # Fallback: Detecta nomes próprios brasileiros comuns por regex
            # Padrão: Nome com inicial maiúscula no início de frase seguido de vírgula/exclamação
            regex_name_pattern = r'^\s*([A-ZÀ-Ú][a-zà-úç]{2,15})\s*[,!]'
            regex_match = re.search(regex_name_pattern, text)
            if regex_match:
                potential_name = regex_match.group(1)
                # Lista de nomes brasileiros comuns (expandível)
                common_names = {
                    'lucas', 'mateus', 'pedro', 'joão', 'gabriel', 'rafael', 'bruno', 'carlos', 'andré',
                    'fernando', 'ricardo', 'rodrigo', 'marcelo', 'paulo', 'thiago', 'vitor', 'daniel',
                    'maria', 'ana', 'julia', 'beatriz', 'leticia', 'amanda', 'carolina', 'fernanda',
                    'mariana', 'juliana', 'patricia', 'camila', 'bruna', 'aline', 'bianca', 'renata',
                    'augusto', 'leonardo', 'eduardo', 'henrique', 'diego', 'felipe', 'guilherme', 'manuela', 'kauan'
                }
                # Filtro de termos técnicos
                tech_terms = {'kotlin', 'java', 'python', 'javascript', 'react', 'typescript', 'node'}
                
                if potential_name.lower() in common_names and potential_name.lower() not in tech_terms:
                    if potential_name not in detected_names:
                        detected_names.append(potential_name)
                        logger.debug(f"🔍 Regex detectou nome: '{potential_name}' em '{text[:50]}...'")
            
            # Analisa vocativos e menções
            for name in detected_names:
                # Detecta vocativo (chamando alguém): "Lucas, ..." no início da frase
                # Regex mais robusto: captura nome seguido de vírgula/exclamação no início
                vocative_pattern = rf'^\s*{re.escape(name)}\s*[,!]'
                if re.search(vocative_pattern, text, re.IGNORECASE):
                    # É um vocativo - a pessoa está SE DIRIGINDO ao nome mencionado
                    target_speaker = None
                    
                    if i + 1 < len(transcript):
                        # Próximo speaker é provavelmente a pessoa sendo chamada
                        target_speaker = transcript[i + 1]['speaker']
                        logger.info(f"🎯 Vocativo detectado: '{name}' em '{text[:60]}...' -> próximo speaker é {target_speaker}")
                    else:
                        # Último turno, mas há vocativo - identifica "o outro speaker"
                        all_speakers = set(e['speaker'] for e in transcript)
                        other_speakers = all_speakers - {speaker}
                        if len(other_speakers) == 1:
                            target_speaker = list(other_speakers)[0]
                            logger.info(f"🎯 Vocativo detectado (último turno): '{name}' em '{text[:60]}...' -> outro speaker é {target_speaker}")
                        elif len(other_speakers) > 1:
                            # Múltiplos speakers - pega o que mais falou recentemente antes deste
                            for j in range(i - 1, -1, -1):
                                if transcript[j]['speaker'] != speaker:
                                    target_speaker = transcript[j]['speaker']
                                    logger.info(f"🎯 Vocativo detectado: '{name}' em '{text[:60]}...' -> speaker anterior é {target_speaker}")
                                    break
                    
                    if target_speaker:
                        vocative_mentions[name][target_speaker] += 2  # Peso maior
                else:
                    # Nome mencionado no meio da frase
                    if i + 1 < len(transcript):
                        next_speaker = transcript[i + 1]['speaker']
                        speaker_mentions[name][next_speaker] += 1
        
        # Prioriza auto-apresentações (mais confiável)
        speaker_names.update(self_introductions)
        
        # Prioridade 1: Vocativos (muito confiável)
        for name, responding_speakers in vocative_mentions.items():
            if responding_speakers:
                most_likely_speaker = max(responding_speakers.items(), key=lambda x: x[1])[0]
                if most_likely_speaker not in speaker_names:
                    # Mesmo com 1 menção vocativa já é forte evidência
                    speaker_names[most_likely_speaker] = name
                    logger.info(f"🎯 Vocativo: {most_likely_speaker} = {name} (chamado {responding_speakers[most_likely_speaker]}x)")
        
        # Prioridade 2: Menções contextuais (somente se não tem termos técnicos)
        invalid_terms = {'kotlin', 'java', 'python', 'javascript', 'react', 'angular', 'node'}
        for name, responding_speakers in speaker_mentions.items():
            if responding_speakers and name.lower() not in invalid_terms:
                most_likely_speaker = max(responding_speakers.items(), key=lambda x: x[1])[0]
                if most_likely_speaker not in speaker_names:
                    if responding_speakers[most_likely_speaker] >= 2:
                        speaker_names[most_likely_speaker] = name
                        logger.info(f"🎯 Contextual: {most_likely_speaker} = {name} (mencionado {responding_speakers[most_likely_speaker]}x antes de responder)")
        
        return speaker_names
    
    def save_new_embeddings(self, unknown_speakers_audio, speaker_names, embeddings_dir):
        """Cria e salva embeddings para novos speakers"""
        from speechbrain.inference.speaker import EncoderClassifier
        
        if not unknown_speakers_audio:
            return
        
        logger.info("\n🆕 Criando embeddings para novos speakers...")
        
        # Carrega o modelo de embedding
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        
        saved_count = 0
        
        for speaker_id, audio_chunks in unknown_speakers_audio.items():
            # Verifica se temos um nome detectado
            actual_name = speaker_names.get(speaker_id, None)
            
            if not actual_name:
                logger.info(f"⚠️ {speaker_id}: Nenhum nome detectado, pulando...")
                continue
            
            # Concatena todos os chunks de áudio
            if len(audio_chunks) == 0:
                continue
                
            # Converte para tensor
            combined_audio = np.concatenate(audio_chunks)
            
            # Normaliza
            mx = np.abs(combined_audio).max()
            if mx > 0:
                combined_audio = combined_audio / mx
            
            # Cria embedding
            signal = torch.from_numpy(combined_audio).float().to(self.device)
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            
            with torch.no_grad():
                embedding = classifier.encode_batch(signal).squeeze().cpu().numpy()
            
            # Salva o embedding
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{actual_name}_{timestamp}.npy"
            filepath = os.path.join(embeddings_dir, filename)
            
            np.save(filepath, embedding)
            logger.info(f"💾 Salvo: {filepath}")
            saved_count += 1
        
        if saved_count > 0:
            logger.info(f"\n✅ {saved_count} novo(s) embedding(s) criado(s) com sucesso!")
            logger.info("🔄 Reinicie o sistema para carregar os novos embeddings.")

    def process_audio(self, audio_path, embeddings_dir=None, num_speakers=None):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        # Inicia cronômetro geral
        start_total = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando: {audio_path}")
        if num_speakers:
            logger.info(f"Limitando para MÁXIMO {num_speakers} speaker(s)")
        logger.info(f"{'='*60}")
        
        # 1. Carregamento de áudio
        start = time.time()
        waveform, sample_rate = torchaudio.load(audio_path)
        elapsed = time.time() - start
        logger.info(f"✅ Carregamento de áudio: {elapsed:.2f}s")
        
        # 2. Diarização
        start = time.time()
        # Configura parâmetros de diarização baseado no número de speakers
        diarization_params = {}
        if num_speakers:
            diarization_params['min_speakers'] = num_speakers
            diarization_params['max_speakers'] = num_speakers
        
        diarization = self.pipeline(audio_path, **diarization_params)
        overlap_timeline = diarization.get_overlap()
        elapsed = time.time() - start
        logger.info(f"✅ Diarização (identificação de speakers): {elapsed:.2f}s")
        
        resample_to_8k = T.Resample(sample_rate, 8000)
        resample_to_16k = T.Resample(8000, 16000)
        resample_orig_to_16k = T.Resample(sample_rate, 16000)
        
        full_transcript = []
        speaker_history = {}
        unknown_speakers_audio = defaultdict(list)  # Coleta áudio de Unknowns
        
        # Contadores para estatísticas
        total_segments = 0
        transcribed_segments = 0
        
        # 3. Início de transcrição
        start_transcription = time.time()
        logger.info("\\n📝 Transcrevendo segmentos...")

        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            
            if duration < self.min_segment_duration:
                logger.debug(
                    f"⏭️ [{turn.start:.1f}s] Segmento muito curto ({duration:.2f}s), pulando..."
                )
                continue
            
            # Limita segmentos muito longos para evitar travamento
            effective_end = turn.end
            if duration > 30.0:
                logger.debug(f"⚠️ [{turn.start:.1f}s] Segmento muito longo ({duration:.2f}s), limitando a 30s...")
                effective_end = turn.start + 30.0
            
            total_segments += 1
            logger.debug(f"🔄 [{turn.start:.1f}s - {effective_end:.1f}s] Processando {speaker_id} ({effective_end - turn.start:.2f}s)")
            
            start_frame = int(turn.start * sample_rate)
            end_frame = int(effective_end * sample_rate)
            if start_frame >= waveform.shape[1]:
                logger.debug(f"⏭️ [{turn.start:.1f}s] Frame fora dos limites, pulando...")
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
                logger.info(f"Sobreposição em {turn.start:.1f}s. Separando...")
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
                    logger.info(f"   ⏱️ Separação: {elapsed_sep:.2f}s")
                except Exception as e:
                    logger.error(f"Falha na separação: {e}. Usando áudio original.")
                    sources_to_transcribe = [resample_orig_to_16k(audio_segment).squeeze().numpy()]
            else:
                audio_16k = resample_orig_to_16k(audio_segment)
                sources_to_transcribe = [audio_16k.squeeze().numpy()]

            for idx, source_np in enumerate(sources_to_transcribe):
                source_np = source_np.squeeze()
                
                # Converte estéreo para mono se necessário
                if source_np.ndim == 2:
                    logger.debug(f"   Convertendo estéreo para mono...")
                    source_np = np.mean(source_np, axis=0)
                
                if source_np.ndim != 1 or source_np.size == 0:
                    logger.debug(f"⏭️ [{turn.start:.1f}s] Voz {idx+1}: Áudio inválido (ndim={source_np.ndim}, size={source_np.size}), pulando...")
                    continue
                
                mx = np.abs(source_np).max()
                if np.isnan(mx) or mx == 0:
                    logger.debug(f"⏭️ [{turn.start:.1f}s] Voz {idx+1}: Áudio silencioso (max={mx}), pulando...")
                    continue
                source_np = source_np / mx
                
                real_name, conf = self.verifier._process_audio_chunk(source_np, sample_rate=16000)
                
                # Coleta áudio de Unknown speakers para criar embedding depois
                if real_name == "Unknown":
                    unknown_speakers_audio[speaker_id].append(source_np.copy())
                
                if is_overlap:
                    print(f"    [DEBUG] Voz {idx+1}: {real_name} ({conf:.1%})")

                verified_name = None
                if real_name != "Unknown" and conf >= self.verifier_confidence_min:
                    verified_name = real_name

                disp = speaker_id
                display_label = disp if not verified_name else f"{disp} ({verified_name})"
                
                try:
                    # Normaliza áudio antes de transcrever
                    source_normalized = self.normalize_audio(source_np)
                    
                    # Garante que é numpy array (não tensor)
                    if isinstance(source_normalized, np.ndarray):
                        audio_for_whisper = source_normalized
                    else:
                        audio_for_whisper = source_normalized.cpu().numpy()
                    
                    logger.debug(f"   Audio shape: {audio_for_whisper.shape}, dtype: {audio_for_whisper.dtype}")
                    
                    # Transcreve
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
                    
                    logger.debug(f"   Whisper: {elapsed_whisper:.2f}s, resultado: '{text[:50]}...'")
                    
                    if not text:
                        logger.debug(f"⏭️ [{turn.start:.1f}s] {speaker_id}: Whisper retornou vazio, pulando...")
                        continue
                    
                    if "Amara.org" in text:
                        logger.debug(f"⏭️ [{turn.start:.1f}s] {speaker_id}: Detectado watermark, pulando...")
                        continue
                    
                    clean_new = self.normalize_text(text)
                    last_entry = speaker_history.get(disp)
                    
                    # Verifica duplicata APENAS se foi transcrito muito recentemente (< 1 segundo)
                    # e é praticamente idêntico (> 80% similar)
                    is_duplicate = False
                    if last_entry and clean_new:
                        time_diff = turn.start - last_entry['time']
                        if time_diff < 1.0:  # Menos de 1 segundo de diferença
                            clean_old = last_entry['clean_text']
                            ratio = SequenceMatcher(None, clean_new, clean_old).ratio()
                            if ratio > 0.8:  # 80% de similaridade
                                is_duplicate = True
                    
                    if is_duplicate:
                        logger.debug(f"⏭️ [{turn.start:.1f}s] {disp}: Duplicata (similar>{ratio:.0%}), pulando...")
                        continue
                    
                    transcribed_segments += 1
                    speaker_history[disp] = {'clean_text': clean_new, 'time': turn.start}

                    suffix = f" (Voz {idx+1})" if is_overlap else ""
                    logger.debug(f"✅ [{turn.start:.1f}s] {display_label}{suffix}: TRANSCRITO")
                    print(f" [{turn.start:.1f}s] {display_label}{suffix}: {text}")
                    
                    full_transcript.append({
                        "speaker": disp,
                        "verified_name": verified_name,
                        "text": text,
                        "start": turn.start,
                        "suffix": suffix
                    })
                except Exception as e:
                    logger.error(f"Erro Whisper: {e}")
        
        # Finaliza cronômetro de transcrição
        elapsed_transcription = time.time() - start_transcription
        logger.info(f"\n✅ Transcrição total: {elapsed_transcription:.2f}s")
        
        # Calcula estatísticas
        coverage_rate = (transcribed_segments / total_segments * 100) if total_segments > 0 else 0
        unique_speakers = len(set(line['speaker'] for line in full_transcript))
        logger.info(f"\n📊 ESTATÍSTICAS DE TRANSCRIÇÃO:")
        logger.info(f"  - Segmentos processados: {total_segments}")
        logger.info(f"  - Segmentos transcritos: {transcribed_segments}")
        logger.info(f"  - Taxa de cobertura: {coverage_rate:.1f}%")
        logger.info(f"  - Speakers únicos: {unique_speakers}")
        
        # 4. Detecção de nomes e criação de embeddings
        if unknown_speakers_audio:
            start_names = time.time()
            logger.info(f"\n🔍 Detectados {len(unknown_speakers_audio)} speaker(s) desconhecido(s)")
            logger.info("📝 Analisando transcrição para identificar nomes...")
            
            speaker_names = self.associate_names_with_speakers(full_transcript)
            
            if speaker_names:
                elapsed_names = time.time() - start_names
                logger.info(f"\n✨ Nomes identificados: {speaker_names}")
                logger.info(f"✅ Detecção de nomes: {elapsed_names:.2f}s")
                
                # Determina diretório de embeddings
                embeddings_dir = embeddings_dir or "../data/embeddings"
                if not os.path.exists(embeddings_dir):
                    os.makedirs(embeddings_dir)
                
                start_embeddings = time.time()
                self.save_new_embeddings(unknown_speakers_audio, speaker_names, embeddings_dir)
                elapsed_embeddings = time.time() - start_embeddings
                logger.info(f"✅ Criação de embeddings: {elapsed_embeddings:.2f}s")
            else:
                logger.info("⚠️ Nenhum nome foi identificado na conversa")
        
        # Tempo total
        elapsed_total = time.time() - start_total
        logger.info(f"\n{'='*60}")
        logger.info(f"⏱️ TEMPO TOTAL: {elapsed_total:.2f}s")
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
        print("Pasta de embeddings não encontrada")
        exit()

    print("Inicializando...")
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
            f = input("\nArquivo: ").strip()
            if f in ['sair', 'exit']:
                break
            path = f if os.path.exists(f) else f"../data/testes/{f}"
            
            if os.path.exists(path):
                # Pergunta número de speakers (opcional)
                num_speakers_input = input("Número MÁXIMO de speakers (Enter para detectar automaticamente): ").strip()
                num_speakers = None
                if num_speakers_input.isdigit():
                    num_speakers = int(num_speakers_input)
                    print(f"✅ Limitando para máximo {num_speakers} speaker(s)")
                else:
                    print("✅ Detecção automática de speakers")
                
                transcript_result = system.process_audio(path, embeddings_dir, num_speakers=num_speakers)
                
                base_name = os.path.splitext(os.path.basename(path))[0]
                output_filename = f"{base_name}_transcricao.txt"
                
                if transcript_result:
                    print(f"\nSalvando transcrição em: {output_filename} ...")
                    with open(output_filename, "w", encoding="utf-8") as txt_file:
                        txt_file.write(f"ARQUIVO: {f}\n")
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
                    
                    print(f"Arquivo '{output_filename}' salvo com sucesso!")
                else:
                    print("Nenhuma fala detectada.")
            else:
                print("Arquivo não existe.")
    except Exception as e:
        logger.error(f"Erro Fatal: {e}")
