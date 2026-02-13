import os
import sys
import logging
import numpy as np
import re
from difflib import SequenceMatcher
from collections import defaultdict, deque
from datetime import datetime
import time
import threading
import queue
import tempfile

# Lazy imports para evitar problemas de versão
try:
    import sounddevice as sd
    AUDIO_BACKEND = "sounddevice"
except ImportError:
    try:
        import pyaudio
        AUDIO_BACKEND = "pyaudio"
    except ImportError:
        AUDIO_BACKEND = None

# Imports que causam problemas - lazy
torch = None
torchaudio = None
T = None
Pipeline = None
WhisperModel = None
SepformerSeparation = None
EncoderClassifier = None
MultiSpeakerVerifier = None
login = None
snapshot_download = None
spacy_nlp = None
hf_pipeline_func = None

HF_TOKEN = "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"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("pyannote").setLevel(logging.WARNING)

# Importa spacy e librosa no nível de módulo (são rápidos)
try:
    import spacy
except ImportError:
    spacy = None

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None


# Dicionários de nomes para validação de gênero
NAMES_MALE = {
    'lucas', 'mateus', 'pedro', 'joão', 'gabriel', 'rafael', 'bruno', 'carlos', 'andré',
    'fernando', 'ricardo', 'rodrigo', 'marcelo', 'paulo', 'thiago', 'vitor', 'daniel',
    'augusto', 'leonardo', 'eduardo', 'henrique', 'diego', 'felipe', 'guilherme',
    'kauan', 'marcos', 'jorge', 'cesar', 'antonio', 'luis', 'francisco', 'manuel',
    'xavier', 'mario', 'sergio', 'alberto', 'manoel'
}

NAMES_FEMALE = {
    'maria', 'ana', 'julia', 'beatriz', 'leticia', 'amanda', 'carolina', 'fernanda',
    'mariana', 'juliana', 'patricia', 'camila', 'bruna', 'aline', 'bianca', 'renata',
    'manuela', 'barbara', 'iris', 'ines', 'isis', 'isadora', 'ivana', 'joana',
    'josefa', 'joséfina', 'josiane', 'jovita', 'joyceana', 'judite', 'julieta',
    'jussara', 'justina', 'justine', 'juvita', 'kaila', 'karina', 'karla', 'carla',
    'cassandra', 'cecilia', 'célia', 'celina', 'celia', 'clara', 'clarice', 'claudia',
    'cleide', 'cleonice', 'conceição', 'consuelo', 'constanza', 'corina',
    'cornelia', 'coromila', 'corsina', 'cosma', 'covadonga', 'rosa', 'rosana', 'roselia',
    'vitoria', 'vanessa', 'valeria', 'vanira', 'veronica', 'verena', 'vera', 'vidya',
    'viviana', 'violeta', 'vilma', 'virgilia', 'virginia', 'antonia', 'anatolia'
}

# Mapeamento de nomes masculinos para femininos (correção por gênero)
MALE_TO_FEMALE = {
    'manoel': 'manuela',
    'manuel': 'manuela',
    'gabriel': 'gabriela',
    'ricardo': 'ricarda',
    'luis': 'luisa',
    'francisco': 'francisca',
    'antonio': 'antonia',
    'rio': 'ria',
    'mario': 'maria',
    'carlos': 'carla',
    'paulo': 'paula',
    'julio': 'julia',
    'sergio': 'serbia',
    'pedro': 'petra'
}


def _init_heavy_deps():
    """Inicializa dependências pesadas sob demanda"""
    global torch, torchaudio, T, Pipeline, WhisperModel, SepformerSeparation, EncoderClassifier
    global MultiSpeakerVerifier, login, snapshot_download, spacy_nlp, hf_pipeline_func
    
    if torch is not None:  # Já inicializado
        return True
    
    logger.info("⏳ Carregando dependências pesadas...")
    
    try:
        import torch as torch_module
        import torchaudio as torchaudio_module
        import torchaudio.transforms as T_module
        from pyannote.audio import Pipeline as Pipeline_module
        from faster_whisper import WhisperModel as WhisperModel_module
        from speechbrain.inference.separation import SepformerSeparation as SepformerSeparation_module
        from speechbrain.inference.speaker import EncoderClassifier as EncoderClassifier_module
        from .multi_speaker_verifier import MultiSpeakerVerifier as MSV_module
        from huggingface_hub import login as login_module, snapshot_download as sd_module
        from transformers import pipeline as hf_pipeline_module
        
        # Atribui aos globals
        torch = torch_module
        torchaudio = torchaudio_module
        T = T_module
        Pipeline = Pipeline_module
        WhisperModel = WhisperModel_module
        SepformerSeparation = SepformerSeparation_module
        EncoderClassifier = EncoderClassifier_module
        MultiSpeakerVerifier = MSV_module
        login = login_module
        snapshot_download = sd_module
        hf_pipeline_func = hf_pipeline_module
        
        # Setup torch globals para serialização
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        torch.serialization.add_safe_globals([
            torch.torch_version.TorchVersion,
            Specifications,
            Problem,
            Resolution
        ])
        
        logger.info("✅ Dependências pesadas carregadas")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dependências: {e}")
        raise


class RealtimeTranscriber:
    def __init__(
        self,
        verifier,
        hf_token,
        whisper_size="small",
        device="cuda",
        use_ai_analysis=True,
        diarization_clustering_threshold=0.6,
        verifier_confidence_min=0.8,
        chunk_duration=2.0,
        sample_rate=16000,
    ):
        # Inicializa dependências pesadas
        _init_heavy_deps()
        
        self.verifier = verifier
        self.device = device
        self.use_ai_analysis = use_ai_analysis
        self.diarization_clustering_threshold = diarization_clustering_threshold
        self.verifier_confidence_min = verifier_confidence_min
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        
        # Controle de fluxo
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False
        self.recording_thread = None
        
        # Buffer para processamento
        self.audio_buffer = deque(maxlen=int(sample_rate * chunk_duration * 2))
        self.processing_buffer = []
        
        # Histórico de speakers e transcrições
        self.speaker_history = {}
        self.full_transcript = []
        self.unknown_speakers_audio = defaultdict(list)
        self.speaker_names = {}
        
        logger.info("🎙️ Autenticando...")
        login(token=hf_token)
        
        # Inicializa NER com spaCy
        if use_ai_analysis:
            logger.info("📚 Carregando modelo NER (spaCy)...")
            try:
                self.nlp = spacy.load("pt_core_news_lg")
            except OSError:
                logger.warning("Modelo pt_core_news_lg não encontrado. Tentando pt_core_news_sm...")
                try:
                    self.nlp = spacy.load("pt_core_news_sm")
                except OSError:
                    logger.warning("⚠️ spaCy não disponível. Instale com: python -m spacy download pt_core_news_sm")
                    self.nlp = None
            
            # Carrega LLM
            logger.info("🤖 Carregando LLM para análise contextual...")
            try:
                self.llm = hf_pipeline_func(
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

        logger.info("🎯 Carregando PyAnnote...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(torch.device(device))

        if self.diarization_clustering_threshold is not None:
            try:
                self.pipeline.instantiate({
                    "clustering": {"threshold": self.diarization_clustering_threshold},
                    "segmentation": {"threshold": 0.4}
                })
                logger.info(f"✅ Clustering threshold: {self.diarization_clustering_threshold}")
                logger.info("✅ Segmentation threshold: 0.4")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao ajustar thresholds: {e}")

        # Carrega SepFormer
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        logger.info("🔊 Carregando SepFormer...")
        local_path = "pretrained_models/sepformer"
        if not os.path.exists(os.path.join(local_path, "hyperparams.yaml")):
            logger.warning("SepFormer local não encontrado. Baixando...")
            snapshot_download(
                repo_id="speechbrain/sepformer-whamr",
                local_dir=local_path,
                local_dir_use_symlinks=False
            )

        self.separator = SepformerSeparation.from_hparams(
            source=local_path,
            savedir=local_path,
            run_opts={"device": device}
        )

        logger.info(f"🗣️ Carregando Whisper ({whisper_size})...")
        try:
            self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="float16")
            logger.info("✅ Usando compute_type: float16")
        except Exception as e:
            logger.warning(f"⚠️ float16 não suportado, usando int8...")
            self.whisper = WhisperModel(whisper_size, device="cpu", compute_type="int8")

        logger.info("👤 Carregando classificador de embeddings...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        
        logger.info("✅ Sistema em tempo real carregado com sucesso!")

    def _check_audio_backend(self):
        """Verifica disponibilidade do backend de áudio"""
        if AUDIO_BACKEND is None:
            raise RuntimeError(
                "❌ Nenhum backend de áudio disponível!\n"
                "Instale com: pip install sounddevice -ou- pip install PyAudio"
            )
        logger.info(f"✅ Usando backend de áudio: {AUDIO_BACKEND}")

    def _record_audio(self):
        """Thread para capturar áudio do microfone"""
        logger.info("🎤 Iniciando captura de áudio do microfone...")
        
        if AUDIO_BACKEND == "sounddevice":
            self._record_sounddevice()
        elif AUDIO_BACKEND == "pyaudio":
            self._record_pyaudio()

    def _record_sounddevice(self):
        """Captura áudio usando sounddevice"""
        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms chunks
                dtype=np.float32
            ) as stream:
                logger.info("🔴 GRAVANDO... (Pressione Ctrl+C para parar)")
                print("\n" + "="*60)
                print("🔴 GRAVANDO EM TEMPO REAL")
                print("Fale normalmente (a transcrição aparecerá conforme você fala)")
                print("Pressione Ctrl+C para parar (pode levar alguns segundos)")
                print("="*60 + "\n")
                
                while self.running:
                    try:
                        data, overflowed = stream.read(int(self.sample_rate * 0.1))
                        if overflowed:
                            logger.warning("⚠️ Buffer overflow - algumas amostras perdidas")
                        
                        audio_data = data.flatten().astype(np.float32)
                        try:
                            self.audio_queue.put(audio_data, timeout=0.1)
                        except queue.Full:
                            logger.debug("⏭️ Fila cheia, descartando dados")
                    except KeyboardInterrupt:
                        logger.info("⏹️ Ctrl+C detectado na captura")
                        self.running = False
                        break
        except KeyboardInterrupt:
            logger.info("⏹️ Ctrl+C detectado")
            self.running = False
        except Exception as e:
            logger.error(f"Erro na captura de áudio: {e}")
            self.running = False

    def _record_pyaudio(self):
        """Captura áudio usando PyAudio"""
        try:
            import pyaudio
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=int(self.sample_rate * 0.1)
            )
            
            logger.info("🔴 GRAVANDO... (Pressione Ctrl+C para parar)")
            print("\n" + "="*60)
            print("🔴 GRAVANDO EM TEMPO REAL")
            print("Fale normalmente (a transcrição aparecerá conforme você fala)")
            print("Pressione Ctrl+C para parar")
            print("="*60 + "\n")
            
            while self.running:
                data = stream.read(int(self.sample_rate * 0.1), exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                try:
                    self.audio_queue.put(audio_data, timeout=0.1)
                except queue.Full:
                    logger.debug("⏭️ Fila cheia, descartando dados")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logger.error(f"Erro na captura de áudio: {e}")
            self.running = False

    def _process_audio_chunks(self):
        """Thread para processar chunks de áudio conforme chegam"""
        logger.info("⚙️ Thread de processamento iniciada")
        resample_to_8k = T.Resample(self.sample_rate, 8000)
        resample_to_16k = T.Resample(8000, 16000)
        
        while self.running:
            try:
                # Coleta dados da fila com timeout curto
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.2)
                    self.processing_buffer.extend(audio_chunk)
                except queue.Empty:
                    # Checa a cada 200ms se continua rodando
                    continue
                
                # Processa quando tem ~2 segundos de áudio
                if len(self.processing_buffer) >= int(self.sample_rate * self.chunk_duration):
                    chunk_to_process = np.array(
                        self.processing_buffer[:int(self.sample_rate * self.chunk_duration)],
                        dtype=np.float32
                    )
                    self.processing_buffer = self.processing_buffer[int(self.sample_rate * self.chunk_duration):]
                    
                    # Processa em thread separada para não bloquear captura
                    proc_thread = threading.Thread(
                        target=self._process_chunk,
                        args=(chunk_to_process,),
                        daemon=True
                    )
                    proc_thread.start()
                    
            except KeyboardInterrupt:
                logger.info("Parando processamento...")
                break
            except Exception as e:
                logger.error(f"Erro no processamento: {e}")

    def _process_chunk(self, audio_chunk):
        """Processa um chunk de áudio"""
        if not self.running:
            return
            
        try:
            # Normaliza
            mx = np.abs(audio_chunk).max()
            if mx == 0 or np.isnan(mx):
                return
            audio_chunk = audio_chunk / mx
            
            # Converte para tensor
            waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            
            # Cria arquivo temporário para diarização (funciona em Windows/Linux/Mac)
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            
            try:
                if not self.running:
                    return
                    
                torchaudio.save(temp_path, waveform, self.sample_rate)
                
                # Diarização
                try:
                    diarization = self.pipeline(temp_path)
                except Exception as e:
                    logger.debug(f"Erro na diarização: {e}")
                    return
                
                if not self.running:
                    return
                
                overlap_timeline = diarization.get_overlap()
                
                # Processamento de tracks
                resample_to_8k = T.Resample(self.sample_rate, 8000)
                resample_to_16k = T.Resample(8000, 16000)
                resample_orig_to_16k = T.Resample(self.sample_rate, 16000)
                
                for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                    if not self.running:
                        return
                        
                    duration = turn.end - turn.start
                    if duration < 0.5:
                        continue
                    
                    start_frame = int(turn.start * self.sample_rate)
                    end_frame = min(int(turn.end * self.sample_rate), len(audio_chunk))
                    
                    if start_frame >= len(audio_chunk):
                        continue
                    
                    audio_segment = torch.from_numpy(audio_chunk[start_frame:end_frame]).float()
                    
                    # Verifica overlap
                    overlap_duration = 0.0
                    for ov in overlap_timeline:
                        inter = turn & ov
                        if inter:
                            overlap_duration += inter.duration
                    is_overlap = (overlap_duration / max(duration, 0.001)) > 0.30
                    
                    sources_to_transcribe = []
                    
                    if is_overlap:
                        # Separação de áudio
                        mono = audio_segment.unsqueeze(0)
                        mono_8k = resample_to_8k(mono)
                        
                        try:
                            est_sources = self.separator.separate_batch(mono_8k.unsqueeze(0))
                            src1_8k = est_sources[0, :, 0].detach().cpu()
                            src2_8k = est_sources[0, :, 1].detach().cpu()
                            
                            src1_16k = resample_to_16k(src1_8k).numpy()
                            src2_16k = resample_to_16k(src2_8k).numpy()
                            
                            sources_to_transcribe = [src1_16k, src2_16k]
                        except Exception as e:
                            logger.debug(f"Falha na separação: {e}")
                            sources_to_transcribe = [resample_orig_to_16k(audio_segment.unsqueeze(0)).squeeze().numpy()]
                    else:
                        audio_16k = resample_orig_to_16k(audio_segment.unsqueeze(0))
                        sources_to_transcribe = [audio_16k.squeeze().numpy()]
                    
                    for idx, source_np in enumerate(sources_to_transcribe):
                        if not self.running:
                            return
                            
                        source_np = source_np.squeeze()
                        if source_np.ndim == 2:
                            source_np = np.mean(source_np, axis=0)
                        
                        if source_np.ndim != 1 or source_np.size == 0:
                            continue
                        
                        mx = np.abs(source_np).max()
                        if np.isnan(mx) or mx == 0:
                            continue
                        source_np = source_np / mx
                        
                        # Verifica speaker
                        real_name, conf = self.verifier._process_audio_chunk(source_np, sample_rate=16000)
                        
                        if real_name == "Unknown":
                            self.unknown_speakers_audio[speaker_id].append(source_np.copy())
                        
                        verified_name = None
                        if real_name != "Unknown" and conf >= self.verifier_confidence_min:
                            verified_name = real_name
                        
                        display_label = speaker_id if not verified_name else f"{speaker_id} ({verified_name})"
                        
                        # Transcreve
                        source_normalized = self._normalize_audio(source_np)
                        try:
                            segs, _ = self.whisper.transcribe(
                                source_normalized,
                                language="pt",
                                beam_size=1,
                                vad_filter=False,
                                without_timestamps=True
                            )
                            text = " ".join([s.text for s in segs]).strip()
                        except Exception as e:
                            logger.debug(f"Erro ao transcrever: {e}")
                            continue
                        
                        if not text or "Amara.org" in text:
                            continue
                        
                        # Verifica duplicata
                        clean_new = self._normalize_text(text)
                        last_entry = self.speaker_history.get(speaker_id)
                        is_duplicate = False
                        
                        if last_entry and clean_new:
                            time_diff = turn.start - last_entry['time']
                            if time_diff < 1.0:
                                clean_old = last_entry['clean_text']
                                ratio = SequenceMatcher(None, clean_new, clean_old).ratio()
                                if ratio > 0.8:
                                    is_duplicate = True
                        
                        if is_duplicate:
                            continue
                        
                        self.speaker_history[speaker_id] = {'clean_text': clean_new, 'time': turn.start}
                        
                        suffix = f" (Voz {idx+1})" if is_overlap else ""
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        # Exibe em tempo real
                        print(f"\n[{timestamp}] {display_label}{suffix}: {text}")
                        
                        # Adiciona ao histórico
                        self.full_transcript.append({
                            "speaker": speaker_id,
                            "verified_name": verified_name,
                            "text": text,
                            "timestamp": datetime.now(),
                            "suffix": suffix
                        })
                        
                        # Detecta nomes progressivamente (com análise de áudio)
                        self._update_speaker_names_incremental(speaker_id, text, source_np)
                        
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
        except Exception as e:
            logger.error(f"Erro ao processar chunk: {e}")

    def _normalize_text(self, text):
        """Normaliza texto para comparação"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def _normalize_audio(self, audio, target_db=-20.0):
        """Normaliza áudio"""
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

    def _extract_names_with_ner(self, text):
        """Extrai nomes usando NER"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            names = []
            for ent in doc.ents:
                if ent.label_ == "PER":
                    name = ent.text.strip()
                    name = re.sub(r'^(Sr\.|Sra\.|Dr\.|Dra\.)\s+', '', name, flags=re.IGNORECASE)
                    if len(name) > 2:
                        names.append(name)
            return names
        except Exception as e:
            logger.debug(f"Erro no NER: {e}")
            return []

    def _detect_gender_from_text(self, text):
        """Detecta pistas de gênero no texto (feminino, masculino, outro)"""
        text_lower = text.lower()
        
        # Primeiro: verifica nomes conhecidos extraídos
        extracted_names = self._extract_names_with_ner(text)
        for name in extracted_names:
            name_lower = name.lower().strip()
            # Remove acentos simples
            name_normalized = name_lower.replace('é', 'e').replace('á', 'a').replace('à', 'a')
            
            if name_lower in NAMES_FEMALE or name_normalized in NAMES_FEMALE:
                logger.debug(f"✅ Nome feminino detectado: {name}")
                return "female"
            elif name_lower in NAMES_MALE or name_normalized in NAMES_MALE:
                logger.debug(f"✅ Nome masculino detectado: {name}")
                return "male"
        
        # Marcadores femininos
        female_markers = [
            r'\beu sou a\b',
            r'\bsou a\b',
            r'\bsou mulher',
            r'\beu sou mulher',
            r'\bmeu nome é\s+([a-zà-ú]+a)\b',  # Nomes terminados em A
            r'\bme chamo\s+([a-zà-ú]+a)\b',
            r'\baqui é\s+([a-zà-ú]+a)\b',
            r'\baqui quem fala é\s+([a-zà-ú]+a)\b',
        ]
        
        # Marcadores masculinos
        male_markers = [
            r'\beu sou o\b',
            r'\bsou o\b',
            r'\bsou homem',
            r'\beu sou homem',
            r'\bmeu nome é\s+([a-zà-ú]+o)\b',  # Nomes terminados em O
            r'\bme chamo\s+([a-zà-ú]+o)\b',
            r'\baqui é\s+([a-zà-ú]+o)\b',
            r'\baqui quem fala é\s+([a-zà-ú]+o)\b',
        ]
        
        female_score = sum(1 for pattern in female_markers if re.search(pattern, text_lower))
        male_score = sum(1 for pattern in male_markers if re.search(pattern, text_lower))
        
        if female_score > male_score:
            return "female"
        elif male_score > female_score:
            return "male"
        return None
    
    def _detect_gender_from_audio(self, audio_chunk):
        """Detecta gênero pela análise de pitch (frequência fundamental)"""
        if librosa is None:
            logger.debug("⚠️ librosa não disponível, pulando análise de pitch")
            return None
        
        try:
            # Detecta pitch usando librosa
            f0 = librosa.yin(audio_chunk, fmin=50, fmax=500, sr=self.sample_rate)
            
            # Filtra valores válidos (pitch > 0)
            valid_f0 = f0[f0 > 0]
            
            if len(valid_f0) < 10:
                logger.debug("⚠️ Pouco pitch detectado no áudio")
                return None
            
            # Calcula pitch médio
            mean_f0 = np.median(valid_f0)
            
            logger.debug(f"🎵 Pitch médio detectado: {mean_f0:.1f} Hz")
            
            # Classifica por gênero baseado em pitch
            # Feminino: 130-280 Hz (vozes femininas podem ser mais graves)
            # Masculino: 50-130 Hz
            if mean_f0 > 145:
                logger.debug(f"🎵 Áudio classificado como feminino (pitch: {mean_f0:.1f} Hz)")
                return "female"
            elif mean_f0 < 125:
                logger.debug(f"🎵 Áudio classificado como masculino (pitch: {mean_f0:.1f} Hz)")
                return "male"
            else:
                logger.debug(f"⚠️ Pitch ambíguo (pitch: {mean_f0:.1f} Hz) - indeterminado")
                return None
                
        except Exception as e:
            logger.debug(f"⚠️ Erro ao analisar pitch: {e}")
            return None
    
    def _combine_gender_detection(self, text_gender, audio_gender):
        """Combina detecção de gênero por texto e áudio"""
        if audio_gender and text_gender:
            # Se ambas concordam, retorna
            if audio_gender == text_gender:
                logger.debug(f"✅ Gênero confirmado por texto E áudio: {audio_gender}")
                return audio_gender
            else:
                # Conflito: texto tem prioridade (nome é mais confiável que pitch)
                logger.info(f"⚠️ Conflito de gênero - Texto: {text_gender}, Áudio: {audio_gender} → Usando texto (nome): {text_gender}")
                return text_gender
        elif text_gender:
            # Prefere texto/nome (mais confiável)
            return text_gender
        elif audio_gender:
            return audio_gender
        return None
    
    def _correct_name_for_gender(self, name, gender):
        """Corrige nomes baseado no gênero detectado"""
        if not gender:
            return name
        
        name_lower = name.lower()
        
        if gender == "female":
            # Se é masculino mas detectamos feminino, corrige
            if name_lower in MALE_TO_FEMALE:
                corrected = MALE_TO_FEMALE[name_lower]
                logger.info(f"🔧 Corrigindo nome por gênero: {name} → {corrected.capitalize()} (feminino)")
                return corrected.capitalize()
            # Se termina em O masculino, tenta adicionar A
            elif name_lower.endswith('o') and name_lower not in NAMES_FEMALE:
                corrected = name_lower[:-1] + 'a'
                logger.info(f"🔧 Ajustando nome para feminino: {name} → {corrected.capitalize()}")
                return corrected.capitalize()
        
        elif gender == "male":
            # Se detectamos masculino, valida se é realmente masculino
            if name_lower in NAMES_FEMALE and name_lower not in NAMES_MALE:
                logger.warning(f"⚠️ Nome {name} é feminino mas detectado voz masculina - mantendo {name}")
        
        return name
    
    def _update_speaker_names_incremental(self, speaker_id, text, audio_chunk=None):
        """Identifica nomes conforme a conversa avança com validação de gênero (texto + áudio)"""
        if speaker_id in self.speaker_names:
            return  # Já tem nome
        
        # Detecta gênero da voz a partir do texto
        text_gender = self._detect_gender_from_text(text)
        
        # Detecta gênero a partir do áudio (pitch)
        audio_gender = None
        if audio_chunk is not None:
            audio_gender = self._detect_gender_from_audio(audio_chunk)
        
        # Combina ambas as detecções
        detected_gender = self._combine_gender_detection(text_gender, audio_gender)
        
        # Padrões de auto-apresentação
        self_intro_patterns = [
            r'(?:meu nome é|me chamo|eu sou|sou o|sou a)\s+([A-ZÀ-Ú][a-zà-ú]+)',
            r'(?:aqui é|aqui quem fala é)\s+(?:o|a)?\s*([A-ZÀ-Ú][a-zà-ú]+)',
        ]
        
        for pattern in self_intro_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).capitalize()
                
                # Corrige nome conforme gênero detectado
                corrected_name = self._correct_name_for_gender(name, detected_gender)
                
                self.speaker_names[speaker_id] = corrected_name
                logger.info(f"✨ AUTO-APRESENTAÇÃO: {speaker_id} = {corrected_name} (gênero: {detected_gender or 'não detectado'})")
                return
        
        # Detecta com NER
        detected_names = self._extract_names_with_ner(text)
        if detected_names:
            logger.debug(f"📝 NER detectou nomes: {detected_names}")

    def save_session(self, output_dir="realtime_sessions"):
        """Salva a sessão de transcrição em tempo real"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva transcrição
        transcript_file = os.path.join(output_dir, f"transcript_{timestamp}.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(f"TRANSCRIÇÃO EM TEMPO REAL - {timestamp}\n")
            f.write("="*60 + "\n\n")
            
            for entry in self.full_transcript:
                timestamp_str = entry['timestamp'].strftime("%H:%M:%S")
                speaker = entry['speaker']
                verified_name = entry.get('verified_name')
                suffix = entry.get('suffix', '')
                text = entry['text']
                
                if verified_name:
                    f.write(f"[{timestamp_str}] {speaker} ({verified_name}){suffix}: {text}\n")
                else:
                    f.write(f"[{timestamp_str}] {speaker}{suffix}: {text}\n")
        
        logger.info(f"💾 Transcrição salva em: {transcript_file}")
        
        # Salva nomes detectados
        if self.speaker_names:
            names_file = os.path.join(output_dir, f"speakers_{timestamp}.txt")
            with open(names_file, "w", encoding="utf-8") as f:
                f.write("SPEAKERS IDENTIFICADOS\n")
                f.write("="*60 + "\n\n")
                for speaker_id, name in self.speaker_names.items():
                    f.write(f"{speaker_id}: {name}\n")
            
            logger.info(f"👥 Speakers salvos em: {names_file}")
        
        return transcript_file

    def start_recording(self, embeddings_dir=None):
        """Inicia transcrição em tempo real"""
        self._check_audio_backend()
        
        if self.running:
            logger.warning("⚠️ Já está gravando!")
            return
        
        self.running = True
        self.full_transcript = []
        self.speaker_history = {}
        self.speaker_names = {}
        
        # Inicia threads
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.processing_thread = threading.Thread(target=self._process_audio_chunks, daemon=True)
        
        self.recording_thread.start()
        self.processing_thread.start()
        
        try:
            # Aguarda com timeout para permitir Ctrl+C mais rápido
            while self.running:
                self.recording_thread.join(timeout=0.5)
                if not self.recording_thread.is_alive():
                    break
        except KeyboardInterrupt:
            logger.info("\n⏹️ Parando gravação...")
            self.running = False
            # Aguarda threads com timeout
            self.recording_thread.join(timeout=1)
            self.processing_thread.join(timeout=1)
        
        logger.info("✅ Gravação finalizada!")
        
        # Salva sessão
        output_file = self.save_session()
        
        # Cria embeddings para novos speakers
        if self.unknown_speakers_audio and embeddings_dir:
            self._save_new_embeddings(embeddings_dir)
        
        print(f"\n{'='*60}")
        print(f"📊 SESSÃO FINALIZADA")
        print(f"  - Transcrições: {len(self.full_transcript)}")
        print(f"  - Speakers únicos: {len(set(e['speaker'] for e in self.full_transcript)) if self.full_transcript else 0}")
        print(f"  - Nomes detectados: {len(self.speaker_names)}")
        print(f"  - Arquivo: {output_file}")
        print(f"{'='*60}\n")
        
        return output_file

    def _save_new_embeddings(self, embeddings_dir):
        """Cria embeddings para novos speakers detectados"""
        if not self.unknown_speakers_audio:
            return
        
        logger.info("\n🆕 Criando embeddings para novos speakers em tempo real...")
        
        saved_count = 0
        max_seconds = 20.0
        max_samples = int(self.sample_rate * max_seconds)
        for speaker_id, audio_chunks in self.unknown_speakers_audio.items():
            actual_name = self.speaker_names.get(speaker_id, None)
            
            if not actual_name:
                logger.info(f"⚠️ {speaker_id}: Nenhum nome detectado")
                continue
            
            if len(audio_chunks) == 0:
                continue
            
            # Concatena apenas um limite de audio para evitar OOM
            selected_chunks = []
            remaining = max_samples
            for chunk in audio_chunks:
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    selected_chunks.append(chunk[:remaining])
                    remaining = 0
                else:
                    selected_chunks.append(chunk)
                    remaining -= len(chunk)

            if not selected_chunks:
                continue

            combined_audio = np.concatenate(selected_chunks)
            mx = np.abs(combined_audio).max()
            if mx > 0:
                combined_audio = combined_audio / mx
            
            # Cria embedding
            signal = torch.from_numpy(combined_audio).float().to(self.device)
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            
            try:
                with torch.no_grad():
                    embedding = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
            except RuntimeError as e:
                logger.error(f"⚠️ Erro ao gerar embedding (memoria): {e}")
                continue
            
            # Salva
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{actual_name}_{timestamp}.npy"
            filepath = os.path.join(embeddings_dir, filename)
            
            np.save(filepath, embedding)
            logger.info(f"💾 Embedding salvo: {filename}")
            saved_count += 1
        
        if saved_count > 0:
            logger.info(f"\n✅ {saved_count} novo(s) embedding(s) criado(s)!")


if __name__ == "__main__":
    import sys
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_dirs = [
        os.path.join(base_dir, "data", "embeddings"),
        os.path.join(base_dir, "data", "embeddings", "individual"),
    ]
    embeddings_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
    
    if not embeddings_dir:
        print("❌ Pasta de embeddings não encontrada")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎙️ TRANSCRITOR EM TEMPO REAL - MODO COMPLETO")
    print("="*60)
    print("\nInicializando sistema...")
    print("(Este pode levar 1-2 minutos na primeira execução)")
    print()
    
    try:
        verifier = MultiSpeakerVerifier(embeddings_dir, threshold=0.65)
        system = RealtimeTranscriber(
            verifier,
            HF_TOKEN,
            diarization_clustering_threshold=0.45,
            verifier_confidence_min=0.9,
            chunk_duration=2.0,
        )
        
        print("✅ Sistema pronto!")
        input("\nPressione ENTER para iniciar a gravação (Ctrl+C para parar)...\n")
        
        system.start_recording(embeddings_dir)
        
    except Exception as e:
        logger.error(f"❌ Erro Fatal: {e}", exc_info=True)
        sys.exit(1)
