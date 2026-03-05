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

try:
    import sounddevice as sd
    AUDIO_BACKEND = "sounddevice"
except ImportError:
    try:
        import pyaudio
        AUDIO_BACKEND = "pyaudio"
    except ImportError:
        AUDIO_BACKEND = None

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

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # só os nossos logs em DEBUG
logging.getLogger("src.personid_tracker").setLevel(logging.DEBUG)

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

NAMES_MALE = {
    'lucas', 'mateus', 'pedro', 'joão', 'gabriel', 'rafael', 'bruno', 'carlos', 'andré',
    'fernando', 'ricardo', 'rodrigo', 'marcelo', 'paulo', 'thiago', 'vitor', 'daniel',
    'augusto', 'leonardo', 'eduardo', 'henrique', 'diego', 'felipe', 'guilherme',
    'kauan', 'marcos', 'jorge', 'cesar', 'antonio', 'luis', 'francisco', 'manuel',
    'xavier', 'mario', 'sergio', 'alberto', 'manoel', 'josé', 'jose', 'joaquim',
    'arthur', 'miguel', 'enzo', 'bernardo', 'heitor', 'davi', 'murilo', 'caio'
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

MALE_TO_FEMALE = {
    'manoel': 'manuela', 'manuel': 'manuela', 'gabriel': 'gabriela', 'ricardo': 'ricarda',
    'luis': 'luisa', 'francisco': 'francisca', 'antonio': 'antonia', 'rio': 'ria',
    'mario': 'maria', 'carlos': 'carla', 'paulo': 'paula', 'julio': 'julia',
    'sergio': 'serbia', 'pedro': 'petra'
}

def _init_heavy_deps():
    global torch, torchaudio, T, Pipeline, WhisperModel, SepformerSeparation, EncoderClassifier
    global MultiSpeakerVerifier, login, snapshot_download, spacy_nlp, hf_pipeline_func
    if torch is not None:
        return True
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
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        torch.serialization.add_safe_globals([
            torch.torch_version.TorchVersion, Specifications, Problem, Resolution
        ])
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dependências: {e}")
        raise

class RealtimeTranscriber:
    def __init__(self, verifier, hf_token, whisper_size="medium", device="cuda", use_ai_analysis=True, diarization_clustering_threshold=0.6, verifier_confidence_min=0.8, chunk_duration=2.0, sample_rate=16000, shared_state=None, num_speakers=None, audio_device=None):
        _init_heavy_deps()
        self.verifier = verifier
        self.device = device
        self.use_ai_analysis = use_ai_analysis
        self.diarization_clustering_threshold = diarization_clustering_threshold
        self.verifier_confidence_min = verifier_confidence_min
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.shared_state = shared_state or {"active_faces": {}}
        self.num_speakers = num_speakers
        self.audio_device = audio_device  # None = padrão do sistema
        self.audio_queue = queue.Queue(maxsize=10000)
        self.running = False
        self.recording_thread = None
        self.audio_buffer = deque(maxlen=int(sample_rate * chunk_duration * 2))
        self.processing_buffer = []
        self.speaker_history = {}
        self.full_transcript = []
        self.all_audio_chunks = []
        self.unknown_speakers_audio = defaultdict(list)
        # person_id → display_name (shared com o lado de vídeo via shared_state)
        self.speaker_names = self.shared_state.setdefault("person_names", {})
        # embedding_key (nome no verifier) → person_id estável
        self._emb_to_pid = self.shared_state.setdefault("embedding_to_person", {})
        self.identified_speakers = set()   # conjunto de person_ids com nome real confirmado
        self._session_unknown_embs = {}    # person_id -> np.array embedding
        self._session_unknown_counter = 0
        if num_speakers:
            logger.info(f"Limite de speakers: {num_speakers}")
        login(token=hf_token)
        if use_ai_analysis:
            try:
                self.nlp = spacy.load("pt_core_news_lg")
            except OSError:
                try:
                    self.nlp = spacy.load("pt_core_news_sm")
                except OSError:
                    self.nlp = None
            try:
                self.llm = hf_pipeline_func("text2text-generation", model="google/flan-t5-small", device=0 if device == "cuda" else -1, max_length=512)
            except Exception:
                self.llm = None
        else:
            self.nlp = None
            self.llm = None
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device(device))
        if self.diarization_clustering_threshold is not None:
            try:
                self.pipeline.instantiate({"clustering": {"threshold": self.diarization_clustering_threshold}, "segmentation": {"threshold": 0.4}})
            except Exception:
                pass
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        local_path = "pretrained_models/sepformer"
        if not os.path.exists(os.path.join(local_path, "hyperparams.yaml")):
            snapshot_download(repo_id="speechbrain/sepformer-whamr", local_dir=local_path, local_dir_use_symlinks=False)
        self.separator = SepformerSeparation.from_hparams(source=local_path, savedir=local_path, run_opts={"device": device})
        try:
            self.whisper = WhisperModel(whisper_size, device=self.device, compute_type="float16")
        except Exception:
            self.whisper = WhisperModel(whisper_size, device=self.device, compute_type="int8")
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

    def _check_audio_backend(self):
        if AUDIO_BACKEND is None:
            raise RuntimeError("❌ Nenhum backend de áudio disponível!")

    def _record_audio(self):
        if AUDIO_BACKEND == "sounddevice":
            self._record_sounddevice()
        elif AUDIO_BACKEND == "pyaudio":
            self._record_pyaudio()

    def _record_sounddevice(self):
        try:
            with sd.InputStream(device=self.audio_device, channels=1, samplerate=self.sample_rate, blocksize=int(self.sample_rate * 0.1), dtype=np.float32) as stream:
                while self.running:
                    data, overflowed = stream.read(int(self.sample_rate * 0.1))
                    audio_data = data.flatten().astype(np.float32)
                    self.all_audio_chunks.append(audio_data.copy())
                    self.audio_queue.put(audio_data)
        except Exception:
            self.running = False

    def _record_pyaudio(self):
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=int(self.sample_rate * 0.1))
            while self.running:
                data = stream.read(int(self.sample_rate * 0.1), exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.all_audio_chunks.append(audio_data.copy())
                self.audio_queue.put(audio_data)
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            self.running = False

    def _process_audio_chunks(self):
        # Janela deslizante com overlap grande para labels de diarização consistentes
        # entre chunks consecutivos.
        #   janela  : 15 s  — contexto suficiente para o pyannote estabilizar os labels
        #   passo   :  5 s  — avança 5 s por iteração (overlap de 10 s)
        #   zona nova: últimos 5 s — única parte transcrita; os primeiros 10 s são contexto
        window_samples   = int(self.sample_rate * 15.0)
        step_samples     = int(self.sample_rate * 5.0)
        new_zone_samples = window_samples - step_samples   # 10 s * sample_rate

        # Primeiro chunk: janela de 8 s para identificação inicial mais rápida.
        # Consome o chunk inteiro (sem overlap prévio). O segundo chunk (15 s) também
        # processa tudo (zone_start=0) pois não há overlap do primeiro. A partir do
        # terceiro chunk, a janela deslizante normal com 10 s de contexto entra em vigor.
        FIRST_WINDOW_S = 8.0
        first_window_samples = int(self.sample_rate * FIRST_WINDOW_S)
        _chunks_processed = 0

        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.2)
                self.processing_buffer.extend(audio_chunk)
                required = first_window_samples if _chunks_processed == 0 else window_samples
                while len(self.processing_buffer) >= required:
                    if _chunks_processed == 0:
                        # Primeiro chunk: 8 s, consome tudo
                        chunk_to_process = np.array(self.processing_buffer[:first_window_samples], dtype=np.float32)
                        self.processing_buffer = self.processing_buffer[first_window_samples:]
                        chunk_wall_start = time.time() - first_window_samples / self.sample_rate
                        zone_start = 0
                    else:
                        chunk_to_process = np.array(self.processing_buffer[:window_samples], dtype=np.float32)
                        self.processing_buffer = self.processing_buffer[step_samples:]
                        chunk_wall_start = time.time() - window_samples / self.sample_rate
                        # Segundo chunk: sem overlap do primeiro → processa tudo
                        zone_start = 0 if _chunks_processed == 1 else new_zone_samples
                    self._process_chunk(chunk_to_process,
                                        chunk_wall_start=chunk_wall_start,
                                        new_zone_start_samples=zone_start)
                    _chunks_processed += 1
                    required = window_samples
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erro no processamento: {e}")

    def _process_chunk(self, audio_chunk, chunk_wall_start: float | None = None, new_zone_start_samples: int = 0):
        if not self.running:
            return
        try:
            mx = np.abs(audio_chunk).max()
            if mx < 0.01 or np.isnan(mx):
                return
            audio_np = np.array(audio_chunk, dtype=np.float32)
            audio_np = self._normalize_audio(audio_np)
            if self.sample_rate != 16000:
                resampler = T.Resample(orig_freq=int(self.sample_rate), new_freq=16000)
                audio_np = resampler(torch.from_numpy(audio_np).float()).numpy()

            if chunk_wall_start is None:
                chunk_wall_start = time.time() - len(audio_np) / 16000

            # Diarização para separar turnos de fala por speaker
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            diarization_params = {}
            if self.num_speakers:
                diarization_params['max_speakers'] = self.num_speakers
            try:
                diarization = self.pipeline(
                    {"waveform": audio_tensor, "sample_rate": 16000},
                    **diarization_params
                )
                turns = [
                    (turn, spk)
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                    if (turn.end - turn.start) >= 0.5
                ]
            except Exception as e:
                logger.warning(f"Diarização falhou ({e}). Processando como speaker único.")
                turns = []

            if not turns:
                # Sem diarização — processa só a zona nova para evitar retranscrever o contexto
                new_zone = audio_np[new_zone_start_samples:]
                if len(new_zone) >= int(16000 * 0.5):
                    self._process_segment(
                        new_zone,
                        wall_time_start=chunk_wall_start + new_zone_start_samples / 16000,
                        wall_time_end=chunk_wall_start + len(audio_np) / 16000,
                    )
                return

            distinct_speakers = {spk for _, spk in turns}

            # --- SepFormer: quando há >= 2 speakers detectados pela diarização ---
            # Usa a separação de fontes para obter streams mais limpos por speaker.
            if len(distinct_speakers) >= 2:
                separated = self._separate_with_sepformer(audio_np)
                if separated is not None:
                    # Calcula session_id por speaker da diarização para usar como hints
                    spk_segments = defaultdict(list)
                    for turn, spk in turns:
                        s = int(turn.start * 16000)
                        e = min(int(turn.end * 16000), len(audio_np))
                        seg = audio_np[s:e]
                        if len(seg) >= int(16000 * 0.5):
                            spk_segments[spk].append(seg)
                    spk_to_session = {
                        spk: self._get_or_create_session_speaker_id(np.concatenate(segs))
                        for spk, segs in spk_segments.items()
                    }
                    # Associa cada stream separado ao session speaker com maior cosine sim
                    stream_hints = self._match_streams_to_speakers(separated, spk_to_session)
                    for i, stream in enumerate(separated):
                        if np.abs(stream).max() < 0.005:
                            continue  # stream quase silencioso (artefato do SepFormer)
                        # Só transcreve a zona nova; os primeiros new_zone_start_samples
                        # serviram apenas para o SepFormer ter contexto de separação.
                        stream_new = stream[new_zone_start_samples:]
                        if len(stream_new) < int(16000 * 0.5) or np.abs(stream_new).max() < 0.005:
                            continue
                        stream_norm = self._normalize_audio(stream_new)
                        wall_new_start = chunk_wall_start + new_zone_start_samples / 16000
                        self._process_segment(
                            stream_norm,
                            session_hint=stream_hints.get(i),
                            wall_time_start=wall_new_start,
                            wall_time_end=chunk_wall_start + len(stream) / 16000,
                        )
                    return  # SepFormer processou — não cai no processamento padrão

            # --- Processamento padrão por diarização (1 speaker ou SepFormer falhou) ---
            spk_segments = defaultdict(list)
            for turn, spk in turns:
                start_s = int(turn.start * 16000)
                end_s = min(int(turn.end * 16000), len(audio_np))
                seg = audio_np[start_s:end_s]
                if len(seg) >= int(16000 * 0.5):
                    spk_segments[spk].append(seg)

            spk_to_session = {
                spk: self._get_or_create_session_speaker_id(np.concatenate(segs))
                for spk, segs in spk_segments.items()
            }

            for turn, spk in turns:
                start_s = int(turn.start * 16000)
                end_s   = min(int(turn.end * 16000), len(audio_np))
                # Só emite segmentos dentro da zona nova (contexto não gera transcrição)
                seg_start = max(start_s, new_zone_start_samples)
                if seg_start >= end_s:
                    continue
                segment = audio_np[seg_start:end_s]
                if len(segment) >= int(16000 * 0.5):
                    self._process_segment(
                        segment,
                        session_hint=spk_to_session.get(spk),
                        wall_time_start=chunk_wall_start + seg_start / 16000,
                        wall_time_end=chunk_wall_start + end_s / 16000,
                    )
        except Exception as e:
            print(f"Erro ao processar chunk: {e}")

    def _separate_with_sepformer(self, audio_np: np.ndarray) -> list[np.ndarray] | None:
        """Separa o áudio misto em 2 streams usando SepFormer (requer 8 kHz internamente)."""
        try:
            # 16 kHz → 8 kHz (SepFormer foi treinado com 8 kHz)
            down = T.Resample(16000, 8000)
            audio_8k = down(torch.from_numpy(audio_np).float()).unsqueeze(0).to(self.device)

            with torch.no_grad():
                est = self.separator.separate_batch(audio_8k)
            # est: [1, T_8k, n_sources]

            # 8 kHz → 16 kHz
            up = T.Resample(8000, 16000)
            streams = []
            n_sources = est.shape[-1]
            for i in range(n_sources):
                src = est[0, :, i].cpu()
                src_16k = up(src.unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
                streams.append(src_16k)
            return streams
        except Exception as e:
            logger.warning(f"SepFormer falhou: {e}")
            return None

    def _match_streams_to_speakers(self, streams: list[np.ndarray],
                                   spk_to_session: dict) -> dict[int, str]:
        """Associa índice de stream SepFormer → session speaker_id por similaridade de embedding."""
        if not spk_to_session:
            return {}
        # Embeddings dos streams separados
        stream_embs = []
        for stream in streams:
            sig = torch.from_numpy(stream).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                emb = self.classifier.encode_batch(sig).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            stream_embs.append(emb / norm if norm > 0 else emb)

        # Embeddings dos speakers já rastreados (de _session_unknown_embs)
        spk_embs = {
            spk: self._session_unknown_embs[sid]
            for spk, sid in spk_to_session.items()
            if sid in self._session_unknown_embs
        }
        if not spk_embs:
            # Sem referência → atribui 1:1 por ordem
            return {i: sid for i, (_, sid) in enumerate(spk_to_session.items()) if i < len(streams)}

        hint_map: dict[int, str] = {}
        used_spks: set[str] = set()
        for i, s_emb in enumerate(stream_embs):
            best_spk, best_sim = None, -1.0
            for spk, ref_emb in spk_embs.items():
                if spk in used_spks:
                    continue
                sim = float(np.dot(s_emb, ref_emb))
                if sim > best_sim:
                    best_sim, best_spk = sim, spk
            if best_spk is not None:
                hint_map[i] = spk_to_session[best_spk]
                used_spks.add(best_spk)
        return hint_map

    def _process_segment(self, audio_np, session_hint=None,
                         wall_time_start: float | None = None,
                         wall_time_end:   float | None = None):
        """Processa um segmento de fala de um único speaker."""
        try:
            bad_phrases = ["Amara.org", "Legendas", "Obrigado", "tchau gente", "tchau tchau",
                           "transmissão", "inscreva-se", "obrigada por assistir",
                           "continue assistindo", "não se esqueça"]
            try:
                segs_gen, _ = self.whisper.transcribe(
                    audio_np,
                    language="pt",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    initial_prompt="Olá, tudo bem? Sim, estou ouvindo. Ok, pode continuar.",
                )
                segs_list = list(segs_gen)
            except Exception:
                return
            if not segs_list:
                return
            # Descarta segmentos onde Whisper não tem confiança de que há fala real
            avg_no_speech = sum(getattr(s, 'no_speech_prob', 0.0) for s in segs_list) / len(segs_list)
            if avg_no_speech > 0.5:
                return
            # avg_logprob muito baixo indica hallucination (Whisper inventando texto)
            avg_logprob = sum(getattr(s, 'avg_logprob', 0.0) for s in segs_list) / len(segs_list)
            text = " ".join([s.text for s in segs_list]).strip()
            if not text or len(text) < 5 or any(bp.lower() in text.lower() for bp in bad_phrases):
                return
            # Frases curtas com logprob baixo ou no_speech moderado → provável hallucination
            if len(text) < 20 and (avg_logprob < -0.8 or avg_no_speech > 0.3):
                return

            # Identifica o speaker: verifier → visual → tracking por voz
            raw_best_name, raw_conf, all_candidates = self.verifier._process_audio_chunk(audio_np, sample_rate=16000)

            # --- Seleção gender-aware + margin entre candidatos do verifier ---
            audio_gender = self._detect_gender_from_audio(audio_np) if librosa is not None else None

            # LOG: candidatos do verifier e gênero detectado
            cand_str = " | ".join(f"{n}: {s:.2f}" for n, s in all_candidates[:4]) if all_candidates else "nenhum"
            logger.debug(f"🔍 Verifier: [{cand_str}] gender={audio_gender} threshold={self.verifier.threshold}")

            def _gender_of(name):
                base = name.lower().split()[0]
                if base in NAMES_FEMALE: return "female"
                if base in NAMES_MALE:   return "male"
                return None

            # Filtra candidatos: remove os que contradizem o gênero do áudio
            if audio_gender and all_candidates:
                filtered = []
                for cand_name, cand_score in all_candidates:
                    cand_g = _gender_of(cand_name)
                    if cand_g and cand_g != audio_gender:
                        continue  # gênero contradiz → pula
                    filtered.append((cand_name, cand_score))
                if not filtered:
                    filtered = all_candidates  # fallback: se filtrou tudo, usa original
            else:
                filtered = all_candidates

            # Margin check nos candidatos filtrados: melhor vs segundo melhor
            real_name, conf = "Unknown", 0.0
            if filtered:
                best_name, best_score = filtered[0]
                if len(filtered) >= 2:
                    _, second_score = filtered[1]
                    margin = best_score - second_score
                else:
                    margin = 1.0  # candidato único → margem máxima
                if margin >= 0.04:
                    real_name, conf = best_name, best_score
                elif audio_gender:
                    # Margem baixa mas gênero foi determinado — confia no filtro de gênero
                    real_name, conf = best_name, best_score

            # LOG: resultado após filtragem
            filt_str = " | ".join(f"{n}: {s:.2f}" for n, s in filtered[:4]) if filtered else "nenhum"
            logger.debug(f"🔍 Filtrado: [{filt_str}] → escolhido={real_name} conf={conf:.2f}")

            active_faces = self.shared_state.get("active_faces", {})
            visual_people = list(active_faces.values())

            # --- ASD: consulta qual face estava se mexendo no segmento ---
            asd_person_id = None
            asd = self.shared_state.get("asd")
            if asd and wall_time_start is not None and active_faces:
                t_end = wall_time_end if wall_time_end is not None else wall_time_start + len(audio_np) / 16000
                asd_track = asd.get_active_speaker(wall_time_start, t_end)
                if asd_track is not None and asd_track in active_faces:
                    asd_person_id = active_faces[asd_track]

            # LOG: estado visual
            faces_str = ", ".join(f"t{tid}={self.speaker_names.get(pid, pid)}" for tid, pid in active_faces.items())
            asd_name = self.speaker_names.get(asd_person_id, asd_person_id) if asd_person_id else "None"
            logger.debug(f"🔍 Faces: [{faces_str}] ASD={asd_name}")

            if real_name != "Unknown" and conf >= self.verifier_confidence_min:
                # 1) Verifier com alta confiança — fonte mais confiável
                if real_name not in self._emb_to_pid:
                    pid = self._new_person_id()
                    self._emb_to_pid[real_name] = pid
                    self.speaker_names[pid] = real_name
                speaker_id = self._emb_to_pid[real_name]
                verified_name = real_name
                decision = f"VERIFIER_HIGH ({conf:.2f})"
            elif real_name != "Unknown" and conf >= self.verifier.threshold:
                # 2) Verifier com confiança moderada (gender já filtrado) — biometria
                #    de voz é mais confiável que ASD (lip pixel diff) especialmente
                #    quando o speaker está fora de câmera.
                if real_name not in self._emb_to_pid:
                    pid = self._new_person_id()
                    self._emb_to_pid[real_name] = pid
                    self.speaker_names[pid] = real_name
                speaker_id = self._emb_to_pid[real_name]
                verified_name = real_name
                decision = f"VERIFIER_MOD ({conf:.2f})"
            elif asd_person_id is not None:
                # 3) ASD detectou quem estava com a boca se mexendo — vínculo face-voz
                speaker_id = asd_person_id
                verified_name = self.speaker_names.get(asd_person_id)
                decision = f"ASD ({asd_name})"
            elif len(visual_people) == 1 and raw_conf >= 0.35:
                # 4) Heurística: única face + áudio soa como voz humana
                speaker_id = visual_people[0]
                verified_name = self.speaker_names.get(speaker_id)
                decision = f"SINGLE_FACE ({raw_conf:.2f})"
            elif session_hint is not None:
                # Usa o person_id pré-computado pela diarização (evita misturar vozes)
                speaker_id = session_hint
                verified_name = None
                decision = "SESSION_HINT"
            else:
                speaker_id = self._get_or_create_session_speaker_id(audio_np)
                verified_name = None
                decision = f"SESSION_TRACKER ({speaker_id})"

            # --- Gender cross-check: rejeita decisões visuais (ASD/HINT/SINGLE_FACE)
            #     quando o gênero do áudio contradiz o nome atribuído.
            #     Ex: áudio masculino atribuído a "Manoela" via ASD → cria speaker novo.
            if audio_gender and decision.startswith(("ASD", "SINGLE_FACE", "SESSION_HINT")):
                assigned_name = verified_name or self.speaker_names.get(speaker_id)
                if assigned_name and not self._is_generic_name(assigned_name):
                    name_gender = _gender_of(assigned_name)
                    if name_gender and name_gender != audio_gender:
                        logger.debug(f"⚠️ Gender mismatch: áudio={audio_gender} vs nome={assigned_name}({name_gender}) — criando speaker separado")
                        rejected_id = speaker_id
                        speaker_id = self._get_or_create_session_speaker_id(audio_np, exclude_ids={rejected_id})
                        verified_name = None
                        decision = f"GENDER_OVERRIDE ({audio_gender}≠{name_gender})"

            # LOG: decisão final
            logger.info(f"🎯 Decisão: {decision} → {verified_name or speaker_id}")

            if verified_name:
                self.speaker_names[speaker_id] = verified_name
                if not self._is_generic_name(verified_name):
                    self.identified_speakers.add(speaker_id)  # rastreia por person_id

            # Acumula o chunk ANTES de tentar detectar nome, para que o próprio
            # segmento onde o nome é dito esteja disponível ao criar o embedding.
            speakers_full = self.num_speakers and len(self.identified_speakers) >= self.num_speakers
            current_name_pre = self.speaker_names.get(speaker_id)
            if (not current_name_pre or self._is_generic_name(current_name_pre)) and not speakers_full:
                self.unknown_speakers_audio[speaker_id].append(audio_np.copy())

            self._update_speaker_names_incremental(speaker_id, text, audio_np)
            final_name = self.speaker_names.get(speaker_id)

            # Sincroniza câmera: quando o speaker_id é um ID de áudio (não está em
            # active_faces), propaga o nome identificado para o person_id do vídeo.
            # Sync câmera: propaga nome de voz para rosto genérico visível.
            # Só sincroniza se a evidência de voz veio do VERIFIER (biometria real),
            # NÃO de ASD/SINGLE_FACE/SESSION — essas decisões já dependem do rosto,
            # então reescrevê-lo a partir delas cria loops de erro.
            # Camera sync só em alta confiança (VERIFIER_HIGH ≥ verifier_confidence_min)
            # VERIFIER_MOD (≥0.55) é muito fraco para renomear um rosto de forma irreversível.
            verifier_decision = decision.startswith("VERIFIER_HIGH")
            if final_name and not self._is_generic_name(final_name) and verifier_decision:
                video_pids = set(self.shared_state.get("active_faces", {}).values())
                if speaker_id not in video_pids:
                    # Determina o rosto alvo apenas via ASD (lip detection confirmado)
                    # Não usa heurística de 1 rosto visível: se o pai fala fora de câmera,
                    # a heurística renomearia o rosto errado.
                    if asd_person_id is not None and asd_person_id in video_pids:
                        cam_target = asd_person_id
                    else:
                        cam_target = None
                    if cam_target is not None:
                        current_cam = self.speaker_names.get(cam_target, "")
                        # Guard extra: não renomeia se o face tracker já confirmou outra identidade real
                        ft_pre = self.shared_state.get("face_tracker")
                        asd_tid_pre = None
                        for tid, pid in self.shared_state.get("active_faces", {}).items():
                            if pid == cam_target:
                                asd_tid_pre = tid
                                break
                        if ft_pre and asd_tid_pre is not None:
                            face_confirmed = ft_pre._track_to_name.get(asd_tid_pre)
                            if face_confirmed and not self._is_generic_name(face_confirmed) and face_confirmed != final_name:
                                logger.debug(f"📷 Camera sync BLOQUEADO: rosto confirmado como '{face_confirmed}' ≠ voz '{final_name}'")
                                cam_target = None  # cancela sync
                        if cam_target is not None and self._is_generic_name(current_cam):
                            self.speaker_names[cam_target] = final_name
                            self._emb_to_pid[final_name] = cam_target
                            ft = self.shared_state.get("face_tracker")
                            if ft and current_cam and current_cam not in ("Unknown", ""):
                                # Encontra o track_id do cam_target via active_faces
                                asd_track_for_rename = None
                                for tid, pid in self.shared_state.get("active_faces", {}).items():
                                    if pid == cam_target:
                                        asd_track_for_rename = tid
                                        break
                                ft.rename_person(current_cam, final_name, only_track_id=asd_track_for_rename)
                            logger.info(f"📷 Câmera sincronizada: '{current_cam or cam_target}' → '{final_name}'")

            display_label = f"({final_name})" if final_name else f"({speaker_id})"
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] 🎙️ {display_label}: {text}")
            self.full_transcript.append({"speaker": speaker_id, "verified_name": final_name, "text": text, "timestamp": datetime.now()})
        except Exception as e:
            print(f"Erro ao processar segmento: {e}")

    def _get_or_create_session_speaker_id(self, audio_np, exclude_ids=None):
        """Retorna ID estável de sessão para speaker desconhecido usando similaridade de voz."""
        exclude_ids = exclude_ids or set()
        signal = torch.from_numpy(audio_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Busca o speaker mais próximo entre os já conhecidos nesta sessão
        best_id, best_sim = None, -1.0
        for sid, known_emb in self._session_unknown_embs.items():
            if sid in exclude_ids:
                continue
            sim = float(np.dot(emb, known_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        # Threshold de 0.65 para separar speakers distintos sem fragmentar o mesmo speaker
        if best_id and best_sim >= 0.65:
            alpha = 0.1
            self._session_unknown_embs[best_id] = (1 - alpha) * self._session_unknown_embs[best_id] + alpha * emb
            return best_id

        # Respeita o limite de speakers: conta todos os person_ids únicos já atribuídos
        all_tracked = set(self.speaker_names.keys()) | set(self._session_unknown_embs.keys())
        if self.num_speakers and len(all_tracked) >= self.num_speakers:
            if best_id:
                return best_id
            # Fallback: pick first non-excluded session speaker
            for sid in self._session_unknown_embs:
                if sid not in exclude_ids:
                    return sid

        new_id = self._new_person_id()
        self._session_unknown_embs[new_id] = emb
        return new_id

    def _normalize_audio(self, audio, target_db=-20.0):
        if len(audio) == 0: return audio
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-6: return audio
        target_amplitude = 10**(target_db/20.0)
        scaling_factor = target_amplitude / rms
        normalized = (audio * scaling_factor).astype(np.float32)
        max_val = np.abs(normalized).max()
        if max_val > 1.0: normalized = normalized / max_val
        return normalized

    def _new_person_id(self):
        """Cria um person_id opaco estável, usando a factory do shared_state se disponível."""
        factory = self.shared_state.get("person_id_factory")
        if factory:
            return factory()
        self._session_unknown_counter += 1
        return f"spk_{self._session_unknown_counter:03d}"

    def _is_generic_name(self, name):
        """Retorna True se o nome/ID é genérico (spk_001, Person_1, Desconhecido_1, Unknown…)."""
        return name is not None and bool(re.match(r'^(spk_\d+|Person_\d+|Desconhecido_\d+|Unknown)$', name))

    def _extract_names_with_ner(self, text):
        if not self.nlp: return []
        try:
            doc = self.nlp(text)
            return [re.sub(r'^(Sr\.|Sra\.|Dr\.|Dra\.)\s+', '', ent.text.strip(), flags=re.IGNORECASE) for ent in doc.ents if ent.label_ == "PER" and len(ent.text.strip()) > 2]
        except Exception: return []

    def _detect_gender_from_text(self, text):
        text_lower = text.lower()
        extracted_names = self._extract_names_with_ner(text)
        for name in extracted_names:
            name_lower = name.lower().strip()
            name_normalized = name_lower.replace('é', 'e').replace('á', 'a').replace('à', 'a')
            if name_lower in NAMES_FEMALE or name_normalized in NAMES_FEMALE: return "female"
            if name_lower in NAMES_MALE or name_normalized in NAMES_MALE: return "male"
        female_markers = [r'\beu sou a\b', r'\bsou a\b', r'\bsou mulher', r'\bmeu nome é\s+([a-zà-ú]+a)\b']
        male_markers = [r'\beu sou o\b', r'\bsou o\b', r'\bsou homem', r'\bmeu nome é\s+([a-zà-ú]+o)\b']
        f_score = sum(1 for p in female_markers if re.search(p, text_lower))
        m_score = sum(1 for p in male_markers if re.search(p, text_lower))
        if f_score > m_score: return "female"
        if m_score > f_score: return "male"
        return None

    def _detect_gender_from_audio(self, audio_chunk):
        if librosa is None: return None
        try:
            f0 = librosa.yin(audio_chunk, fmin=50, fmax=500, sr=self.sample_rate)
            valid_f0 = f0[f0 > 0]
            if len(valid_f0) < 10: return None
            mean_f0 = np.median(valid_f0)
            if mean_f0 > 155: return "female"
            if mean_f0 < 140: return "male"
            return None
        except Exception: return None

    def _correct_name_for_gender(self, name, gender):
        if not gender: return name
        # Só corrige o PRIMEIRO nome — sobrenomes (Otávio, Marques) não devem ser alterados
        parts = name.split()
        first = parts[0].lower()
        if gender == "female":
            if first in MALE_TO_FEMALE:
                parts[0] = MALE_TO_FEMALE[first].capitalize()
            elif first.endswith('o') and first not in NAMES_FEMALE:
                parts[0] = (first[:-1] + 'a').capitalize()
        return ' '.join(parts)

    def _update_speaker_names_incremental(self, speaker_id, text, audio_chunk=None):
        current_name = self.speaker_names.get(speaker_id)
        is_rename = self._is_generic_name(current_name)
        # Já tem nome real — não sobrescreve
        if current_name and not is_rename:
            return
        # Só bloqueia por limite de speakers ao adicionar speaker novo (não ao renomear)
        if not is_rename and self.num_speakers and len(self.identified_speakers) >= self.num_speakers:
            return
        t_gender = self._detect_gender_from_text(text)
        a_gender = self._detect_gender_from_audio(audio_chunk) if audio_chunk is not None else None
        gender = t_gender or a_gender
        patterns = [r'(?:meu nome é|me chamo|eu sou|sou o|sou a)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)', r'(?:aqui é|aqui quem fala é)\s+(?:o|a)?\s*([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)']
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                name = self._correct_name_for_gender(m.group(1).title(), gender)
                # Se o nome já existe (ex: "José Marques" carregado do disco), assume
                # que é a mesma pessoa se re-apresentando. Homônimos reais são raros
                # em reuniões pequenas — prefere merge a sufixar "José Marques 2".
                display_name = name
                if display_name in self._emb_to_pid and self._emb_to_pid[display_name] != speaker_id:
                    # Merge: aponta o nome para o speaker_id atual da sessão
                    self._emb_to_pid[display_name] = speaker_id
                name = display_name
                old_name = current_name  # display name anterior (ou None)
                self.speaker_names[speaker_id] = name
                self._emb_to_pid[name] = speaker_id  # registra para futuros matches do verifier
                self.identified_speakers.add(speaker_id)  # rastreia por person_id
                self._save_live_embedding(speaker_id, name, old_name=old_name)
                return

    def _save_live_embedding(self, speaker_id, name, old_name=None):
        audio_chunks = self.unknown_speakers_audio.get(speaker_id, [])
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        emb_dir = os.path.join(base_dir, "data", "embeddings")

        face_tracker = self.shared_state.get("face_tracker")

        # Renomeia embeddings existentes do nome genérico para o nome real
        if old_name and self._is_generic_name(old_name):
            self.verifier.rename_embedding(old_name, name)
            if face_tracker:
                # Se old_name é "Unknown", a face pode já estar enrolled com outro nome
                # (ex: "Person_1") — procura pelo nome real no _track_to_name
                face_old = old_name
                target_track = None
                active_faces = self.shared_state.get("active_faces", {})
                for track_id, pid in active_faces.items():
                    if pid == speaker_id:
                        if old_name not in face_tracker.known_embeddings:
                            enrolled = face_tracker._track_to_name.get(track_id)
                            if enrolled and enrolled in face_tracker.known_embeddings:
                                face_old = enrolled
                        target_track = track_id
                        break
                face_tracker.rename_person(face_old, name, only_track_id=target_track)
            logger.info(f"✅ Identidade atualizada: '{old_name}' → '{name}'")

        # Se veio do session tracking, tenta vincular a um rosto genérico visível
        elif old_name is None and face_tracker:
            active_faces = self.shared_state.get("active_faces", {})
            # active_faces = {track_id: person_id} — encontra person_ids genéricos visíveis
            generic_pids = [
                pid for pid in set(active_faces.values())
                if pid != speaker_id and self._is_generic_name(self.speaker_names.get(pid, ""))
            ]
            if len(generic_pids) == 1:
                target_pid = generic_pids[0]
                old_face_name = self.speaker_names.get(target_pid, "")
                if old_face_name:
                    # Encontra o track_id associado ao target_pid
                    target_track = None
                    for tid, pid in active_faces.items():
                        if pid == target_pid:
                            target_track = tid
                            break
                    face_tracker.rename_person(old_face_name, name, only_track_id=target_track)
                    self.speaker_names[target_pid] = name
                    self._emb_to_pid[name] = target_pid
                    logger.info(f"✅ Câmera: '{old_face_name}' → '{name}'")

        if not audio_chunks:
            return
        norm_name = name.lower().strip()
        # Pula se já existe embedding para este nome — EXCETO quando acabamos de renomear
        # de um nome genérico (o áudio atual é mais recente e deve ser salvo)
        renamed_from_generic = old_name and self._is_generic_name(old_name)
        if not renamed_from_generic and os.path.exists(emb_dir):
            for f in os.listdir(emb_dir):
                fl = f.lower()
                # Exige correspondência exata do nome (seguido de '_' ou '.npy'),
                # para não confundir "João" com "João 2" etc.
                if fl == norm_name + ".npy" or fl.startswith(norm_name + "_"):
                    return
        combined_audio = np.concatenate(audio_chunks)
        mx = np.abs(combined_audio).max()
        if mx > 0: combined_audio = combined_audio / mx
        signal = torch.from_numpy(combined_audio).float().to(self.device).unsqueeze(0)
        try:
            with torch.no_grad():
                embedding = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
            os.makedirs(emb_dir, exist_ok=True)
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
            np.save(os.path.join(emb_dir, filename), embedding)
            logger.info(f"✅ Nova biometria de voz salva para {name}.")
            self.verifier.load_embeddings(emb_dir)
        except Exception as e:
            logger.error(f"Erro ao salvar biometria: {e}")

    def save_session(self, output_dir="realtime_sessions"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = os.path.join(output_dir, f"transcript_{ts}.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            for entry in self.full_transcript:
                f.write(f"[{entry['timestamp'].strftime('%H:%M:%S')}] {entry.get('verified_name') or entry['speaker']}: {entry['text']}\n")
        audio_file = None
        if self.all_audio_chunks:
            import wave
            audio_file = os.path.join(output_dir, f"audio_{ts}.wav")
            full_audio = np.concatenate(self.all_audio_chunks)
            pcm16 = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
            with wave.open(audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm16.tobytes())
            logger.info(f"🔊 Áudio salvo: {audio_file}")
        return transcript_file, audio_file

    def feed_audio_chunk(self, audio_data: np.ndarray):
        """Enfileira um chunk de áudio (np.float32, 16 kHz) — usado no modo arquivo."""
        chunk = audio_data.flatten().astype(np.float32)
        self.all_audio_chunks.append(chunk.copy())
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def stop(self):
        self.running = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)

    def start_file_mode(self):
        """Inicia só os threads de processamento (sem captura de microfone).
        Use feed_audio_chunk() para fornecer áudio externamente."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_audio_chunks, daemon=True)
        self.processing_thread.start()

    def start_recording(self, embeddings_dir=None):
        self._check_audio_backend()
        self.running = True
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.processing_thread = threading.Thread(target=self._process_audio_chunks, daemon=True)
        self.recording_thread.start()
        self.processing_thread.start()
        try:
            while self.running:
                self.recording_thread.join(timeout=0.5)
                if not self.recording_thread.is_alive(): break
        except KeyboardInterrupt:
            self.stop()
        return self.save_session()