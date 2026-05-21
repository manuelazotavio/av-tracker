import os
import sys
import json
import logging
import numpy as np
import re
import unicodedata
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # only our logs at DEBUG level
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

# Portuguese male first names -- data set for gender detection (do not translate)
NAMES_MALE = {
    'lucas', 'mateus', 'pedro', 'joão', 'gabriel', 'rafael', 'bruno', 'carlos', 'andré',
    'fernando', 'ricardo', 'rodrigo', 'marcelo', 'paulo', 'thiago', 'vitor', 'daniel',
    'augusto', 'leonardo', 'eduardo', 'henrique', 'diego', 'felipe', 'guilherme',
    'kauan', 'marcos', 'jorge', 'cesar', 'antonio', 'luis', 'francisco', 'manuel',
    'xavier', 'mario', 'sergio', 'alberto', 'manoel', 'josé', 'jose', 'joaquim',
    'arthur', 'miguel', 'enzo', 'bernardo', 'heitor', 'davi', 'murilo', 'caio',
    'christopher', 'gustavo', 'william', 'nicolas', 'victor', 'rafael', 'ryan',
    'nicolas', 'renan', 'leandro', 'alex', 'alexandre', 'anderson', 'alan'
}

# Portuguese female first names -- data set for gender detection (do not translate)
NAMES_FEMALE = {
    'maria', 'ana', 'julia', 'beatriz', 'leticia', 'amanda', 'carolina', 'fernanda',
    'mariana', 'juliana', 'patricia', 'camila', 'bruna', 'aline', 'bianca', 'renata',
    'manuela', 'barbara', 'iris', 'ines', 'isis', 'isadora', 'ivana', 'isabel', 'joana',
    'josefa', 'joséfina', 'josiane', 'jovita', 'joyceana', 'judite', 'julieta',
    'jussara', 'justina', 'justine', 'juvita', 'kaila', 'karina', 'karla', 'carla',
    'cassandra', 'cecilia', 'célia', 'celina', 'celia', 'clara', 'clarice', 'claudia',
    'cleide', 'cleonice', 'conceição', 'consuelo', 'constanza', 'corina',
    'cornelia', 'coromila', 'corsina', 'cosma', 'covadonga', 'rosa', 'rosana', 'roselia',
    'vitoria', 'vanessa', 'valeria', 'vanira', 'veronica', 'verena', 'vera', 'vidya',
    'viviana', 'violeta', 'vilma', 'virgilia', 'virginia', 'antonia', 'anatolia'
}

# Map from masculine to feminine form of Portuguese names (do not translate)
MALE_TO_FEMALE = {
    'manoel': 'manuela', 'manuel': 'manuela', 'gabriel': 'gabriela', 'ricardo': 'ricarda',
    'luis': 'luisa', 'francisco': 'francisca', 'antonio': 'antonia', 'rio': 'ria',
    'mario': 'maria', 'carlos': 'carla', 'paulo': 'paula', 'julio': 'julia',
    'sergio': 'serbia', 'pedro': 'petra'
}


def _normalize_name_token(name: str) -> str:
    """Normalize name token for robust comparisons (Joao == João)."""
    if not name:
        return ""
    token = name.strip().lower()
    token = unicodedata.normalize("NFKD", token)
    token = "".join(ch for ch in token if not unicodedata.combining(ch))
    return token


NAMES_MALE_NORMALIZED = {_normalize_name_token(n) for n in NAMES_MALE}
NAMES_FEMALE_NORMALIZED = {_normalize_name_token(n) for n in NAMES_FEMALE}


def _gender_of(name: str) -> str | None:
    """Return 'male'/'female' based on first name lookup, or None if unknown."""
    base = _normalize_name_token(name.split()[0] if name else "")
    if base in NAMES_FEMALE_NORMALIZED:
        return "female"
    if base in NAMES_MALE_NORMALIZED:
        return "male"
    return None


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
        logger.error(f"Error loading dependencies: {e}")
        raise

class RealtimeTranscriber:
    def __init__(self, verifier, hf_token, whisper_size="medium", device="cuda", use_ai_analysis=True, diarization_clustering_threshold=0.6, verifier_confidence_min=0.8, chunk_duration=2.0, sample_rate=16000, shared_state=None, num_speakers=None, audio_device=None, language="pt"):
        _init_heavy_deps()
        self.verifier = verifier
        self.device = device
        self.language = language  # Whisper transcription language ("pt", "en", ...)
        self.use_ai_analysis = use_ai_analysis
        self.diarization_clustering_threshold = diarization_clustering_threshold
        self.verifier_confidence_min = verifier_confidence_min
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.shared_state = shared_state or {"active_faces": {}}
        self.num_speakers = num_speakers
        self.audio_device = audio_device  # None = system default
        self.audio_queue = queue.Queue(maxsize=10000)
        self.running = False
        self.recording_thread = None
        self.audio_buffer = deque(maxlen=int(sample_rate * chunk_duration * 2))
        # Rolling buffer of (t_start, audio_float32) chunks for Light-ASD sync
        self.shared_state.setdefault("asd_audio_buf", deque(maxlen=50))
        self.processing_buffer = []
        self.speaker_history = {}
        self.full_transcript = []
        self.all_audio_chunks = []
        self.unknown_speakers_audio = defaultdict(list)
        # person_id -> display_name (shared with the video side via shared_state)
        self.speaker_names = self.shared_state.setdefault("person_names", {})
        # embedding_key (name in verifier) -> stable person_id
        self._emb_to_pid = self.shared_state.setdefault("embedding_to_person", {})
        self.identified_speakers = set()   # set of person_ids with confirmed real name
        self._voice_emb_saved = set()      # speaker_ids with auto-saved voice embeddings
        self._fv_bind_last = {}            # name -> timestamp of last face-voice binding
        self._session_to_face = {}         # session_id -> person_id (face-confirmed binding)
        self._session_gender = {}          # session_id -> 'male'/'female'
        self._session_unknown_embs = {}    # person_id -> np.array embedding
        self._session_unknown_counter = 0
        if num_speakers:
            logger.info(f"Speaker limit: {num_speakers}")
        login(token=hf_token)
        self.llm = None  # not used
        # spaCy NER — local, fast, always loaded. Used to validate detected
        # names so discourse markers ("Yeah,", "So,", "Well,") are not mistaken
        # for vocatives. Model is chosen to match the transcription language.
        self.nlp = None
        if spacy is not None:
            _spacy_model = "en_core_web_sm" if self.language == "en" else "pt_core_news_sm"
            for _m in (_spacy_model, "pt_core_news_sm", "en_core_web_sm"):
                try:
                    self.nlp = spacy.load(_m)
                    logger.info(f"spaCy NER model: {_m}")
                    break
                except OSError:
                    continue
            if self.nlp is None:
                logger.warning("No spaCy model available — NER name validation disabled.")
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device(device))
        if self.diarization_clustering_threshold is not None:
            try:
                self.pipeline.instantiate({"clustering": {"threshold": self.diarization_clustering_threshold}, "segmentation": {"threshold": 0.4}})
            except Exception:
                pass
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        self._sepformer_local_path = "pretrained_models/sepformer"
        if not os.path.exists(os.path.join(self._sepformer_local_path, "hyperparams.yaml")):
            snapshot_download(repo_id="speechbrain/sepformer-whamr", local_dir=self._sepformer_local_path, local_dir_use_symlinks=False)
        self.separator = None  # loaded lazily on first separation needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self.whisper = WhisperModel(whisper_size, device=self.device, compute_type="float16")
        except Exception:
            self.whisper = WhisperModel(whisper_size, device=self.device, compute_type="int8")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

        # Perf tracking
        self._perf = defaultdict(list)
        self._perf_chunk_count = 0
        self._perf_summary_interval = 5  # print summary every N processed chunks

        # False-negative tracking
        self._fn_events = []  # list of dicts with FN details

        # Contextual speaker naming: infer names from conversation context
        self._pending_addressee = None   # {"name": str, "from_speaker": str, "ts": float}
        self._context_names = set()      # names mentioned in conversation (participant pool)
        self._addressee_votes = defaultdict(lambda: defaultdict(int))  # speaker_id -> {name: count}

        # LLM-based contextual speaker identification
        self._llm_analysis_interval = 10  # run LLM analysis every N segments
        self._llm_last_analysis = 0       # segment count at last analysis
        self._llm_client = None           # lazy-initialized HF InferenceClient

        # Structured metrics for every processed segment (saved as JSON)
        self._segment_metrics = []
        self._session_start = datetime.now()
        self._whisper_size = whisper_size

    def _log_perf(self, key: str, ms: float):
        self._perf[key].append(ms)
        logger.debug(f"PERF {key}: {ms:.0f}ms")

    def _print_perf_summary(self):
        lines = ["=== Audio Perf Summary ==="]
        for key in sorted(self._perf):
            vals = self._perf[key]
            if vals:
                lines.append(f"  {key:30s} avg={np.mean(vals):6.0f}ms  min={min(vals):5.0f}ms  max={max(vals):5.0f}ms  n={len(vals)}")
        print("\n".join(lines))

    def _print_fn_summary(self):
        """Print a summary of potential false-negative events detected during the session."""
        if not self._fn_events:
            logger.info("[FN Summary] No false-negative events detected.")
            return

        lines = [f"\n{'='*60}", f"  FALSE NEGATIVE REPORT — {len(self._fn_events)} event(s)", f"{'='*60}"]

        # Aggregate by reason type
        reason_counts = defaultdict(int)
        reason_examples = defaultdict(list)
        for ev in self._fn_events:
            for r in ev["reasons"]:
                rtype = r.split(":")[0]
                reason_counts[rtype] += 1
                if len(reason_examples[rtype]) < 3:  # keep up to 3 examples
                    reason_examples[rtype].append(ev)

        lines.append("\n  By reason:")
        for rtype, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {rtype}: {count}")

        lines.append(f"\n  Details (showing up to 15):")
        for i, ev in enumerate(self._fn_events[:15]):
            lines.append(f"    [{ev['timestamp']}] {' | '.join(ev['reasons'])}")
            lines.append(f"      decision={ev['decision']} final={ev['final']} "
                         f"raw_best={ev['raw_best']} gender={ev['audio_gender']} faces={ev['faces']}")
            lines.append(f"      text=\"{ev['text']}\"")
            if ev["all_candidates"]:
                cands = " | ".join(f"{n}:{s:.3f}" for n, s in ev["all_candidates"])
                lines.append(f"      candidates=[{cands}]")

        lines.append(f"{'='*60}")
        summary = "\n".join(lines)
        print(summary)
        logger.info(summary)

    def _get_separator(self):
        """Load SepFormer lazily the first time it is needed."""
        if self.separator is None:
            print("\nLoading SepFormer (first separation)...")
            self.separator = SepformerSeparation.from_hparams(
                source=self._sepformer_local_path,
                savedir=self._sepformer_local_path,
                run_opts={"device": self.device}
            )
        return self.separator

    def _check_audio_backend(self):
        if AUDIO_BACKEND is None:
            raise RuntimeError("No audio backend available!")

    def _record_audio(self):
        if AUDIO_BACKEND == "sounddevice":
            self._record_sounddevice()
        elif AUDIO_BACKEND == "pyaudio":
            self._record_pyaudio()

    def _record_sounddevice(self):
        try:
            # Capture at the device's native sample rate to avoid PortAudio resampling
            # artifacts (pitch shift). Then resample manually to self.sample_rate (16000).
            dev_idx = self.audio_device if self.audio_device is not None else sd.default.device[0]
            device_info = sd.query_devices(dev_idx)
            native_sr = int(device_info['default_samplerate'])
            target_sr = self.sample_rate  # 16000
            logger.info(f"Device: {device_info['name']} | native={native_sr} Hz -> target={target_sr} Hz")

            block_frames = int(native_sr * 0.1)  # 100 ms at native rate
            resampler = T.Resample(orig_freq=native_sr, new_freq=target_sr) if native_sr != target_sr else None

            with sd.InputStream(device=self.audio_device, channels=1, samplerate=native_sr,
                                blocksize=block_frames, dtype=np.float32) as stream:
                while self.running:
                    data, overflowed = stream.read(block_frames)
                    audio_data = data.flatten().astype(np.float32)
                    if resampler is not None:
                        audio_data = resampler(torch.from_numpy(audio_data)).numpy()
                    self.all_audio_chunks.append(audio_data.copy())
                    self.audio_queue.put(audio_data)
                    # Feed Light-ASD rolling buffer with timestamps
                    self.shared_state["asd_audio_buf"].append(
                        (time.time() - len(audio_data) / target_sr, audio_data.copy())
                    )
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
                self.shared_state["asd_audio_buf"].append(
                    (time.time() - len(audio_data) / self.sample_rate, audio_data.copy())
                )
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            self.running = False

    def _process_audio_chunks(self):
        # Sliding window with large overlap for consistent diarization labels
        # between consecutive chunks.
        #   window  : 15 s  -- enough context for pyannote to stabilize labels
        #   step    :  5 s  -- advances 5 s per iteration (10 s overlap)
        #   new zone: last 5 s -- only part transcribed; the first 10 s are context
        window_samples   = int(self.sample_rate * 15.0)
        step_samples     = int(self.sample_rate * 5.0)
        new_zone_samples = window_samples - step_samples   # 10 s * sample_rate

        # First chunk: 8 s window for faster initial identification.
        # Consumes the entire chunk (no prior overlap). The second chunk (15 s) also
        # processes everything (zone_start=0) since there is no overlap from the first.
        # From the third chunk onward, the normal sliding window with 10 s context kicks in.
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
                        # First chunk: 8 s, consume all
                        chunk_to_process = np.array(self.processing_buffer[:first_window_samples], dtype=np.float32)
                        self.processing_buffer = self.processing_buffer[first_window_samples:]
                        chunk_wall_start = time.time() - first_window_samples / self.sample_rate
                        zone_start = 0
                    else:
                        chunk_to_process = np.array(self.processing_buffer[:window_samples], dtype=np.float32)
                        self.processing_buffer = self.processing_buffer[step_samples:]
                        chunk_wall_start = time.time() - window_samples / self.sample_rate
                        # Second chunk: no overlap from the first -> process all
                        zone_start = 0 if _chunks_processed == 1 else new_zone_samples
                    self._process_chunk(chunk_to_process,
                                        chunk_wall_start=chunk_wall_start,
                                        new_zone_start_samples=zone_start)
                    _chunks_processed += 1
                    required = window_samples
            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                logger.error(f"Error in processing: {e}\n{traceback.format_exc()}")

    def _process_chunk(self, audio_chunk, chunk_wall_start: float | None = None, new_zone_start_samples: int = 0):
        if not self.running:
            return
        _t0_chunk = time.perf_counter()
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

            # Diarization to separate speech turns by speaker
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            diarization_params = {}
            if self.num_speakers:
                # Force pyannote to find exactly N speakers when the user
                # specifies the number. Without min_speakers, pyannote may collapse
                # everything into 1 speaker in chunks with fast dialogue.
                diarization_params['min_speakers'] = self.num_speakers
                diarization_params['max_speakers'] = self.num_speakers
            try:
                _t0_diar = time.perf_counter()
                diarization = self.pipeline(
                    {"waveform": audio_tensor, "sample_rate": 16000},
                    **diarization_params
                )
                self._log_perf("diarization", (time.perf_counter() - _t0_diar) * 1000)
                turns = [
                    (turn, spk)
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                    if (turn.end - turn.start) >= 0.5
                ]
            except Exception as e:
                logger.warning(f"Diarization failed ({e}). Processing as single speaker.")
                turns = []

            if not turns:
                # No diarization -- process only the new zone to avoid re-transcribing context
                new_zone = audio_np[new_zone_start_samples:]
                if len(new_zone) >= int(16000 * 0.5):
                    self._process_segment(
                        new_zone,
                        wall_time_start=chunk_wall_start + new_zone_start_samples / 16000,
                        wall_time_end=chunk_wall_start + len(audio_np) / 16000,
                    )
                return

            distinct_speakers = {spk for _, spk in turns}

            # --- SepFormer: when >= 2 speakers detected by diarization ---
            # Uses source separation ONLY for speaker identification (verifier).
            # Transcription always uses original (non-separated) audio to preserve quality.
            # SepFormer resamples to 8kHz and back, which degrades Whisper accuracy.
            _sep_spk_to_session = {}
            if len(distinct_speakers) >= 2:
                _t0_sep = time.perf_counter()
                separated = self._separate_with_sepformer(audio_np)
                self._log_perf("sepformer", (time.perf_counter() - _t0_sep) * 1000)
                if separated is not None:
                    # Compute session_id per diarization speaker for hint matching
                    _sep_spk_segments = defaultdict(list)
                    for turn, spk in turns:
                        s = int(turn.start * 16000)
                        e = min(int(turn.end * 16000), len(audio_np))
                        seg = audio_np[s:e]
                        if len(seg) >= int(16000 * 0.5):
                            _sep_spk_segments[spk].append(seg)
                    _sep_spk_to_session = {
                        spk: self._get_or_create_session_speaker_id(np.concatenate(segs))
                        for spk, segs in _sep_spk_segments.items()
                    }
                    # Match separated streams to session speakers (for speaker ID only)
                    self._match_streams_to_speakers(separated, _sep_spk_to_session)
                    logger.debug(f"SepFormer: speaker hints computed for {len(distinct_speakers)} speakers")
                # Fall through to diarization-based processing with ORIGINAL audio

            # --- Default diarization-based processing (1 speaker or SepFormer failed) ---
            # Group consecutive turns into segments. Diarization sometimes labels
            # two DIFFERENT people with the SAME label inside one chunk, so a
            # matching label is not enough to merge — the actual voice (ECAPA
            # cosine similarity) must also match. Otherwise one segment ends up
            # containing two speakers and gets attributed to just one of them.
            MIN_SEGMENT_S  = 2.0
            SAME_VOICE_SIM = 0.40   # consecutive same-label turns merge only if voices match
            merged_groups = []  # list of [spk, start_sample, end_sample, voice_emb]
            for turn, spk in turns:
                start_s = int(turn.start * 16000)
                end_s   = min(int(turn.end * 16000), len(audio_np))
                seg_start = max(start_s, new_zone_start_samples)
                if seg_start >= end_s:
                    continue
                turn_emb = self._voice_embedding(audio_np[seg_start:end_s])
                _same_voice, _sim = True, 1.0
                if merged_groups and merged_groups[-1][3] is not None and turn_emb is not None:
                    _sim = float(np.dot(merged_groups[-1][3], turn_emb))
                    _same_voice = _sim >= SAME_VOICE_SIM
                if merged_groups and merged_groups[-1][0] == spk and _same_voice:
                    # Same diarization label AND same voice — extend the group
                    merged_groups[-1][2] = end_s
                else:
                    if merged_groups and merged_groups[-1][0] == spk and not _same_voice:
                        logger.debug(f"Turn split: label '{spk}' kept but voice differs "
                                     f"(sim={_sim:.2f} < {SAME_VOICE_SIM}) — separate speaker")
                    merged_groups.append([spk, seg_start, end_s, turn_emb])

            for spk, seg_start, seg_end, _g_emb in merged_groups:
                segment = audio_np[seg_start:seg_end]
                seg_dur = len(segment) / 16000
                if seg_dur < 0.5:
                    continue
                # Session hint from this group's OWN single-voice audio — never
                # from the diarization label, which may lump two speakers together.
                _g_gender = self._detect_gender_from_audio(segment) if librosa is not None else None
                _sid = self._get_or_create_session_speaker_id(segment, audio_gender=_g_gender)
                _face_pid = self._session_to_face.get(_sid)
                if _face_pid and _face_pid in self.speaker_names:
                    _sid = _face_pid
                # For very short segments (<2s), try to extend with surrounding audio
                # to give Whisper enough context for decent transcription
                if seg_dur < MIN_SEGMENT_S:
                    pad_needed = int((MIN_SEGMENT_S - seg_dur) * 16000)
                    pad_before = min(seg_start - new_zone_start_samples, pad_needed // 2)
                    pad_after = min(len(audio_np) - seg_end, pad_needed - pad_before)
                    segment = audio_np[seg_start - pad_before:seg_end + pad_after]
                self._process_segment(
                    segment,
                    session_hint=_sid,
                    wall_time_start=chunk_wall_start + seg_start / 16000,
                    wall_time_end=chunk_wall_start + seg_end / 16000,
                )
        except Exception as e:
            print(f"Error processing chunk: {e}")
        finally:
            self._log_perf("chunk_total", (time.perf_counter() - _t0_chunk) * 1000)
            self._perf_chunk_count += 1
            if self._perf_chunk_count % self._perf_summary_interval == 0:
                self._print_perf_summary()

    def _separate_with_sepformer(self, audio_np: np.ndarray) -> list[np.ndarray] | None:
        """Separate mixed audio into 2 streams using SepFormer (requires 8 kHz internally)."""
        try:
            # 16 kHz -> 8 kHz (SepFormer was trained at 8 kHz)
            down = T.Resample(16000, 8000)
            audio_8k = down(torch.from_numpy(audio_np).float()).unsqueeze(0).to(self.device)

            with torch.no_grad():
                est = self._get_separator().separate_batch(audio_8k)
            # est: [1, T_8k, n_sources]

            # 8 kHz -> 16 kHz
            up = T.Resample(8000, 16000)
            streams = []
            n_sources = est.shape[-1]
            for i in range(n_sources):
                src = est[0, :, i].cpu()
                src_16k = up(src.unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
                streams.append(src_16k)
            return streams
        except Exception as e:
            logger.warning(f"SepFormer failed: {e}")
            return None

    def _match_streams_to_speakers(self, streams: list[np.ndarray],
                                   spk_to_session: dict) -> dict[int, str]:
        """Associate SepFormer stream index -> session speaker_id by embedding similarity."""
        if not spk_to_session:
            return {}
        # Embeddings of the separated streams
        stream_embs = []
        for stream in streams:
            sig = torch.from_numpy(stream).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                emb = self.classifier.encode_batch(sig).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            stream_embs.append(emb / norm if norm > 0 else emb)

        # Embeddings of speakers already tracked (from _session_unknown_embs)
        spk_embs = {
            spk: self._session_unknown_embs[sid]
            for spk, sid in spk_to_session.items()
            if sid in self._session_unknown_embs
        }
        if not spk_embs:
            # No reference -> assign 1:1 by order
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
        """Process a speech segment from a single speaker."""
        _t0_seg = time.perf_counter()
        try:
            # Spam / subtitle / YouTube artifact phrases to filter out of transcription.
            # Portuguese phrases kept as-is because they match actual PT-BR transcription output.
            bad_phrases = ["Amara.org", "Legendas", "Obrigado", "tchau gente", "tchau tchau",
                           "transmissão", "inscreva-se", "obrigada por assistir",
                           "continue assistindo", "não se esqueça",
                           "estou ouvindo", "i'm listening", "subtitles by",
                           # Whisper YouTube-training hallucinations (PT-BR)
                           "se inscreva", "inscreva no canal", "sininho", "ative o",
                           "ative as notificações", "deixe seu like", "deixa o like",
                           "curta o vídeo", "curtam o vídeo", "compartilhe", "comentários",
                           "próximo vídeo", "valeu galera", "até a próxima",
                           "nos vemos", "obrigado por assistir"]
            try:
                _t0_whisper = time.perf_counter()
                # VAD always on: Silero strips non-speech so Whisper does not
                # hallucinate YouTube phrases on silent/low-speech segments.
                # speech_pad_ms keeps 300ms around real speech to preserve context.
                _whisper_prompt = (
                    "Transcrição de conversa em português brasileiro."
                    if self.language == "pt"
                    else "Transcript of a conversation."
                )
                segs_gen, _ = self.whisper.transcribe(
                    audio_np,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                    condition_on_previous_text=False,
                    without_timestamps=False,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    # Prompt hint to guide Whisper toward the conversation style
                    initial_prompt=_whisper_prompt,
                )
                segs_list = list(segs_gen)
                _whisper_ms = (time.perf_counter() - _t0_whisper) * 1000
                self._log_perf("whisper", _whisper_ms)
            except Exception:
                return
            if not segs_list:
                return
            # Discard segments where Whisper is not confident there is real speech
            avg_no_speech = sum(getattr(s, 'no_speech_prob', 0.0) for s in segs_list) / len(segs_list)
            if avg_no_speech > 0.5:
                return
            # Very low avg_logprob indicates hallucination (Whisper making up text)
            avg_logprob = sum(getattr(s, 'avg_logprob', 0.0) for s in segs_list) / len(segs_list)
            text = " ".join([s.text for s in segs_list]).strip()
            if not text or len(text) < 5 or any(bp.lower() in text.lower() for bp in bad_phrases):
                return
            # Prompt echo: Whisper regurgitates initial_prompt on low-speech audio
            # ("Apesar de conversa em português brasileiro." etc.)
            if SequenceMatcher(None, text.lower(), _whisper_prompt.lower()).ratio() > 0.6:
                logger.debug(f"Discarded prompt-echo hallucination: {text!r}")
                return
            # Detect hallucination by repetition: "X. X. X." or "X X X"
            _parts = [p.strip() for p in re.split(r"[.!?,;]+", text) if p.strip()]
            if len(_parts) >= 3 and len(set(p.lower() for p in _parts)) == 1:
                return  # all fragments are identical -> hallucination
            # Caption/subtitle artifacts that are hallucinations ONLY when they
            # are the entire transcription. Matched against the whole text, so
            # legit uses inside a sentence ("preste atenção") are not filtered.
            _exact_halluc = {"atenção", "música", "legenda", "legendas", "aplausos",
                             "risos", "obrigado", "obrigada", "fim", "the end"}
            _text_norm = text.lower().strip(" .!?,;:\"'-[]()")
            if _text_norm in _exact_halluc:
                logger.debug(f"Discarded caption-artifact hallucination: {text!r}")
                return
            # Short phrases with low logprob or moderate no_speech -> likely hallucination
            if len(text) < 20 and (avg_logprob < -0.8 or avg_no_speech > 0.3):
                return
            # Single short word with any elevated no-speech probability — a very
            # common hallucination on silence/noise that VAD let through.
            _words = _text_norm.split()
            if len(_words) <= 1 and len(_text_norm) < 14 and avg_no_speech > 0.15:
                logger.debug(f"Discarded single-word hallucination: {text!r} "
                             f"(no_speech={avg_no_speech:.2f})")
                return

            # Identify the speaker: verifier -> visual -> voice tracking
            _t0_ver = time.perf_counter()
            raw_best_name, raw_conf, all_candidates = self.verifier._process_audio_chunk(audio_np, sample_rate=16000)
            _verifier_ms = (time.perf_counter() - _t0_ver) * 1000
            self._log_perf("verifier", _verifier_ms)

            # --- Gender-aware selection + margin between verifier candidates ---
            audio_gender = self._detect_gender_from_audio(audio_np) if librosa is not None else None

            # LOG: verifier candidates and detected gender
            # Always show the best score (even below threshold) for diagnostics
            if all_candidates:
                cand_str = " | ".join(f"{n}: {s:.2f}" for n, s in all_candidates[:4])
            elif raw_conf > 0:
                cand_str = f"(below thr) {raw_best_name}: {raw_conf:.2f}"
            else:
                cand_str = "no embedding"
            logger.debug(f"Verifier: [{cand_str}] gender={audio_gender} threshold={self.verifier.threshold}")

            # Filter candidates: remove those that contradict the audio gender
            if audio_gender and all_candidates:
                filtered = []
                for cand_name, cand_score in all_candidates:
                    cand_g = _gender_of(cand_name)
                    if cand_g and cand_g != audio_gender:
                        continue  # gender contradicts -> skip
                    filtered.append((cand_name, cand_score))
                if not filtered:
                    filtered = all_candidates  # fallback: if everything was filtered, use original
            else:
                filtered = all_candidates

            # Margin check on filtered candidates: best vs second best
            real_name, conf = "Unknown", 0.0
            if filtered:
                best_name, best_score = filtered[0]
                if len(filtered) >= 2:
                    _, second_score = filtered[1]
                    margin = best_score - second_score
                else:
                    margin = 1.0  # single candidate -> maximum margin
                if margin >= 0.04:
                    real_name, conf = best_name, best_score
                elif audio_gender:
                    # Low margin but gender was determined -- trust the gender filter
                    real_name, conf = best_name, best_score

            # LOG: result after filtering
            filt_str = " | ".join(f"{n}: {s:.2f}" for n, s in filtered[:4]) if filtered else "none"
            logger.debug(f"Filtered: [{filt_str}] -> chosen={real_name} conf={conf:.2f}")

            active_faces = self.shared_state.get("active_faces", {})
            visual_people = list(active_faces.values())

            # --- ASD: query which face was moving during the segment ---
            asd_person_id = None
            asd_track_id = None
            _ASD_PAD = 0.4  # expand ASD query window to capture frames near segment edges
            asd = self.shared_state.get("asd")
            if asd and wall_time_start is not None and active_faces:
                t_end = wall_time_end if wall_time_end is not None else wall_time_start + len(audio_np) / 16000
                asd_track = asd.get_active_speaker(wall_time_start - _ASD_PAD, t_end + _ASD_PAD)
                if asd_track is not None and asd_track in active_faces:
                    asd_person_id = active_faces[asd_track]
                    asd_track_id = asd_track

            # LOG: visual state
            faces_str = ", ".join(f"t{tid}={self.speaker_names.get(pid, pid)}" for tid, pid in active_faces.items())
            asd_name = self.speaker_names.get(asd_person_id, asd_person_id) if asd_person_id else "None"
            logger.debug(f"Faces: [{faces_str}] ASD={asd_name}")

            # Gender sanity check: reject verifier match if audio gender clearly
            # contradicts the candidate name (e.g., female voice → "Arthur").
            # This catches contaminated embeddings (wrong person's voice saved under another name).
            _verifier_name_gender = _gender_of(real_name) if real_name != "Unknown" else None
            # Skip gender check when verifier confidence is very high (≥ 0.90) —
            # biometric voice match is far more reliable than audio pitch analysis.
            _verifier_gender_ok = (
                conf >= 0.90
                or not (audio_gender and _verifier_name_gender and audio_gender != _verifier_name_gender)
            )
            _raw_name_gender = _gender_of(raw_best_name) if raw_best_name and raw_best_name != "Unknown" else None
            _raw_gender_ok = not (audio_gender and _raw_name_gender and audio_gender != _raw_name_gender)

            if real_name != "Unknown" and conf >= self.verifier_confidence_min and _verifier_gender_ok:
                # 1) Verifier with high confidence -- most reliable source
                if real_name not in self._emb_to_pid:
                    pid = self._new_person_id()
                    self._emb_to_pid[real_name] = pid
                    self.speaker_names[pid] = real_name
                speaker_id = self._emb_to_pid[real_name]
                verified_name = real_name
                decision = f"VERIFIER_HIGH ({conf:.2f})"
                # Auto-enroll: use high-confidence audio to enrich the voice bank
                if len(audio_np) >= 4 * self.sample_rate:
                    threading.Thread(
                        target=self._try_auto_enroll,
                        args=(real_name, audio_np.copy()),
                        daemon=True,
                    ).start()
            elif real_name != "Unknown" and conf >= self.verifier.threshold and _verifier_gender_ok:
                # 2) Verifier with moderate confidence -- voice
                #    biometrics is more reliable than ASD (lip pixel diff) especially
                #    when the speaker is off-camera.
                if real_name not in self._emb_to_pid:
                    pid = self._new_person_id()
                    self._emb_to_pid[real_name] = pid
                    self.speaker_names[pid] = real_name
                speaker_id = self._emb_to_pid[real_name]
                verified_name = real_name
                decision = f"VERIFIER_MOD ({conf:.2f})"
            elif raw_best_name and raw_best_name != "Unknown" and raw_conf >= 0.65 and _raw_gender_ok:
                # 3) Weak verifier -- needs STRONG corroboration (ASD or gender)
                #    single_face alone is NOT corroboration (just means 1 face visible)
                #    Gender CONTRADICTION (audio=female, name=male) vetoes ASD corroboration.
                _name_gender = _gender_of(raw_best_name)
                _gender_contradicts = audio_gender and _name_gender and audio_gender != _name_gender
                _corr = []
                if asd_person_id is not None and not _gender_contradicts:
                    _corr.append("ASD")
                if audio_gender and _name_gender == audio_gender:
                    _corr.append("gender")
                if _corr:
                    # Strip homonym suffix ("Manuela 2" → "Manuela")
                    _vw_name = re.sub(r'\s+\d+$', '', raw_best_name)
                    if _vw_name not in self._emb_to_pid:
                        pid = self._new_person_id()
                        self._emb_to_pid[_vw_name] = pid
                        self.speaker_names[pid] = _vw_name
                    speaker_id = self._emb_to_pid[_vw_name]
                    verified_name = _vw_name
                    decision = f"VERIFIER_WEAK ({raw_conf:.2f} +{'+'.join(_corr)})"
                else:
                    # No corroboration -- fall through to ASD/heuristics
                    if asd_person_id is not None:
                        speaker_id = asd_person_id
                        verified_name = self.speaker_names.get(asd_person_id)
                        decision = f"ASD ({asd_name})"
                    elif len(visual_people) == 1 and not self._is_generic_name(self.speaker_names.get(visual_people[0]) or ""):
                        speaker_id = visual_people[0]
                        verified_name = self.speaker_names.get(speaker_id)
                        decision = "FACE_ONLY"
                    elif len(visual_people) == 1:
                        speaker_id = visual_people[0]
                        verified_name = self.speaker_names.get(speaker_id)
                        decision = f"SINGLE_FACE ({raw_conf:.2f})"
                    elif session_hint is not None:
                        _face_bound = self._session_to_face.get(session_hint)
                        if _face_bound is not None and _face_bound in self.speaker_names:
                            speaker_id = _face_bound
                            verified_name = self.speaker_names.get(_face_bound)
                            decision = f"SESSION_FACE ({self.speaker_names.get(_face_bound, _face_bound)})"
                        else:
                            speaker_id = session_hint
                            verified_name = None
                            decision = "SESSION_HINT"
                    else:
                        speaker_id = self._get_or_create_session_speaker_id(audio_np, audio_gender=audio_gender)
                        verified_name = None
                        decision = f"SESSION_TRACKER ({speaker_id})"
            elif asd_person_id is not None:
                # 4) ASD detected who had their mouth moving -- face-voice binding
                speaker_id = asd_person_id
                verified_name = self.speaker_names.get(asd_person_id)
                decision = f"ASD ({asd_name})"
            elif len(visual_people) == 1 and not self._is_generic_name(self.speaker_names.get(visual_people[0]) or "") and (not self.num_speakers or self.num_speakers <= 1):
                # 4) FACE_ONLY: single face with confirmed identity -- ONLY when we know
                #    there's at most 1 speaker. With 2+ speakers, 1 visible face ≠ speaker.
                speaker_id = visual_people[0]
                verified_name = self.speaker_names.get(speaker_id)
                decision = "FACE_ONLY"
            elif len(visual_people) == 1 and raw_conf >= 0.35 and (not self.num_speakers or self.num_speakers <= 1):
                # 5) Heuristic: single generic face + audio sounds like human voice
                #    Same guard: skip when multiple speakers expected.
                speaker_id = visual_people[0]
                verified_name = self.speaker_names.get(speaker_id)
                decision = f"SINGLE_FACE ({raw_conf:.2f})"
            elif (len(visual_people) == 1 or len(visual_people) >= 2) and asd is not None and wall_time_start is not None:
                # 6) ASD_GUESS: try a weaker "best guess" based on relative mouth movement.
                #    Covers both cases:
                #    - 2+ faces: need dominance (ratio check inside get_best_guess)
                #    - 1 face + num_speakers>=2: FACE_ONLY/SINGLE_FACE are blocked,
                #      but any mouth movement is enough (1 face = trivially dominant)
                # Conflict check: reject ASD_GUESS if the face returned is already bound
                # to a DIFFERENT session_id. E.g.: face A is bound to SPEAKER_0 (Manuela),
                # but this segment is SPEAKER_1 (Kauan) — ASD_GUESS should not override.
                t_end_fb = wall_time_end if wall_time_end is not None else wall_time_start + len(audio_np) / 16000
                guess_track = asd.get_best_guess(wall_time_start - _ASD_PAD, t_end_fb + _ASD_PAD)
                _asd_guess_ok = False
                if guess_track is not None and guess_track in active_faces:
                    _guess_pid = active_faces[guess_track]
                    # Build inverse map: person_id → which session_id it was bound to
                    _bound_session = next(
                        (s for s, p in self._session_to_face.items() if p == _guess_pid), None
                    )
                    # Allow if: no prior binding, OR bound to the SAME session_hint
                    if _bound_session is None or session_hint is None or _bound_session == session_hint:
                        _asd_guess_ok = True
                    else:
                        logger.debug(
                            f"ASD_GUESS conflict: face {_guess_pid} is bound to "
                            f"{_bound_session}, session_hint={session_hint} — rejecting"
                        )
                if _asd_guess_ok:
                    asd_person_id = _guess_pid
                    asd_track_id = guess_track
                    speaker_id = asd_person_id
                    verified_name = self.speaker_names.get(asd_person_id)
                    decision = f"ASD_GUESS ({self.speaker_names.get(asd_person_id, asd_person_id)})"
                elif session_hint is not None:
                    _face_bound = self._session_to_face.get(session_hint)
                    if _face_bound is not None and _face_bound in self.speaker_names:
                        speaker_id = _face_bound
                        verified_name = self.speaker_names.get(_face_bound)
                        decision = f"SESSION_FACE ({self.speaker_names.get(_face_bound, _face_bound)})"
                    else:
                        speaker_id = session_hint
                        verified_name = None
                        decision = "SESSION_HINT"
                else:
                    speaker_id = self._get_or_create_session_speaker_id(audio_np, audio_gender=audio_gender)
                    verified_name = None
                    decision = f"SESSION_TRACKER ({speaker_id})"
            elif session_hint is not None:
                _face_bound = self._session_to_face.get(session_hint)
                if _face_bound is not None and _face_bound in self.speaker_names:
                    speaker_id = _face_bound
                    verified_name = self.speaker_names.get(_face_bound)
                    decision = f"SESSION_FACE ({self.speaker_names.get(_face_bound, _face_bound)})"
                else:
                    speaker_id = session_hint
                    verified_name = None
                    decision = "SESSION_HINT"
            else:
                speaker_id = self._get_or_create_session_speaker_id(audio_np, audio_gender=audio_gender)
                verified_name = None
                decision = f"SESSION_TRACKER ({speaker_id})"

            # --- Upgrade SESSION decisions: when the audio gender matches
            #     exactly one visible identified face, use it instead of spk_XXX.
            #     Safe when: all speakers are identified, OR there's only 1 identified
            #     person of that gender (e.g., only 1 woman → all female audio is hers).
            _all_identified = not self.num_speakers or len(self.identified_speakers) >= self.num_speakers
            _gender_unique = False
            if not _all_identified and audio_gender:
                # Count how many TOTAL speakers could be this gender (identified + context names)
                _id_same_gender = sum(
                    1 for pid in self.identified_speakers
                    if _gender_of(self.speaker_names.get(pid, "")) == audio_gender
                )
                _ctx_same_gender = sum(
                    1 for n in self._context_names
                    if n not in {self.speaker_names.get(p, "") for p in self.identified_speakers}
                    and _gender_of(n) == audio_gender
                )
                _total_same_gender = _id_same_gender + _ctx_same_gender
                # Only safe if exactly 1 person of this gender in the entire meeting
                _gender_unique = _id_same_gender == 1 and _total_same_gender <= 1
            if (_all_identified or _gender_unique) and decision.startswith(("SESSION_HINT", "SESSION_TRACKER")) and audio_gender and len(visual_people) >= 2:
                _gf_pid = None
                _gf_count = 0
                for vp in visual_people:
                    vp_name = self.speaker_names.get(vp, "")
                    if not self._is_generic_name(vp_name) and _gender_of(vp_name) == audio_gender:
                        _gf_pid = vp
                        _gf_count += 1
                if _gf_count == 1:
                    speaker_id = _gf_pid
                    verified_name = self.speaker_names.get(_gf_pid)
                    decision = f"GENDER_FACE ({audio_gender}->{verified_name})"

            # --- Gender cross-check: reject visual decisions (ASD/HINT/SINGLE_FACE)
            #     when the audio gender contradicts the assigned name.
            #     E.g.: male audio assigned to "Manoela" via ASD -> create new speaker.
            #     Only apply when gender is unambiguous (all identified, or unique gender).
            if (_all_identified or _gender_unique) and audio_gender and decision.startswith(("ASD", "FACE_ONLY", "SINGLE_FACE", "SESSION_HINT")):
                assigned_name = verified_name or self.speaker_names.get(speaker_id)
                # Special case: ASD pointed to a generic face (Person_N)
                # -> look for 1 non-generic face matching the audio gender for reroute
                if assigned_name and self._is_generic_name(assigned_name) and decision.startswith("ASD"):
                    _asd_gm_pid, _asd_gm_count = None, 0
                    for _pid in active_faces.values():
                        _pname = self.speaker_names.get(_pid, "")
                        if not self._is_generic_name(_pname) and _gender_of(_pname) == audio_gender:
                            _asd_gm_pid = _pid
                            _asd_gm_count += 1
                    if _asd_gm_count == 1:
                        _gm = self.speaker_names.get(_asd_gm_pid, _asd_gm_pid)
                        logger.debug(f"ASD generic reroute: '{assigned_name}' -> '{_gm}' ({audio_gender})")
                        speaker_id = _asd_gm_pid
                        verified_name = self.speaker_names.get(_asd_gm_pid)
                        decision = f"GENDER_REROUTE ({audio_gender}->{_gm})"
                elif assigned_name and not self._is_generic_name(assigned_name):
                    name_gender = _gender_of(assigned_name)
                    if name_gender and name_gender != audio_gender:
                        logger.debug(f"Gender mismatch: audio={audio_gender} vs name={assigned_name}({name_gender})")
                        # Look for another confirmed face with compatible gender in the frame
                        gender_match_pid = None
                        for pid in active_faces.values():
                            pid_name = self.speaker_names.get(pid, "")
                            if pid != speaker_id and not self._is_generic_name(pid_name) and _gender_of(pid_name) == audio_gender:
                                gender_match_pid = pid
                                break
                        if gender_match_pid is not None:
                            gm_name = self.speaker_names.get(gender_match_pid, gender_match_pid)
                            logger.debug(f"Gender reroute: '{assigned_name}' -> '{gm_name}'")
                            speaker_id = gender_match_pid
                            verified_name = self.speaker_names.get(gender_match_pid)
                            decision = f"GENDER_REROUTE ({audio_gender}->{gm_name})"
                        else:
                            # Diagnostic log: why did the reroute loop not find anyone?
                            for _dbg_tid, _dbg_pid in active_faces.items():
                                _dbg_name = self.speaker_names.get(_dbg_pid, "")
                                logger.debug(f"  reroute miss: t{_dbg_tid} pid={_dbg_pid} name='{_dbg_name}' same={_dbg_pid==speaker_id} generic={self._is_generic_name(_dbg_name)} gender={_gender_of(_dbg_name)}")
                            # Fallback: verifier hint with correct gender
                            if raw_best_name and raw_best_name != "Unknown" and raw_conf >= 0.15 and _gender_of(raw_best_name) == audio_gender:
                                if raw_best_name not in self._emb_to_pid:
                                    pid = self._new_person_id()
                                    self._emb_to_pid[raw_best_name] = pid
                                    self.speaker_names[pid] = raw_best_name
                                speaker_id = self._emb_to_pid[raw_best_name]
                                verified_name = raw_best_name
                                decision = f"GENDER_VERIFIER_HINT ({raw_best_name} {raw_conf:.2f})"
                            else:
                                rejected_id = speaker_id
                                speaker_id = self._get_or_create_session_speaker_id(audio_np, exclude_ids={rejected_id}, audio_gender=audio_gender)
                                verified_name = None
                                decision = f"GENDER_OVERRIDE ({audio_gender}!={name_gender})"

            # LOG: final decision
            logger.info(f"Decision: {decision} -> {verified_name or speaker_id}")

            # Bind session_hint to confirmed face decision so future segments of the
            # same diarization label resolve to the same face instead of spk_N.
            _face_decisions = ("ASD", "ASD_GUESS", "FACE_ONLY", "SINGLE_FACE",
                               "GENDER_FACE", "GENDER_REROUTE", "VERIFIER_HIGH",
                               "VERIFIER_MOD", "VERIFIER_WEAK")
            if session_hint is not None and any(decision.startswith(d) for d in _face_decisions):
                self._session_to_face[session_hint] = speaker_id
                # Mirror the session embedding into the face ID's slot so the
                # voice tracker can match directly to the face ID in future chunks,
                # unifying the two parallel ID spaces (spk_N ↔ Person_N).
                if (session_hint in self._session_unknown_embs
                        and speaker_id not in self._session_unknown_embs):
                    self._session_unknown_embs[speaker_id] = self._session_unknown_embs[session_hint].copy()
                    if session_hint in self._session_gender:
                        self._session_gender[speaker_id] = self._session_gender[session_hint]

            # --- FALSE NEGATIVE DETECTION ---
            # When the decision is NOT from the verifier, check if the verifier
            # had a plausible match that was rejected (potential missed identification).
            _fn_awareness = 0.45  # min score to consider a candidate "plausible"
            _fn_reasons = []  # always defined for metrics collection
            if not decision.startswith("VERIFIER"):

                # 1) Verifier had candidate above awareness but below operational threshold
                if raw_best_name and raw_best_name != "Unknown" and raw_conf >= _fn_awareness:
                    if raw_conf < self.verifier.threshold:
                        _fn_reasons.append(
                            f"BELOW_THRESHOLD: {raw_best_name}={raw_conf:.3f} "
                            f"(threshold={self.verifier.threshold})"
                        )
                    elif raw_conf < self.verifier_confidence_min:
                        _fn_reasons.append(
                            f"BELOW_CONF_MIN: {raw_best_name}={raw_conf:.3f} "
                            f"(conf_min={self.verifier_confidence_min})"
                        )

                # 2) Gender filter actually removed the best raw candidate
                #    Only report if it was in all_candidates but removed by gender
                #    (not if it was already below threshold)
                if raw_best_name and raw_best_name != "Unknown" and raw_conf >= _fn_awareness:
                    was_candidate = any(c[0] == raw_best_name for c in (all_candidates or []))
                    filtered_names = {c[0] for c in filtered} if filtered else set()
                    if was_candidate and raw_best_name not in filtered_names:
                        _fn_reasons.append(
                            f"GENDER_FILTERED: {raw_best_name}={raw_conf:.3f} "
                            f"removed (audio_gender={audio_gender})"
                        )

                # 3) Margin rejection: two candidates too close, no gender to break tie
                if filtered and len(filtered) >= 2:
                    _f0_name, _f0_score = filtered[0]
                    _f1_name, _f1_score = filtered[1]
                    _margin = _f0_score - _f1_score
                    if _f0_score >= _fn_awareness and _margin < 0.04 and not audio_gender:
                        _fn_reasons.append(
                            f"MARGIN_REJECT: {_f0_name}={_f0_score:.3f} vs "
                            f"{_f1_name}={_f1_score:.3f} (margin={_margin:.3f}<0.04)"
                        )

                if _fn_reasons:
                    _text_preview = text[:60] + "..." if len(text) > 60 else text
                    _fn_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "reasons": _fn_reasons,
                        "decision": decision,
                        "final": str(verified_name or speaker_id),
                        "text": _text_preview,
                        "raw_best": f"{raw_best_name}={raw_conf:.3f}" if raw_best_name else "none",
                        "all_candidates": [(n, round(s, 3)) for n, s in (all_candidates or [])[:4]],
                        "audio_gender": audio_gender,
                        "faces": len(active_faces),
                    }
                    self._fn_events.append(_fn_entry)
                    for reason in _fn_reasons:
                        logger.warning(f"[FN] {reason} | decision={decision} "
                                       f"final={verified_name or speaker_id} | \"{_text_preview}\"")

            if verified_name:
                self.speaker_names[speaker_id] = verified_name
                if not self._is_generic_name(verified_name):
                    self.identified_speakers.add(speaker_id)  # track by person_id

            # Always accumulate audio for every speaker — needed for creating
            # voice embeddings when they get identified (by name, context, or LLM).
            # Limit buffer to ~60s per speaker to prevent unbounded memory growth.
            _MAX_CHUNKS_PER_SPEAKER = 30  # ~60s at 2s/chunk
            buf = self.unknown_speakers_audio[speaker_id]
            buf.append(audio_np.copy())
            if len(buf) > _MAX_CHUNKS_PER_SPEAKER:
                buf.pop(0)

            # Auto-save voice embedding for speakers WITHOUT existing embeddings.
            # - Generic names (spk_N, Person_N): always save so validation can rename them.
            # - Named speakers already in verifier.embeddings: skip — _try_auto_enroll
            #   handles them with stricter quality checks. Saving here would contaminate
            #   their centroid if the verifier made a false-positive attribution.
            _MIN_CHUNKS_FOR_EMB = 1  # save after first segment (~2s)
            _current = self.speaker_names.get(speaker_id, speaker_id)
            _current = re.sub(r'\s+\d+$', '', _current)  # strip homonym suffix
            _has_existing_emb = (not self._is_generic_name(_current)
                                 and _current in self.verifier.embeddings)
            if (speaker_id not in self._voice_emb_saved
                    and len(buf) >= _MIN_CHUNKS_FOR_EMB
                    and not _has_existing_emb):
                self._save_live_embedding(speaker_id, _current)
                self._voice_emb_saved.add(speaker_id)

            # Face-voice binding: if ASD confirmed a face-recognized person is
            # speaking but the verifier has no voice embedding for them yet,
            # create one from this audio — face identity bootstraps voice identity.
            if asd_person_id is not None and asd_track_id is not None:
                threading.Thread(
                    target=self._try_face_voice_binding,
                    args=(asd_track_id, audio_np.copy()),
                    daemon=True,
                ).start()

            self._update_speaker_names_incremental(speaker_id, text, audio_np)

            # Contextual speaker naming: detect names from conversation and apply
            self._detect_context_names(text, speaker_id)
            self._apply_context_naming(speaker_id, text)

            # Periodic LLM analysis for remaining unidentified speakers
            self._maybe_run_llm_analysis()

            # --- Last-speaker deduction: if num_speakers is set and all but one
            #     are identified, the remaining generic speaker must be the only
            #     unused name from _context_names or _emb_to_pid.
            if self.num_speakers and len(self.identified_speakers) == self.num_speakers - 1:
                # Find all generic speaker_ids that appeared in the session
                generic_pids = set()
                for entry in self.full_transcript:
                    _epid = entry["speaker"]
                    _ename = self.speaker_names.get(_epid, "")
                    if self._is_generic_name(_ename):
                        generic_pids.add(_epid)
                # Also check active faces
                for _pid in self.shared_state.get("active_faces", {}).values():
                    _pname = self.speaker_names.get(_pid, "")
                    if self._is_generic_name(_pname):
                        generic_pids.add(_pid)
                if len(generic_pids) == 1:
                    _last_pid = generic_pids.pop()
                    # Collect all known names (identified + context)
                    _used_names = {
                        self.speaker_names.get(pid) for pid in self.identified_speakers
                        if not self._is_generic_name(self.speaker_names.get(pid, ""))
                    }
                    _candidate_names = self._context_names - _used_names
                    if len(_candidate_names) == 1:
                        _deduced = _candidate_names.pop()
                        old = self.speaker_names.get(_last_pid)
                        self.speaker_names[_last_pid] = _deduced
                        self._emb_to_pid[_deduced] = _last_pid
                        self.identified_speakers.add(_last_pid)
                        self._save_live_embedding(_last_pid, _deduced, old_name=old)
                        logger.info(f"Last-speaker deduction: {_last_pid} -> '{_deduced}' (only remaining name)")

            final_name = self.speaker_names.get(speaker_id)

            # Camera sync: when the speaker_id is an audio ID (not in active_faces),
            # propagate the identified name to the video person_id.
            # Camera sync: propagate voice name to visible generic face.
            # Only sync if the voice evidence came from VERIFIER (real biometrics),
            # NOT from ASD/SINGLE_FACE/SESSION -- those decisions already depend on the face,
            # so rewriting it from them creates error loops.
            # VERIFIER_HIGH and VERIFIER_MOD both passed the verifier threshold.
            # VERIFIER_WEAK is too uncertain to irreversibly rename a face.
            verifier_decision = decision.startswith(("VERIFIER_HIGH", "VERIFIER_MOD"))
            if final_name and not self._is_generic_name(final_name) and verifier_decision:
                video_pids = set(self.shared_state.get("active_faces", {}).values())
                if speaker_id not in video_pids:
                    # Determine the target face only via ASD (confirmed lip detection)
                    # Does not use single-face heuristic: if the speaker talks off-camera,
                    # the heuristic would rename the wrong face.
                    if asd_person_id is not None and asd_person_id in video_pids:
                        cam_target = asd_person_id
                    else:
                        cam_target = None
                    if cam_target is not None:
                        current_cam = self.speaker_names.get(cam_target, "")
                        # Extra guard: do not rename if the face tracker already confirmed another real identity
                        ft_pre = self.shared_state.get("face_tracker")
                        asd_tid_pre = None
                        for tid, pid in self.shared_state.get("active_faces", {}).items():
                            if pid == cam_target:
                                asd_tid_pre = tid
                                break
                        if ft_pre and asd_tid_pre is not None:
                            face_confirmed = ft_pre._track_to_name.get(asd_tid_pre)
                            if face_confirmed and not self._is_generic_name(face_confirmed) and face_confirmed != final_name:
                                logger.debug(f"Camera sync BLOCKED: face confirmed as '{face_confirmed}' != voice '{final_name}'")
                                cam_target = None  # cancel sync
                        # Fallback: if ASD pointed to wrong face (or no ASD), find a
                        # visible generic face matching the audio gender
                        if cam_target is None and audio_gender:
                            for _tid, _pid in self.shared_state.get("active_faces", {}).items():
                                _pname = self.speaker_names.get(_pid, "")
                                if self._is_generic_name(_pname) and _pid != speaker_id:
                                    cam_target = _pid
                                    current_cam = _pname
                                    logger.debug(f"Camera sync fallback: generic face '{_pname}' ({_pid})")
                                    break
                        if cam_target is not None and self._is_generic_name(current_cam):
                            self.speaker_names[cam_target] = final_name
                            self._emb_to_pid[final_name] = cam_target
                            ft = self.shared_state.get("face_tracker")
                            if ft and current_cam and current_cam not in ("Unknown", ""):
                                # Find the track_id of cam_target via active_faces
                                asd_track_for_rename = None
                                for tid, pid in self.shared_state.get("active_faces", {}).items():
                                    if pid == cam_target:
                                        asd_track_for_rename = tid
                                        break
                                ft.rename_person(current_cam, final_name, only_track_id=asd_track_for_rename)
                            logger.info(f"Camera synced: '{current_cam or cam_target}' -> '{final_name}'")

            # --- FALSE POSITIVE RISK DETECTION ---
            _fp_risk = "none"
            _fp_reasons = []
            _is_id = final_name is not None and not self._is_generic_name(final_name)
            if _is_id:
                # 1) Weak decision assigned a known name from stored embeddings
                _weak_decisions = {"VERIFIER_WEAK", "FACE_ONLY", "SINGLE_FACE"}
                _decision_type_pre = decision.split("(")[0].split(" ")[0].strip()
                if _decision_type_pre in _weak_decisions:
                    _fp_reasons.append(f"WEAK_DECISION:{_decision_type_pre}")
                    _fp_risk = "medium"

                # 2) Name from stored embeddings but verifier didn't confirm it
                _name_in_verifier = final_name in self.verifier.embeddings or any(
                    final_name in k for k in self.verifier.embeddings
                )
                if _name_in_verifier and raw_conf < self.verifier.threshold:
                    _fp_reasons.append(f"UNCONFIRMED_KNOWN:{final_name} raw={raw_conf:.3f}<thr={self.verifier.threshold}")
                    _fp_risk = "high"

                # 3) Gender mismatch between audio and assigned name
                if audio_gender:
                    _name_gender = _gender_of(final_name)
                    if _name_gender and _name_gender != audio_gender:
                        _fp_reasons.append(f"GENDER_MISMATCH:audio={audio_gender} name={_name_gender}({final_name})")
                        _fp_risk = "high"

                # 4) Too many unique identified speakers vs num_speakers limit
                if self.num_speakers:
                    _n_identified = len(self.identified_speakers)
                    if _n_identified > self.num_speakers:
                        _fp_reasons.append(f"SPEAKER_OVERFLOW:{_n_identified}>{self.num_speakers}")
                        if _fp_risk != "high":
                            _fp_risk = "medium"

            if _fp_reasons:
                _text_preview = text[:60] + "..." if len(text) > 60 else text
                logger.warning(f"[FP risk={_fp_risk}] {' | '.join(_fp_reasons)} | "
                               f"decision={decision} final={final_name} | \"{_text_preview}\"")

            # --- STRUCTURED METRICS ---
            _decision_type = decision.split("(")[0].split(" ")[0].strip()
            _margin_val = None
            if filtered and len(filtered) >= 2:
                _margin_val = round(filtered[0][1] - filtered[1][1], 4)
            _seg_ms = (time.perf_counter() - _t0_seg) * 1000
            self._segment_metrics.append({
                "timestamp": datetime.now().isoformat(),
                "decision": decision,
                "decision_type": _decision_type,
                "raw_best_name": raw_best_name if raw_best_name != "Unknown" else None,
                "raw_conf": round(raw_conf, 4),
                "chosen_name": real_name if real_name != "Unknown" else None,
                "chosen_conf": round(conf, 4),
                "final_name": final_name,
                "final_speaker": speaker_id,
                "is_identified": final_name is not None and not self._is_generic_name(final_name),
                "audio_gender": audio_gender,
                "n_candidates": len(all_candidates) if all_candidates else 0,
                "n_filtered": len(filtered) if filtered else 0,
                "margin": _margin_val,
                "asd_active": asd_person_id is not None,
                "n_faces": len(active_faces),
                "is_fn": bool(_fn_reasons),
                "fn_reasons": [r.split(":")[0] for r in _fn_reasons],
                "fp_risk": _fp_risk,
                "fp_reasons": _fp_reasons,
                "text_len": len(text),
                "segment_duration_s": round(len(audio_np) / 16000, 2),
                "whisper_ms": round(_whisper_ms, 1),
                "verifier_ms": round(_verifier_ms, 1),
                "segment_ms": round(_seg_ms, 1),
                "avg_logprob": round(avg_logprob, 3),
                "avg_no_speech": round(avg_no_speech, 3),
                "candidates": [(n, round(s, 4)) for n, s in (all_candidates or [])[:5]],
            })

            # Use real name if available; if generic/Unknown, show the speaker_id
            # so that utterances from distinct people are distinguishable in the transcript.
            if final_name and not self._is_generic_name(final_name):
                display_label = f"({final_name})"
            else:
                display_label = f"({speaker_id})"
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Dedup: short padded segments from the same speaker can produce identical
            # transcriptions. Skip if this text is >80% similar to a recent entry from
            # the same speaker within the last 5 seconds.
            _now = datetime.now()
            _norm = lambda t: re.sub(r'[^\w\s]', '', t.lower()).strip()
            _is_dup = False
            for _prev in reversed(self.full_transcript[-6:]):
                # Cross-speaker dedup: padded segments produce the same text regardless
                # of which speaker_id was assigned — check any recent entry within 5s.
                if (_now - _prev["timestamp"]).total_seconds() <= 5.0:
                    _ratio = SequenceMatcher(None, _norm(text), _norm(_prev["text"])).ratio()
                    if _ratio >= 0.80:
                        logger.debug(f"Dedup: skipped near-duplicate ({_ratio:.2f}): '{text[:50]}'")
                        _is_dup = True
                        break
            if not _is_dup:
                print(f"\n[{timestamp}] {display_label}: {text}")
                self.full_transcript.append({"speaker": speaker_id, "verified_name": final_name, "text": text, "timestamp": _now})
        except Exception as e:
            print(f"Error processing segment: {e}")
        finally:
            self._log_perf("segment_total", (time.perf_counter() - _t0_seg) * 1000)

    def _try_face_voice_binding(self, asd_track_id, audio_np):
        """Face-voice binding: when ASD confirms a face-recognized person is speaking,
        create a voice embedding so face identity bootstraps voice identity."""
        try:
            face_tracker = self.shared_state.get("face_tracker")
            if face_tracker is None:
                return

            # 1. Get the face-recognized name for this track
            face_name = face_tracker._track_to_name.get(asd_track_id)
            if not face_name or self._is_generic_name(face_name):
                return  # face not recognized as a known person

            # 2. Voice embedding already exists? Nothing to do.
            if face_name in self.verifier.embeddings:
                return

            # 3. Cooldown (120s per name)
            now = time.time()
            if now - self._fv_bind_last.get(face_name, 0.0) < 120.0:
                return
            self._fv_bind_last[face_name] = now

            # 4. Audio quality gate
            if len(audio_np) < 3 * self.sample_rate:
                return
            rms = float(np.sqrt(np.mean(audio_np ** 2)))
            if rms < 1e-4:
                return  # near-silent

            # 5. Gender sanity check
            audio_gender = self._detect_gender_from_audio(audio_np)
            if audio_gender:
                name_gender = _gender_of(face_name)
                if name_gender and name_gender != audio_gender:
                    logger.debug(f"[FaceVoiceBind] Gender mismatch: audio={audio_gender} face={face_name}({name_gender})")
                    return

            # 6. Save voice embedding
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            emb_dir = os.path.join(base_dir, "data", "embeddings")

            mx = np.abs(audio_np).max()
            if mx > 0:
                audio_np = audio_np / mx
            signal = torch.from_numpy(audio_np).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                embedding = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
            norm_val = np.linalg.norm(embedding)
            if norm_val > 0:
                embedding = embedding / norm_val

            os.makedirs(emb_dir, exist_ok=True)
            filename = f"{face_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fvbind.npy"
            np.save(os.path.join(emb_dir, filename), embedding)

            # Register in verifier memory immediately
            emb_tensor = torch.from_numpy(embedding).float().to(self.verifier.device)
            self.verifier.embeddings[face_name] = emb_tensor
            self.verifier._raw_embeddings.setdefault(face_name, []).append(emb_tensor)

            # Register in shared_state registry
            if face_name not in self._emb_to_pid:
                for tid, pid in self.shared_state.get("active_faces", {}).items():
                    if face_tracker._track_to_name.get(tid) == face_name:
                        self._emb_to_pid[face_name] = pid
                        self.speaker_names[pid] = face_name
                        self.identified_speakers.add(pid)
                        break

            # Sync to database
            try:
                from src.database import get_db
                db = get_db()
                spk_id = db.add_speaker(face_name)
                db.save_voice_embedding(spk_id, embedding, source_file=filename)
            except Exception as db_err:
                logger.warning(f"[FaceVoiceBind] DB sync failed: {db_err}")

            logger.info(f"[FaceVoiceBind] Voice embedding created for '{face_name}' (face+ASD): {filename}")
        except Exception as e:
            logger.error(f"[FaceVoiceBind] Failed for track {asd_track_id}: {e}")

    def _try_auto_enroll(self, name: str, audio_np):
        """Try to save an automatic embedding for a speaker identified with high confidence."""
        try:
            # Strip homonym suffix ("Arthur 2" → "Arthur") — save under the base name
            name = re.sub(r'\s+\d+$', '', name)
            saved = self.verifier.auto_enroll(name, audio_np)
            if saved:
                logger.info(f"Auto-enroll: new embedding saved for '{name}'")
        except Exception as e:
            logger.debug(f"Auto-enroll failed ({name}): {e}")

    def _voice_embedding(self, audio_np):
        """Normalized ECAPA voice embedding for a short segment; None if too short/failed."""
        if audio_np is None or len(audio_np) < int(16000 * 0.4):
            return None
        try:
            signal = torch.from_numpy(np.ascontiguousarray(audio_np)).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                emb = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else None
        except Exception:
            return None

    def _get_or_create_session_speaker_id(self, audio_np, exclude_ids=None, audio_gender=None):
        """Return stable session ID for unknown speaker using voice similarity.

        audio_gender: 'male'/'female'/None — prevents cross-gender merges so that
        e.g. a female speaker is never merged with a male session bucket.
        """
        exclude_ids = exclude_ids or set()
        signal = torch.from_numpy(audio_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Find the closest speaker — respecting gender constraint
        best_id, best_sim = None, -1.0
        for sid, known_emb in self._session_unknown_embs.items():
            if sid in exclude_ids:
                continue
            # Skip if gender is known for both and they differ
            known_gender = self._session_gender.get(sid)
            if audio_gender and known_gender and audio_gender != known_gender:
                continue
            sim = float(np.dot(emb, known_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = sid

        # Threshold: gender-aware.
        # Cross-gender merges are blocked by the constraint above, so within-gender
        # merges can use a low threshold (0.35) — recording variation across chunks
        # causes same-person embeddings to land at 0.40-0.60.
        # When gender is unknown for either side, use a moderate threshold (0.45).
        _same_gender = (audio_gender and self._session_gender.get(best_id) == audio_gender)
        _threshold = 0.35 if _same_gender else 0.45
        if best_id and best_sim >= _threshold:
            alpha = 0.1
            self._session_unknown_embs[best_id] = (1 - alpha) * self._session_unknown_embs[best_id] + alpha * emb
            if audio_gender and best_id not in self._session_gender:
                self._session_gender[best_id] = audio_gender
            return best_id

        # Respect the speaker limit — forced merge only if minimally plausible (>= 0.20)
        all_tracked = set(self.speaker_names.keys()) | set(self._session_unknown_embs.keys())
        if self.num_speakers and len(all_tracked) >= self.num_speakers:
            if best_id and best_sim >= 0.20:
                return best_id

        new_id = self._new_person_id()
        self._session_unknown_embs[new_id] = emb
        if audio_gender:
            self._session_gender[new_id] = audio_gender
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
        """Create a stable opaque person_id, using the shared_state factory if available."""
        factory = self.shared_state.get("person_id_factory")
        if factory:
            return factory()
        self._session_unknown_counter += 1
        return f"spk_{self._session_unknown_counter:03d}"

    def _is_generic_name(self, name):
        """Return True if the name/ID is generic (spk_001, Person_1, Desconhecido_1, Unknown...)."""
        return name is not None and bool(re.match(r'^(spk_\d+|Person_\d+|Desconhecido_\d+|Unknown(_\d+)?)$', name))

    def _detect_context_names(self, text, current_speaker_id):
        """Detect names from conversational context: vocatives, references, introductions.

        - Vocative (addressing someone): "Arthur, pode falar" → next speaker is Arthur
        - Invitation: "Fala, Kauan" / "Sua vez, Manu" → next speaker is that name
        - Thanks/goodbye: "Obrigado, Arthur" → Arthur was the previous or current speaker
        - Third-person reference: "a parte do Kauan" → Kauan is a participant

        Returns: list of (name, role) where role is "addressee" or "mentioned"
        """
        results = []
        # Words that should never be treated as person names in vocative context
        _blocklist = getattr(self, '_vocative_blocklist', None)
        if _blocklist is None:
            self._vocative_blocklist = {
                'pessoal', 'gente', 'galera', 'turma', 'cara', 'mano', 'brother',
                'professor', 'professora', 'doutor', 'doutora',
                'obrigado', 'obrigada', 'desculpa', 'tchau', 'oi', 'olá',
                'sim', 'não', 'bom', 'boa', 'tudo', 'certo', 'pronto',
                'então', 'agora', 'aqui', 'assim', 'tipo', 'enfim',
                # Common PT-BR words that Whisper capitalizes at sentence start
                'claro', 'ou', 'mas', 'porque', 'porém', 'pois', 'logo',
                'talvez', 'nunca', 'sempre', 'ainda', 'também', 'aliás',
                'legal', 'verdade', 'exato', 'beleza', 'tranquilo', 'show',
                'viado', 'meu', 'minha', 'nosso', 'nossa', 'dele', 'dela',
                'isso', 'esse', 'essa', 'aquele', 'aquela', 'qual', 'quem',
                'onde', 'como', 'quando', 'quanto', 'vamos', 'bora',
                'hein', 'né', 'pô', 'putz', 'caramba', 'caraca',
                'maravilha', 'perfeito', 'exatamente', 'simplesmente',
                # English discourse markers / fillers that Whisper capitalizes
                # at sentence start — never person names.
                'yeah', 'yep', 'yes', 'no', 'nope', 'ok', 'okay', 'so', 'well',
                'right', 'now', 'look', 'hey', 'hi', 'hello', 'um', 'uh', 'hmm',
                'oh', 'ah', 'anyway', 'alright', 'actually', 'basically', 'like',
                'sure', 'maybe', 'please', 'thanks', 'thank', 'sorry', 'exactly',
                'totally', 'honestly', 'obviously', 'listen', 'wait', 'and', 'but',
                'or', 'because', 'then', 'also', 'just', 'really', 'here', 'there',
                'this', 'that', 'what', 'who', 'when', 'where', 'why', 'how',
                'which', 'guys', 'everyone', 'folks', 'man', 'dude', 'mean',
            }
            _blocklist = self._vocative_blocklist

        def _valid_name(n):
            return (n and len(n) >= 2 and n[0].isupper()
                    and n.lower() not in _blocklist
                    and not re.match(r'^(Person_\d+|spk_\d+)$', n))

        # spaCy NER on the full sentence — collect words/entities tagged PERSON.
        # The "word adjacent to a comma" rules below are weak ("Yeah, so I...") so
        # the candidate must also be confirmed as a real person name by NER.
        _ner_entities = self._extract_names_with_ner(text)
        _ner_set = set()
        for _e in _ner_entities:
            _ner_set.add(_e.lower())
            _ner_set.update(_e.lower().split())

        def _is_person(n):
            # NER must confirm the candidate. When NER is unavailable, don't
            # block — fall back to the blocklist alone.
            if self.nlp is None:
                return True
            nl = n.lower()
            return nl in _ner_set or any(w in _ner_set for w in nl.split())

        # 1) Vocative at start: "Arthur, pode falar" / "Kauan, o que acha?"
        #    NER-gated — rejects "Yeah,", "So,", "Well," and other discourse markers.
        m = re.match(r'^([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)\s*,', text)
        if m and _valid_name(m.group(1)) and _is_person(m.group(1)):
            results.append((m.group(1).title(), "addressee"))

        # 2) Vocative at end: "Pode falar, Arthur" / "Né, Kauan?"
        m = re.search(r',\s*([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)\s*[?.!]?\s*$', text)
        if m and _valid_name(m.group(1)) and _is_person(m.group(1)):
            results.append((m.group(1).title(), "addressee"))

        # 3) Invitation to speak: "Fala, Arthur" / "Vai lá, Manu" / "Pode falar, Kauan"
        for p in [r'(?:fala|vai|pode falar|sua vez|manda)\s*,?\s*([A-ZÀ-Ú][a-zà-ú]+)',
                  r'(?:obrigad[oa]|valeu)\s*,?\s*([A-ZÀ-Ú][a-zà-ú]+)']:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                raw = m.group(1)
                if raw[0].isupper() and _valid_name(raw):
                    results.append((raw.title(), "addressee"))

        # 4) Third-person references: "a parte do Kauan" / "como o Arthur disse"
        for p in [r'(?:d[oa]|com o|com a|que o|que a)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)',
                  r'(?:parte|vez|turno|projeto|trabalho)\s+d[oa]\s+([A-ZÀ-Ú][a-zà-ú]+)']:
            for mm in re.finditer(p, text):
                raw = mm.group(1)
                if _valid_name(raw):
                    results.append((raw.title(), "mentioned"))

        # 5) NER fallback: any PERSON entity not already found
        found_names = {r[0].lower() for r in results}
        for n in _ner_entities:
            if n.lower() not in found_names and _valid_name(n):
                results.append((n.title(), "mentioned"))

        # Register all found names in context pool
        for name, role in results:
            self._context_names.add(name)

        # If addressee found, set pending for next speaker
        for name, role in results:
            if role == "addressee":
                self._pending_addressee = {
                    "name": name,
                    "from_speaker": current_speaker_id,
                    "ts": time.time(),
                }
                logger.debug(f"Context: addressee detected '{name}' (from {current_speaker_id})")
                break  # only one addressee per segment

        return results

    def _apply_context_naming(self, speaker_id, text):
        """Try to name an unidentified speaker using contextual evidence.
        Called after _update_speaker_names_incremental (which handles self-introductions).
        """
        current_name = self.speaker_names.get(speaker_id)
        if current_name and not self._is_generic_name(current_name):
            return  # already has a real name

        # 1) Check pending addressee from previous turn
        pa = self._pending_addressee
        if pa and pa["from_speaker"] != speaker_id and (time.time() - pa["ts"]) < 30:
            name = pa["name"]
            # Don't assign if this name is already taken by another speaker
            if name not in self._emb_to_pid or self._emb_to_pid[name] == speaker_id:
                self._addressee_votes[speaker_id][name] += 2  # strong signal
                logger.debug(f"Context: +2 vote '{name}' for {speaker_id} (addressee)")

        # 2) Check accumulated votes — assign if confident (>= 3 votes)
        votes = self._addressee_votes.get(speaker_id, {})
        if votes:
            best_name = max(votes, key=votes.get)
            best_count = votes[best_name]
            if best_count >= 3:
                # Verify name isn't taken
                if best_name not in self._emb_to_pid or self._emb_to_pid[best_name] == speaker_id:
                    speakers_full = self.num_speakers and len(self.identified_speakers) >= self.num_speakers
                    if not speakers_full:
                        old_name = current_name
                        self.speaker_names[speaker_id] = best_name
                        self._emb_to_pid[best_name] = speaker_id
                        self.identified_speakers.add(speaker_id)
                        self._save_live_embedding(speaker_id, best_name, old_name=old_name)
                        logger.info(f"Context naming: {speaker_id} -> '{best_name}' ({best_count} votes)")
                        # Clear votes after assignment
                        self._addressee_votes.pop(speaker_id, None)

        # Clear expired pending addressee
        if pa and (time.time() - pa["ts"]) > 30:
            self._pending_addressee = None

    def _llm_identify_speakers(self):
        """Use an LLM to analyze the full transcript and identify unnamed speakers.

        Runs in a background thread every _llm_analysis_interval segments.
        Sends the accumulated transcript to an LLM, which returns a JSON mapping
        of generic IDs (spk_016, Person_1) to real names inferred from context.
        """
        if not self.use_ai_analysis:
            return
        # Build transcript
        generic_speakers = set()
        lines = []
        for entry in self.full_transcript[-50:]:  # last 50 entries max (context window)
            name = entry.get('verified_name') or entry['speaker']
            lines.append(f"[{name}]: {entry['text']}")
            if self._is_generic_name(name):
                generic_speakers.add(name)

        if not generic_speakers or not lines:
            return  # nothing to identify

        already_identified = {
            pid: name for pid, name in self.speaker_names.items()
            if not self._is_generic_name(name)
        }
        context_names = self._context_names - set(already_identified.values())

        transcript_text = "\n".join(lines[-40:])  # trim to last 40 lines
        num_spk = self.num_speakers or "desconhecido"

        prompt = (
            f"Analise esta transcrição de uma reunião com {num_spk} participantes.\n"
            f"Alguns falantes já foram identificados: {dict(list(already_identified.items())[:8])}\n"
            f"Nomes mencionados na conversa: {', '.join(context_names) if context_names else 'nenhum'}\n"
            f"Falantes genéricos (não identificados): {', '.join(sorted(generic_speakers))}\n\n"
            f"Transcrição:\n{transcript_text}\n\n"
            f"Baseado no contexto da conversa (quem fala com quem, referências a outros, "
            f"tópicos discutidos, gênero gramatical), identifique o nome real de cada "
            f"falante genérico.\n\n"
            f"Responda APENAS com um JSON mapeando IDs genéricos para nomes reais. "
            f"Exemplo: {{\"spk_016\": \"Arthur\", \"Person_1\": \"Kauan\"}}\n"
            f"Só inclua mapeamentos que você tem CERTEZA. Se não tem certeza, omita."
        )

        try:
            if self._llm_client is None:
                from huggingface_hub import InferenceClient
                self._llm_client = InferenceClient(token=HF_TOKEN)

            response = self._llm_client.text_generation(
                prompt,
                model="mistralai/Mistral-7B-Instruct-v0.3",
                max_new_tokens=150,
                temperature=0.1,
            )

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                import json
                mapping = json.loads(json_match.group())
                logger.info(f"LLM speaker mapping: {mapping}")

                for generic_id, real_name in mapping.items():
                    if not isinstance(real_name, str) or len(real_name) < 2:
                        continue
                    real_name = real_name.strip().title()
                    # Find the speaker_id for this generic label
                    target_pid = None
                    for pid, name in self.speaker_names.items():
                        if name == generic_id or pid == generic_id:
                            target_pid = pid
                            break
                    if target_pid is None:
                        continue
                    # Don't overwrite existing real names
                    current = self.speaker_names.get(target_pid)
                    if current and not self._is_generic_name(current):
                        continue
                    # Don't assign if name is already taken
                    if real_name in self._emb_to_pid and self._emb_to_pid[real_name] != target_pid:
                        continue

                    speakers_full = self.num_speakers and len(self.identified_speakers) >= self.num_speakers
                    if not speakers_full:
                        old_name = current
                        self.speaker_names[target_pid] = real_name
                        self._emb_to_pid[real_name] = target_pid
                        self.identified_speakers.add(target_pid)
                        self._save_live_embedding(target_pid, real_name, old_name=old_name)
                        logger.info(f"LLM naming: {target_pid} ({generic_id}) -> '{real_name}'")
            else:
                logger.debug(f"LLM response (no JSON found): {response[:200]}")
        except Exception as e:
            logger.warning(f"LLM speaker analysis failed: {e}")

    def _maybe_run_llm_analysis(self):
        """Trigger LLM analysis every N segments, in a background thread."""
        seg_count = len(self._segment_metrics)
        if seg_count - self._llm_last_analysis >= self._llm_analysis_interval:
            self._llm_last_analysis = seg_count
            # Check if there are any generic speakers worth analyzing
            has_generic = any(
                self._is_generic_name(self.speaker_names.get(pid, ""))
                for pid in set(e["speaker"] for e in self.full_transcript[-20:])
            )
            if has_generic:
                threading.Thread(target=self._llm_identify_speakers, daemon=True).start()

    def _extract_names_with_ner(self, text):
        """Extract person names from text using spaCy NER. Strips honorific prefixes."""
        if not self.nlp: return []
        try:
            doc = self.nlp(text)
            # pt models label persons "PER"; en models label them "PERSON"
            return [re.sub(r'^(Sr\.|Sra\.|Dr\.|Dra\.|Mr\.|Mrs\.|Ms\.)\s+', '', ent.text.strip(), flags=re.IGNORECASE)
                    for ent in doc.ents
                    if ent.label_ in ("PER", "PERSON") and len(ent.text.strip()) > 2]
        except Exception: return []

    def _detect_gender_from_text(self, text):
        """Detect speaker gender from Portuguese text patterns and NER-extracted names."""
        text_lower = text.lower()
        extracted_names = self._extract_names_with_ner(text)
        for name in extracted_names:
            name_key = _normalize_name_token(name)
            if name_key in NAMES_FEMALE_NORMALIZED:
                return "female"
            if name_key in NAMES_MALE_NORMALIZED:
                return "male"
        # Portuguese gender markers in speech (kept as-is -- they match transcribed PT-BR text)
        # Female: "eu sou a" (I am the [fem]), "sou a", "sou mulher" (I am a woman),
        #         "meu nome e [name ending in -a]" (my name is ...)
        female_markers = [r'\beu sou a\b', r'\bsou a\b', r'\bsou mulher', r'\bmeu nome é\s+([a-zà-ú]+a)\b']
        # Male: "eu sou o" (I am the [masc]), "sou o", "sou homem" (I am a man),
        #       "meu nome e [name ending in -o]" (my name is ...)
        male_markers = [r'\beu sou o\b', r'\bsou o\b', r'\bsou homem', r'\bmeu nome é\s+([a-zà-ú]+o)\b']
        f_score = sum(1 for p in female_markers if re.search(p, text_lower))
        m_score = sum(1 for p in male_markers if re.search(p, text_lower))
        if f_score > m_score: return "female"
        if m_score > f_score: return "male"
        return None

    def _detect_gender_from_audio(self, audio_chunk):
        """Detect speaker gender from fundamental frequency (pitch) of the audio."""
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
        """Correct the first name's gender suffix based on detected audio/text gender."""
        if not gender: return name
        # Only correct the FIRST name -- surnames (e.g. Otavio, Marques) should not be changed
        parts = name.split()
        first = _normalize_name_token(parts[0]) if parts else ""
        if gender == "female":
            if first in MALE_TO_FEMALE:
                parts[0] = MALE_TO_FEMALE[first].capitalize()
            elif first.endswith('o') and first not in NAMES_FEMALE_NORMALIZED:
                parts[0] = (first[:-1] + 'a').capitalize()
        return ' '.join(parts)

    def _update_speaker_names_incremental(self, speaker_id, text, audio_chunk=None):
        """Extract speaker name from transcribed text and update speaker_names if found."""
        current_name = self.speaker_names.get(speaker_id)
        is_rename = self._is_generic_name(current_name)
        # Already has a real name -- do not overwrite
        if current_name and not is_rename:
            return
        # Only block by speaker limit when adding a new speaker (not when renaming)
        if not is_rename and self.num_speakers and len(self.identified_speakers) >= self.num_speakers:
            return
        t_gender = self._detect_gender_from_text(text)
        a_gender = self._detect_gender_from_audio(audio_chunk) if audio_chunk is not None else None
        gender = t_gender or a_gender
        # Portuguese self-introduction patterns (kept as-is -- they match transcribed PT-BR speech):
        #   "meu nome e" = "my name is", "me chamo" = "I'm called", "eu sou" = "I am",
        #   "sou o/sou a" = "I'm the [masc/fem]", "aqui e" = "this is",
        #   "aqui quem fala e" = "this is who's speaking"
        # Words that match "eu sou X" but are NOT names
        _not_names = {
            'lésbica', 'lésbico', 'gay', 'trans', 'travesti', 'hétero', 'hetero',
            'bissexual', 'homem', 'mulher', 'pessoa', 'alguém', 'ninguém',
            'professor', 'professora', 'doutor', 'doutora', 'engenheiro', 'engenheira',
            'advogado', 'advogada', 'médico', 'médica', 'estudante', 'aluno', 'aluna',
            'brasileiro', 'brasileira', 'casado', 'casada', 'solteiro', 'solteira',
            'cristão', 'cristã', 'ateu', 'ateia', 'católico', 'católica',
            'tímido', 'tímida', 'ansioso', 'ansiosa', 'feliz', 'triste',
            'novo', 'nova', 'velho', 'velha', 'gordo', 'gorda', 'magro', 'magra',
            # Common words Whisper capitalizes / NER misidentifies
            'claro', 'ou', 'mas', 'porque', 'porém', 'pois', 'logo',
            'talvez', 'nunca', 'sempre', 'ainda', 'também', 'aliás',
            'legal', 'verdade', 'exato', 'beleza', 'tranquilo', 'show',
            'viado', 'meu', 'minha', 'nosso', 'nossa', 'simplesmente',
            'maravilha', 'perfeito', 'exatamente', 'piloto', 'obrigado',
        }
        patterns = [r'(?:meu nome é|me chamo|eu sou|sou o|sou a)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)', r'(?:aqui é|aqui quem fala é)\s+(?:o|a)?\s*([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)?)']
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                raw = m.group(1)
                # Reject: must start uppercase in original text AND not be a common word
                if not raw[0].isupper() or raw.lower().split()[0] in _not_names:
                    continue
                name = self._correct_name_for_gender(raw.title(), gender)
                # If the name already exists (e.g. "Jose Marques" loaded from disk), assume
                # it is the same person re-introducing themselves. Real homonyms are rare
                # in small meetings -- prefer merge over suffixing "Jose Marques 2".
                display_name = name
                if display_name in self._emb_to_pid and self._emb_to_pid[display_name] != speaker_id:
                    # Merge: point the name to the current session speaker_id
                    self._emb_to_pid[display_name] = speaker_id
                name = display_name
                old_name = current_name  # previous display name (or None)
                self.speaker_names[speaker_id] = name
                self._emb_to_pid[name] = speaker_id  # register for future verifier matches
                self.identified_speakers.add(speaker_id)  # track by person_id
                self._save_live_embedding(speaker_id, name, old_name=old_name)
                return

    def _save_live_embedding(self, speaker_id, name, old_name=None):
        """Save a voice embedding from accumulated audio for a newly identified speaker."""
        audio_chunks = self.unknown_speakers_audio.get(speaker_id, [])
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        emb_dir = os.path.join(base_dir, "data", "embeddings")

        face_tracker = self.shared_state.get("face_tracker")

        # Rename existing embeddings from the generic name to the real name
        if old_name and self._is_generic_name(old_name):
            self.verifier.rename_embedding(old_name, name)
            if face_tracker:
                # If old_name is "Unknown", the face may already be enrolled under another name
                # (e.g. "Person_1") -- look up the actual enrolled name in _track_to_name
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
            logger.info(f"Identity updated: '{old_name}' -> '{name}'")

        # If came from session tracking, try to bind to a visible generic face
        # Skip when name itself is generic (auto-save) — don't rename faces to spk_001
        elif old_name is None and face_tracker and not self._is_generic_name(name):
            active_faces = self.shared_state.get("active_faces", {})
            # active_faces = {track_id: person_id} -- find visible generic person_ids
            generic_pids = [
                pid for pid in set(active_faces.values())
                if pid != speaker_id and self._is_generic_name(self.speaker_names.get(pid, ""))
            ]
            if len(generic_pids) == 1:
                target_pid = generic_pids[0]
                old_face_name = self.speaker_names.get(target_pid, "")
                if old_face_name:
                    # Find the track_id associated with target_pid
                    target_track = None
                    for tid, pid in active_faces.items():
                        if pid == target_pid:
                            target_track = tid
                            break
                    face_tracker.rename_person(old_face_name, name, only_track_id=target_track)
                    self.speaker_names[target_pid] = name
                    self._emb_to_pid[name] = target_pid
                    logger.info(f"Camera synced: '{old_face_name}' -> '{name}'")

        if not audio_chunks:
            return
        norm_name = name.lower().strip()
        # Skip if an embedding already exists for this name -- EXCEPT when we just renamed
        # from a generic name (the current audio is more recent and should be saved)
        renamed_from_generic = old_name and self._is_generic_name(old_name)
        if not renamed_from_generic and os.path.exists(emb_dir):
            for f in os.listdir(emb_dir):
                fl = f.lower()
                # Require exact name match (followed by '_' or '.npy'),
                # to avoid confusing "Joao" with "Joao 2" etc.
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
            logger.info(f"New voice biometric saved for {name}.")
            self.verifier.load_embeddings(emb_dir)
            # Sync to database
            try:
                from src.database import get_db
                db = get_db()
                spk_id = db.add_speaker(name)
                db.save_voice_embedding(spk_id, embedding, source_file=filename)
            except Exception as db_err:
                logger.warning(f"DB sync failed for voice embedding: {db_err}")
        except Exception as e:
            logger.error(f"Error saving voice biometric: {e}")

    def prompt_and_save_unknown_speakers(self, min_audio_s: float = 5.0):
        """
        For each anonymous speaker with enough accumulated audio, interactively
        prompt the user for a name and save the voice embedding.
        Called when closing the session.
        """
        candidates = []
        for spk_id, chunks in self.unknown_speakers_audio.items():
            if not chunks:
                continue
            current_name = self.speaker_names.get(spk_id, spk_id)
            if not self._is_generic_name(current_name):
                continue  # already has a real name -- was identified during the session
            total_s = sum(len(c) for c in chunks) / self.sample_rate
            if total_s < min_audio_s:
                continue
            candidates.append((spk_id, current_name, total_s))

        if not candidates:
            return

        print("\n" + "="*60)
        print("UNIDENTIFIED SPEAKERS -- POST-SESSION ENROLLMENT")
        print("="*60)
        print(f"  {len(candidates)} anonymous speaker(s) with enough audio.\n")

        for spk_id, generic_name, total_s in candidates:
            print(f"  -- {generic_name}  ({total_s:.0f}s of audio) --")

            # Play up to 10s of audio so the user can hear and identify the voice
            chunks = self.unknown_speakers_audio[spk_id]
            preview = np.concatenate(chunks)
            preview = preview[:self.sample_rate * 10]  # max 10s
            mx = np.abs(preview).max()
            if mx > 0:
                preview = preview / mx * 0.8
            print(f"  Playing {len(preview)/self.sample_rate:.1f}s clip... ", end="", flush=True)
            try:
                if AUDIO_BACKEND == "sounddevice":
                    import sounddevice as _sd
                    _sd.play(preview, samplerate=self.sample_rate)
                    _sd.wait()
                elif AUDIO_BACKEND == "pyaudio":
                    import pyaudio as _pa
                    pa = _pa.PyAudio()
                    stream = pa.open(format=_pa.paFloat32, channels=1,
                                     rate=self.sample_rate, output=True)
                    stream.write(preview.tobytes())
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()
                print("OK")
            except Exception as e:
                print(f"(playback error: {e})")

            while True:
                name = input(f"\n  Name for this speaker (Enter=skip, r=replay, 'mixed'=multiple speakers): ").strip()
                if name.lower() == 'r':
                    print(f"  Replaying {len(preview)/self.sample_rate:.1f}s clip... ", end="", flush=True)
                    try:
                        if AUDIO_BACKEND == "sounddevice":
                            import sounddevice as _sd
                            _sd.play(preview, samplerate=self.sample_rate)
                            _sd.wait()
                        elif AUDIO_BACKEND == "pyaudio":
                            import pyaudio as _pa
                            pa = _pa.PyAudio()
                            stream = pa.open(format=_pa.paFloat32, channels=1,
                                             rate=self.sample_rate, output=True)
                            stream.write(preview.tobytes())
                            stream.stop_stream()
                            stream.close()
                            pa.terminate()
                        print("OK")
                    except Exception as e:
                        print(f"(playback error: {e})")
                    continue
                break
            if not name:
                print("  Skipped.\n")
                continue
            if name.lower() == 'mixed':
                print("  Marked as mixed audio (multiple speakers). No embedding saved.\n")
                continue

            self._save_live_embedding(spk_id, name, old_name=generic_name)
            # Update name in the current session as well
            self.speaker_names[spk_id] = name
            self._emb_to_pid[name] = spk_id
            print(f"  Embedding saved for '{name}'.\n")

        print("="*60 + "\n")

    def save_session(self, output_dir="realtime_sessions"):
        """Save the current session transcript and audio to disk."""
        self._print_perf_summary()
        self._print_fn_summary()
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = os.path.join(output_dir, f"transcript_{ts}.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            for entry in self.full_transcript:
                f.write(f"[{entry['timestamp'].strftime('%H:%M:%S')}] {entry.get('verified_name') or entry['speaker']}: {entry['text']}\n")

        # Save FN report to file alongside the transcript
        if self._fn_events:
            fn_file = os.path.join(output_dir, f"fn_report_{ts}.txt")
            with open(fn_file, "w", encoding="utf-8") as f:
                f.write(f"FALSE NEGATIVE REPORT — {len(self._fn_events)} event(s)\n")
                f.write(f"{'='*60}\n\n")
                for i, ev in enumerate(self._fn_events):
                    f.write(f"[{ev['timestamp']}] {' | '.join(ev['reasons'])}\n")
                    f.write(f"  decision={ev['decision']} final={ev['final']} "
                            f"raw_best={ev['raw_best']} gender={ev['audio_gender']} faces={ev['faces']}\n")
                    f.write(f"  text=\"{ev['text']}\"\n")
                    if ev["all_candidates"]:
                        cands = " | ".join(f"{n}:{s:.3f}" for n, s in ev["all_candidates"])
                        f.write(f"  candidates=[{cands}]\n")
                    f.write("\n")
            logger.info(f"FN report saved: {fn_file}")

        # Save FP risk report to file alongside the transcript
        _fp_segments = [m for m in self._segment_metrics if m.get("fp_risk") in ("high", "medium")]
        if _fp_segments:
            fp_file = os.path.join(output_dir, f"fp_report_{ts}.txt")
            with open(fp_file, "w", encoding="utf-8") as f:
                n_high = sum(1 for m in _fp_segments if m["fp_risk"] == "high")
                n_med = sum(1 for m in _fp_segments if m["fp_risk"] == "medium")
                f.write(f"FALSE POSITIVE RISK REPORT — {len(_fp_segments)} suspect segment(s)\n")
                f.write(f"  HIGH risk: {n_high}  |  MEDIUM risk: {n_med}\n")
                f.write(f"{'='*60}\n\n")
                for m in _fp_segments:
                    f.write(f"[{m['timestamp']}] risk={m['fp_risk'].upper()}\n")
                    f.write(f"  decision={m['decision']}  final={m['final_name']}  speaker={m['final_speaker']}\n")
                    f.write(f"  raw_best={m['raw_best_name']}:{m['raw_conf']:.3f}  gender={m['audio_gender']}\n")
                    for r in m.get("fp_reasons", []):
                        f.write(f"  reason: {r}\n")
                    if m.get("candidates"):
                        cands = " | ".join(f"{n}:{s:.3f}" for n, s in m["candidates"])
                        f.write(f"  candidates=[{cands}]\n")
                    f.write("\n")
            logger.info(f"FP report saved: {fp_file}")

        # Save structured metrics as JSON for analysis / graphing
        if self._segment_metrics:
            decision_counts = defaultdict(int)
            for m in self._segment_metrics:
                decision_counts[m["decision_type"]] += 1
            raw_confs = [m["raw_conf"] for m in self._segment_metrics if m["raw_conf"] > 0]
            # Collect face metrics from the tracker if available
            face_metrics = None
            ft = self.shared_state.get("face_tracker")
            if ft and hasattr(ft, "get_session_face_metrics"):
                try:
                    face_metrics = ft.get_session_face_metrics()
                except Exception as e:
                    logger.warning(f"Failed to collect face metrics: {e}")

            metrics_payload = {
                "session_id": ts,
                "session_start": self._session_start.isoformat(),
                "config": {
                    "verifier_threshold": self.verifier.threshold,
                    "verifier_confidence_min": self.verifier_confidence_min,
                    "num_speakers": self.num_speakers,
                    "whisper_size": self._whisper_size,
                    "sample_rate": self.sample_rate,
                    "face_match_threshold": face_metrics["match_threshold"] if face_metrics else None,
                    "face_merge_threshold": face_metrics["merge_threshold"] if face_metrics else None,
                },
                "summary": {
                    "total_segments": len(self._segment_metrics),
                    "total_fn": sum(1 for m in self._segment_metrics if m["is_fn"]),
                    "total_identified": sum(1 for m in self._segment_metrics if m["is_identified"]),
                    "unique_speakers": len(set(m["final_speaker"] for m in self._segment_metrics)),
                    "decision_counts": dict(decision_counts),
                    "fn_rate": round(sum(1 for m in self._segment_metrics if m["is_fn"]) / len(self._segment_metrics), 4) if self._segment_metrics else 0,
                    "identification_rate": round(sum(1 for m in self._segment_metrics if m["is_identified"]) / len(self._segment_metrics), 4) if self._segment_metrics else 0,
                    "avg_raw_conf": round(float(np.mean(raw_confs)), 4) if raw_confs else 0,
                    "median_raw_conf": round(float(np.median(raw_confs)), 4) if raw_confs else 0,
                    "avg_whisper_ms": round(float(np.mean([m["whisper_ms"] for m in self._segment_metrics])), 1),
                    "avg_verifier_ms": round(float(np.mean([m["verifier_ms"] for m in self._segment_metrics])), 1),
                    "avg_segment_ms": round(float(np.mean([m["segment_ms"] for m in self._segment_metrics])), 1),
                    "total_fp_high": sum(1 for m in self._segment_metrics if m.get("fp_risk") == "high"),
                    "total_fp_medium": sum(1 for m in self._segment_metrics if m.get("fp_risk") == "medium"),
                    "fp_rate": round(sum(1 for m in self._segment_metrics if m.get("fp_risk") in ("high", "medium")) / len(self._segment_metrics), 4) if self._segment_metrics else 0,
                },
                "segments": self._segment_metrics,
            }
            if face_metrics:
                metrics_payload["face"] = {
                    "total_tracks": face_metrics["total_tracks"],
                    "tracks_identified": face_metrics["tracks_identified"],
                    "tracks_generic": face_metrics["tracks_generic"],
                    "face_fn_count": face_metrics["face_fn_count"],
                    "face_fn_events": face_metrics["face_fn_events"],
                    "enrollments": face_metrics["enrollments"],
                    "known_embeddings_count": face_metrics["known_embeddings_count"],
                    "total_frames": face_metrics["total_frames"],
                    "match_score_stats": face_metrics["match_score_stats"],
                    "match_scores_sample": face_metrics["match_scores_sample"],
                }
            metrics_file = os.path.join(output_dir, f"metrics_{ts}.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Metrics saved: {metrics_file}")

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
            logger.info(f"Audio saved: {audio_file}")

        # --- Save to SQLite database ---
        try:
            from src.database import get_db
            db = get_db()

            # Read full transcript text
            transcript_text = None
            if os.path.exists(transcript_file):
                with open(transcript_file, encoding="utf-8") as f:
                    transcript_text = f.read()

            session_pk = db.save_session(
                session_id=ts,
                started_at=self._session_start.isoformat(),
                config=metrics_payload.get("config") if self._segment_metrics else None,
                summary=metrics_payload.get("summary") if self._segment_metrics else None,
                transcript=transcript_text,
                audio_file=audio_file,
            )

            if self._segment_metrics:
                db.save_segments(session_pk, self._segment_metrics)

            logger.info(f"Session saved to database (pk={session_pk})")
        except Exception as e:
            logger.warning(f"Failed to save session to database: {e}")

        return transcript_file, audio_file

    def feed_audio_chunk(self, audio_data: np.ndarray):
        """Enqueue an audio chunk (np.float32, 16 kHz) -- used in file mode.

        Does NOT retain audio in all_audio_chunks: in file mode the source file
        already holds the audio, so keeping a full in-RAM copy (~75 MB for a
        20-min meeting) is wasteful and risks exhausting RAM.
        """
        chunk = audio_data.flatten().astype(np.float32)
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
        """Start only the processing threads (no microphone capture).
        Use feed_audio_chunk() to provide audio externally."""
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
