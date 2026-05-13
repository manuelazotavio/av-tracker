import os
import re
import time
import cv2
import logging
import threading
import torch
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # loads .env file (HF_TOKEN, etc.)
from src.realtime_transcriber import RealtimeTranscriber
from src.multi_speaker_verifier import MultiSpeakerVerifier
from src.yolo_detector import YOLOFaceDetector
from src.personid_tracker import PersonIDTracker
from src.active_speaker_detector import ActiveSpeakerDetector

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

class MultimodalFusion:
    def __init__(self, hf_token, device="cuda", num_speakers=None, audio_device=None, video_file=None, screen_region=None, headless=False):
        self.device = 'cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu'
        self.video_file = video_file  # None → real-time webcam
        self.screen_region = screen_region  # dict for mss screen capture (meeting mode)
        self.num_speakers = num_speakers  # limits tracked faces by area
        self.headless = headless  # True = no interactive prompts (launched from GUI)

        # Person registry shared between audio and video
        _counter = [0]
        def _make_person_id():
            _counter[0] += 1
            return f"spk_{_counter[0]:03d}"

        self.shared_state = {
            "active_faces": {},         # track_id → person_id
            "person_names": {},         # person_id → display_name
            "embedding_to_person": {},  # embedding_key → person_id
            "person_id_factory": _make_person_id,
        }

        base_dir = os.path.abspath(os.path.dirname(__file__))
        emb_dir = os.path.join(base_dir, "data", "embeddings")
        face_emb_dir = os.path.join(base_dir, "data", "face_embeddings")

        os.makedirs(emb_dir, exist_ok=True)
        os.makedirs(face_emb_dir, exist_ok=True)

        self.verifier = MultiSpeakerVerifier(emb_dir, threshold=0.80)
        # Register already known embeddings in the registry before starting the transcriber
        for emb_name in self.verifier.embeddings.keys():
            pid = _make_person_id()
            self.shared_state["embedding_to_person"][emb_name] = pid
            self.shared_state["person_names"][pid] = emb_name

        self.transcriber = RealtimeTranscriber(
            self.verifier,
            hf_token,
            device=self.device,
            whisper_size="large-v3-turbo",
            use_ai_analysis=True,
            verifier_confidence_min=0.70,
            shared_state=self.shared_state,
            num_speakers=num_speakers,
            audio_device=audio_device,
        )

        self.detector = YOLOFaceDetector(model_path="od_model/yolov8n-face.pt", device=self.device)
        self.tracker = PersonIDTracker(model_path="od_model/edgeface_xxs.pt", device=self.device)
        self.tracker.load_known_embeddings(face_emb_dir)

        # Expose the tracker in shared_state so the transcriber can rename face embeddings
        self.shared_state["face_tracker"] = self.tracker

        # Active Speaker Detection — Light-ASD (audio-visual) with pixel-diff fallback
        _light_asd_path = "pretrained_models/Light-ASD-repo/weight/pretrain_AVA_CVPR.model"
        self.asd = ActiveSpeakerDetector(
            light_asd_model_path=_light_asd_path,
            device=self.device,
        )
        self.shared_state["asd"] = self.asd

    def _feed_file_audio(self, video_file: str):
        """Extracts audio from the video file and feeds the transcriber in simulated real-time."""
        import subprocess, tempfile, importlib
        sr = self.transcriber.sample_rate
        print("🔊 Loading audio from video...")

        # Try extracting via ffmpeg (more robust for MP4/MKV on Windows)
        audio = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", video_file,
                 "-ar", str(sr), "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True, timeout=120
            )
            if result.returncode == 0:
                sf_mod = importlib.import_module("soundfile")
                audio, _ = sf_mod.read(tmp_path, dtype="float32")
            else:
                raise RuntimeError(result.stderr.decode(errors="ignore")[-300:])
        except FileNotFoundError:
            pass  # ffmpeg not in PATH — try librosa
        except Exception as e:
            print(f"⚠️  ffmpeg failed ({e}), trying librosa...")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # Fallback: librosa (works if audioread/soundfile supports the format)
        if audio is None:
            if not _LIBROSA_AVAILABLE:
                print("❌ Neither ffmpeg nor librosa available. Install ffmpeg or: pip install librosa")
                return
            try:
                audio, _ = librosa.load(video_file, sr=sr, mono=True)
            except Exception as e:
                print(f"❌ Error loading audio from video: {e}")
                print("   Install ffmpeg and add it to PATH: https://ffmpeg.org/download.html")
                return

        chunk_size = int(self.transcriber.sample_rate * 0.1)  # 100 ms per chunk
        total = len(audio)
        pos = 0
        print(f"🔊 Feeding {total / self.transcriber.sample_rate:.1f}s of audio to the transcriber...")
        while pos < total and self.transcriber.running:
            chunk = audio[pos:pos + chunk_size]
            self.transcriber.feed_audio_chunk(chunk)
            pos += chunk_size
            time.sleep(0.1)  # simulate real-time

    def video_loop(self):
        # Screen capture mode (meeting mode) — uses mss instead of cv2.VideoCapture
        sct = None
        cap = None
        if self.screen_region:
            import mss
            sct = mss.mss()
        else:
            cap = cv2.VideoCapture(self.video_file if self.video_file else 0)
            if not cap.isOpened():
                src = f"file '{self.video_file}'" if self.video_file else "webcam"
                print(f"❌ Error: Could not open {src}.")
                return

        person_names = self.shared_state["person_names"]
        emb_to_person = self.shared_state["embedding_to_person"]
        make_pid = self.shared_state["person_id_factory"]
        _generic_re = re.compile(r'^(Person_\d+|spk_\d+|Desconhecido_\d+)$')
        track_to_pid = {}  # track_id → person_id (persistent across frames)

        _vperf = {}  # key -> list[ms]
        _vframe_count = 0
        _VPERF_INTERVAL = 100

        def _vlog(key, ms):
            _vperf.setdefault(key, []).append(ms)

        def _vprint_summary():
            lines = ["⏱ === Video Loop Perf Summary ==="]
            for key in sorted(_vperf):
                vals = _vperf[key]
                lines.append(f"  {key:30s} avg={sum(vals)/len(vals):6.1f}ms  min={min(vals):5.1f}ms  max={max(vals):5.1f}ms  n={len(vals)}")
            print("\n".join(lines))

        # In file mode, 10 fps gives ASD enough frames per segment.
        # 5 fps was causing ASD=None for most short diarization segments (< 1s).
        _target_fps = 10 if self.video_file else 30
        _frame_interval = 1.0 / _target_fps
        _last_frame_time = 0.0

        while (sct is not None) or (cap is not None and cap.isOpened()):
            if sct:
                img = np.array(sct.grab(self.screen_region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                # Skip frames to match target FPS (reduces GPU load)
                now = time.perf_counter()
                if now - _last_frame_time < _frame_interval:
                    continue
                _last_frame_time = now

            _t0_frame = time.perf_counter()

            _t0 = time.perf_counter()
            bboxes = self.detector.detect(frame)
            _vlog("detect", (time.perf_counter() - _t0) * 1000)

            if self.num_speakers and len(bboxes) > self.num_speakers:
                bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:self.num_speakers]

            _t0 = time.perf_counter()
            results = self.tracker.update(frame, bboxes)
            _vlog("tracker_update", (time.perf_counter() - _t0) * 1000)

            current_faces = {}
            used_names_this_frame = {}  # name → track_id (prevents two faces with the same real name)
            for res in results:
                face_name = res['name']   # "Manuela", "Person_1", etc.
                track_id = res['track_id']

                if track_id in track_to_pid:
                    person_id = track_to_pid[track_id]
                    # If the face tracker learned a real name for this track, sync it
                    if face_name in emb_to_person:
                        person_id = emb_to_person[face_name]
                        track_to_pid[track_id] = person_id
                    elif not _generic_re.match(face_name) and face_name != "Unknown":
                        # Real name not yet mapped — register if current display name is generic
                        current_display = person_names.get(person_id, "")
                        if _generic_re.match(current_display) or not current_display or current_display == "Unknown":
                            # Guard: don't assign the same real name to two different tracks
                            if face_name not in used_names_this_frame:
                                person_names[person_id] = face_name
                                emb_to_person[face_name] = person_id
                    elif _generic_re.match(face_name):
                        # Face auto-enrolled (e.g.: Person_1) — upgrade from "Unknown" to the generic name
                        current_display = person_names.get(person_id, "")
                        if not current_display or current_display == "Unknown":
                            # Guard: don't use the same generic name for two different person_ids
                            name_in_use = any(
                                pid != person_id and person_names.get(pid) == face_name
                                for pid in track_to_pid.values()
                            )
                            if not name_in_use:
                                person_names[person_id] = face_name
                elif face_name in emb_to_person:
                    person_id = emb_to_person[face_name]
                    track_to_pid[track_id] = person_id
                else:
                    person_id = make_pid()
                    track_to_pid[track_id] = person_id
                    person_names[person_id] = face_name
                    if not _generic_re.match(face_name) and face_name != "Unknown":
                        emb_to_person[face_name] = person_id

                current_faces[track_id] = person_id

                # Track real names used in this frame to avoid collision
                display_name_check = person_names.get(person_id, face_name)
                if not _generic_re.match(display_name_check) and display_name_check != "Unknown":
                    used_names_this_frame[display_name_check] = track_id

            # Update ASD before drawing so is_speaking_now() reflects the current frame
            _t0 = time.perf_counter()
            self.asd.update(frame, results, time.time(),
                            self.shared_state.get("asd_audio_buf"))
            _vlog("asd_update", (time.perf_counter() - _t0) * 1000)
            self.shared_state["active_faces"] = current_faces

            # Draw boxes only for faces that are currently speaking
            for res in results:
                track_id = res['track_id']
                if not self.asd.is_speaking_now(track_id):
                    continue
                person_id = current_faces.get(track_id)
                display_name = person_names.get(person_id, res['name']) if person_id else res['name']
                x1, y1, x2, y2 = res['bbox']
                label = f"{display_name} ({res['confidence']:.2f})"
                color = (0, 255, 0) if not _generic_re.match(display_name) else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            _vlog("frame_total", (time.perf_counter() - _t0_frame) * 1000)
            _vframe_count += 1
            if _vframe_count % _VPERF_INTERVAL == 0:
                _vprint_summary()

            h, w = frame.shape[:2]
            max_w, max_h = 1280, 720
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                display = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                display = frame
            cv2.imshow("AV-Tracker Multimodal", display)
            # In file mode, wait longer to sync with audio feed (~100ms)
            wait_ms = 1 if not self.video_file else 100
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
            # Periodically free GPU cache to prevent OOM
            if self.video_file and _vframe_count % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cap:
            cap.release()
        if sct:
            sct.close()
        cv2.destroyAllWindows()

    def run(self):
        if self.video_file:
            # File mode: start only processing threads, no microphone
            self.transcriber.start_file_mode()
            audio_thread = threading.Thread(
                target=self._feed_file_audio, args=(self.video_file,), daemon=True
            )
            audio_thread.start()
            print("\n" + "="*60)
            print("🎬 VIDEO FILE MODE")
            print(f"File: {self.video_file}")
            print("="*60 + "\n")
        elif self.screen_region:
            # Meeting mode: screen capture + system audio
            audio_thread = threading.Thread(target=self.transcriber.start_recording, daemon=True)
            audio_thread.start()
            r = self.screen_region
            print("\n" + "="*60)
            print("🖥️  MEETING MODE")
            print(f"Screen region: {r['width']}x{r['height']} at ({r['left']},{r['top']})")
            print("Audio: system loopback (select Mixagem Estereo / Stereo Mix)")
            print("Press 'q' on the preview window to stop.")
            print("="*60 + "\n")
        else:
            audio_thread = threading.Thread(target=self.transcriber.start_recording, daemon=True)
            audio_thread.start()
            print("\n" + "="*60)
            print("🚀 MULTIMODAL SYSTEM ACTIVE")
            print("Camera and Microphone monitoring simultaneously...")
            print("="*60 + "\n")

        self.video_loop()  # main thread — required on Windows for cv2.imshow

        # When closing the video, stop audio and save the session
        self.transcriber.stop()
        if not self.headless:
            self.transcriber.prompt_and_save_unknown_speakers()
        transcript_file, audio_file = self.transcriber.save_session()
        print(f"\n📝 Transcription saved: {transcript_file}")
        if audio_file:
            print(f"🔊 Audio saved: {audio_file}")

def setup_session_logging(log_dir: str = "realtime_sessions") -> str:
    """Adds a FileHandler to the root logger to save session logs to a file.
    Only logs from src.* (DEBUG+) and WARNING+ from any source are written to the file.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"log_{ts}.txt")

    class _AppFilter(logging.Filter):
        """Only passes through logs from src.* or WARNING+ from any source."""
        def filter(self, record):
            return record.name.startswith("src.") or record.levelno >= logging.WARNING

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(_AppFilter())

    # Root at DEBUG to capture our loggers (src.*) that were already
    # configured at DEBUG by realtime_transcriber.py
    root = logging.getLogger()
    root.addHandler(fh)
    if root.level == logging.NOTSET or root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    print(f"📋 Logs saved to: {log_path}")
    return log_path


def _parse_args():
    """Parse CLI arguments. If none provided, fall back to interactive prompts."""
    import argparse
    parser = argparse.ArgumentParser(description="AV-Tracker Multimodal")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--speakers", type=int, default=None, help="Number of speakers (0=no limit)")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--meeting", action="store_true", help="Meeting mode (screen capture)")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index for meeting mode")
    parser.add_argument("--headless", action="store_true", help="No interactive prompts (for GUI launch)")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        HF_TOKEN = input("HF_TOKEN not found in environment. Paste your token: ").strip()

    args = _parse_args()

    video_file = None
    screen_region = None
    audio_device = args.audio_device
    num_speakers = args.speakers

    # --- CLI mode (arguments provided) ---
    if args.video:
        if os.path.isfile(args.video):
            video_file = args.video
        else:
            print(f"File not found: {args.video}")
            exit(1)
    elif args.meeting:
        import mss
        sct = mss.mss()
        monitors = sct.monitors
        mon_idx = min(args.monitor, len(monitors) - 1)
        screen_region = monitors[mon_idx]
        sct.close()
        print(f"Meeting mode: {screen_region['width']}x{screen_region['height']}")

    # --- Interactive mode (no arguments) ---
    elif not any([args.video, args.meeting, args.speakers is not None]):
        print("\n📌 Mode:")
        print("  [1] Webcam + Microphone (default)")
        print("  [2] Video file")
        print("  [3] Meeting mode (screen capture + system audio)")
        mode_input = input("Choose mode (1/2/3): ").strip()

        if mode_input == "2":
            while True:
                video_file_input = input("Path to video file: ").strip()
                if os.path.isfile(video_file_input):
                    video_file = video_file_input
                    break
                print(f"File not found: {video_file_input!r} — try again.")

        elif mode_input == "3":
            import mss
            sct = mss.mss()
            monitors = sct.monitors
            print("\n Available monitors:")
            for i, m in enumerate(monitors):
                if i == 0:
                    print(f"  [0] Full virtual screen ({m['width']}x{m['height']})")
                else:
                    print(f"  [{i}] Monitor {i} ({m['width']}x{m['height']} at {m['left']},{m['top']})")
            mon_input = input("Monitor number (Enter for primary): ").strip()
            mon_idx = int(mon_input) if mon_input.isdigit() else 1
            screen_region = monitors[min(mon_idx, len(monitors) - 1)]
            sct.close()
            print(f"  Capturing: {screen_region['width']}x{screen_region['height']}")

    # Audio device selection (interactive only if not provided via CLI)
    if not video_file and audio_device is None:
        import sounddevice as _sd
        print("\n Available input devices:")
        devices = _sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        for i, d in input_devices:
            hint = ""
            name_lower = d['name'].lower()
            if 'stereo mix' in name_lower or 'mixagem' in name_lower or 'loopback' in name_lower:
                hint = "  <- SYSTEM AUDIO (recommended for meeting mode)" if screen_region else ""
            print(f"  [{i}] {d['name']}{hint}")
        default_idx = _sd.default.device[0]
        print(f"\nCurrent default: [{default_idx}] {devices[default_idx]['name']}")
        dev_input = input("Input device number (Enter to use default): ").strip()
        audio_device = int(dev_input) if dev_input.isdigit() else None

    # Speaker count (interactive only if not provided via CLI)
    if num_speakers is None:
        num_speakers_input = input("How many people in the meeting? (Enter for no limit): ").strip()
        if num_speakers_input.isdigit() and int(num_speakers_input) > 0:
            num_speakers = int(num_speakers_input)

    if num_speakers and num_speakers > 0:
        print(f"Speaker limit: {num_speakers}")
    else:
        print("No speaker limit")
        num_speakers = None

    setup_session_logging()
    app = MultimodalFusion(HF_TOKEN, num_speakers=num_speakers, audio_device=audio_device,
                           video_file=video_file, screen_region=screen_region, headless=args.headless)
    app.run()