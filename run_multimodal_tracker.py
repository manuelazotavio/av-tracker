import os
import re
import time
import cv2
import logging
import threading
import torch
import numpy as np
from datetime import datetime
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
    def __init__(self, hf_token, device="cuda", num_speakers=None, audio_device=None, video_file=None):
        self.device = 'cuda' if torch.cuda.is_available() and device == "cuda" else 'cpu'
        self.video_file = video_file  # None → webcam em tempo real

        # Person registry compartilhado entre áudio e vídeo
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

        self.verifier = MultiSpeakerVerifier(emb_dir, threshold=0.70)
        # Registra embeddings já conhecidos no registry antes de iniciar o transcriber
        for emb_name in self.verifier.embeddings.keys():
            pid = _make_person_id()
            self.shared_state["embedding_to_person"][emb_name] = pid
            self.shared_state["person_names"][pid] = emb_name

        self.transcriber = RealtimeTranscriber(
            self.verifier,
            hf_token,
            device=self.device,
            verifier_confidence_min=0.80,
            shared_state=self.shared_state,
            num_speakers=num_speakers,
            audio_device=audio_device,
        )

        self.detector = YOLOFaceDetector(model_path="od_model/yolov8n-face.pt", device=self.device)
        self.tracker = PersonIDTracker(model_path="od_model/edgeface_xxs.pt", device=self.device)
        self.tracker.load_known_embeddings(face_emb_dir)

        # Expõe o tracker no shared_state para que o transcriber possa renomear embeddings de rosto
        self.shared_state["face_tracker"] = self.tracker

        # Active Speaker Detection — detecta movimento labial por face
        self.asd = ActiveSpeakerDetector()
        self.shared_state["asd"] = self.asd

    def _feed_file_audio(self, video_file: str):
        """Extrai áudio do arquivo de vídeo e alimenta o transcriber em tempo real simulado."""
        import subprocess, tempfile, importlib
        sr = self.transcriber.sample_rate
        print("🔊 Carregando áudio do vídeo...")

        # Tenta extrair via ffmpeg (mais robusto para MP4/MKV no Windows)
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
            pass  # ffmpeg não está no PATH — tenta librosa
        except Exception as e:
            print(f"⚠️  ffmpeg falhou ({e}), tentando librosa...")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # Fallback: librosa (funciona se audioread/soundfile suportar o formato)
        if audio is None:
            if not _LIBROSA_AVAILABLE:
                print("❌ Nem ffmpeg nem librosa disponíveis. Instale ffmpeg ou: pip install librosa")
                return
            try:
                audio, _ = librosa.load(video_file, sr=sr, mono=True)
            except Exception as e:
                print(f"❌ Erro ao carregar áudio do vídeo: {e}")
                print("   Instale ffmpeg e adicione ao PATH: https://ffmpeg.org/download.html")
                return

        chunk_size = int(self.transcriber.sample_rate * 0.1)  # 100 ms por chunk
        total = len(audio)
        pos = 0
        print(f"🔊 Alimentando {total / self.transcriber.sample_rate:.1f}s de áudio para o transcriber...")
        while pos < total and self.transcriber.running:
            chunk = audio[pos:pos + chunk_size]
            self.transcriber.feed_audio_chunk(chunk)
            pos += chunk_size
            time.sleep(0.1)  # simula tempo real

    def video_loop(self):
        cap = cv2.VideoCapture(self.video_file if self.video_file else 0)

        if not cap.isOpened():
            src = f"arquivo '{self.video_file}'" if self.video_file else "webcam"
            print(f"❌ Erro: Não foi possível abrir {src}.")
            return

        person_names = self.shared_state["person_names"]
        emb_to_person = self.shared_state["embedding_to_person"]
        make_pid = self.shared_state["person_id_factory"]
        _generic_re = re.compile(r'^(Person_\d+|spk_\d+|Desconhecido_\d+)$')
        track_to_pid = {}  # track_id → person_id (persistente entre frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bboxes = self.detector.detect(frame)
            results = self.tracker.update(frame, bboxes)

            current_faces = {}
            used_names_this_frame = {}  # name → track_id (evita dois rostos com mesmo nome real)
            for res in results:
                face_name = res['name']   # "Manuela", "Person_1", etc.
                track_id = res['track_id']

                if track_id in track_to_pid:
                    person_id = track_to_pid[track_id]
                    # Se o face tracker aprendeu um nome real para este track, sincroniza
                    if face_name in emb_to_person:
                        person_id = emb_to_person[face_name]
                        track_to_pid[track_id] = person_id
                    elif not _generic_re.match(face_name) and face_name != "Unknown":
                        # Nome real ainda não mapeado — registra se o display atual é genérico
                        current_display = person_names.get(person_id, "")
                        if _generic_re.match(current_display) or not current_display or current_display == "Unknown":
                            # Guard: não atribuir o mesmo nome real a dois tracks diferentes
                            if face_name not in used_names_this_frame:
                                person_names[person_id] = face_name
                                emb_to_person[face_name] = person_id
                    elif _generic_re.match(face_name):
                        # Face auto-enrolled (ex: Person_1) — avança de "Unknown" para o nome genérico
                        current_display = person_names.get(person_id, "")
                        if not current_display or current_display == "Unknown":
                            # Guard: não usar o mesmo nome genérico para dois person_ids diferentes
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

                # Rastreia nomes reais usados neste frame para evitar colisão
                display_name_check = person_names.get(person_id, face_name)
                if not _generic_re.match(display_name_check) and display_name_check != "Unknown":
                    used_names_this_frame[display_name_check] = track_id

                display_name = person_names.get(person_id, face_name)
                x1, y1, x2, y2 = res['bbox']
                label = f"{display_name} ({res['confidence']:.2f})"
                color = (0, 255, 0) if not _generic_re.match(display_name) else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Atualiza ASD com todos os rostos do frame atual (fora do loop para passar a lista completa)
            self.asd.update(frame, results, time.time())
            self.shared_state["active_faces"] = current_faces

            cv2.imshow("AV-Tracker Multimodal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        if self.video_file:
            # Modo arquivo: inicia só threads de processamento, sem microfone
            self.transcriber.start_file_mode()
            audio_thread = threading.Thread(
                target=self._feed_file_audio, args=(self.video_file,), daemon=True
            )
            audio_thread.start()
            print("\n" + "="*60)
            print("🎬 MODO ARQUIVO DE VÍDEO")
            print(f"Arquivo: {self.video_file}")
            print("="*60 + "\n")
        else:
            audio_thread = threading.Thread(target=self.transcriber.start_recording, daemon=True)
            audio_thread.start()
            print("\n" + "="*60)
            print("🚀 SISTEMA MULTIMODAL ATIVO")
            print("Câmera e Microfone monitorando simultaneamente...")
            print("="*60 + "\n")

        self.video_loop()  # main thread — necessário no Windows para cv2.imshow

        # Ao fechar o vídeo, para o áudio e salva a sessão
        self.transcriber.stop()
        transcript_file, audio_file = self.transcriber.save_session()
        print(f"\n📝 Transcrição salva: {transcript_file}")
        if audio_file:
            print(f"🔊 Áudio salvo: {audio_file}")

def setup_session_logging(log_dir: str = "realtime_sessions") -> str:
    """Adiciona FileHandler ao root logger para salvar os logs da sessão em arquivo.
    Bibliotecas de terceiros ruidosas (numba, urllib3, etc.) são limitadas a WARNING.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Root em DEBUG para capturar os nossos loggers (src.*) que já foram
    # configurados em DEBUG por realtime_transcriber.py
    root = logging.getLogger()
    root.addHandler(fh)
    if root.level == logging.NOTSET or root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    # Suprime bibliotecas de terceiros ruidosas no arquivo de log
    _noisy = [
        "numba", "urllib3", "matplotlib", "fsspec",
        "speechbrain", "faster_whisper", "pyannote",
        "torch", "torchaudio", "transformers", "huggingface_hub",
    ]
    for name in _noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    print(f"📋 Logs salvos em: {log_path}")
    return log_path


if __name__ == "__main__":
    HF_TOKEN = "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"

    video_file = None
    while True:
        video_file_input = input("Caminho do arquivo de vídeo (Enter para usar webcam/microfone): ").strip()
        if not video_file_input:
            break
        if os.path.isfile(video_file_input):
            video_file = video_file_input
            break
        print(f"❌ Arquivo não encontrado: {video_file_input!r} — tente novamente.")

    audio_device = None
    if not video_file:
        import sounddevice as _sd
        print("\n🎤 Dispositivos de entrada disponíveis:")
        devices = _sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        for i, d in input_devices:
            print(f"  [{i}] {d['name']}")
        default_idx = _sd.default.device[0]
        print(f"\nPadrão atual: [{default_idx}] {devices[default_idx]['name']}")
        dev_input = input("Número do dispositivo de entrada (Enter para usar o padrão): ").strip()
        audio_device = int(dev_input) if dev_input.isdigit() else None

    num_speakers_input = input("Quantas pessoas na reunião? (Enter para sem limite): ").strip()
    num_speakers = None
    if num_speakers_input.isdigit() and int(num_speakers_input) > 0:
        num_speakers = int(num_speakers_input)
        print(f"Limite de speakers: {num_speakers}")
    else:
        print("Sem limite de speakers")

    setup_session_logging()
    app = MultimodalFusion(HF_TOKEN, num_speakers=num_speakers, audio_device=audio_device, video_file=video_file)
    app.run()