"""
Real-time Audio-Visual Active Speaker Detection using Light-ASD.

Model: "A Light Weight Model for Active Speaker Detection" (Liao et al., CVPR 2023)
Architecture copied from https://github.com/Junhua-Liao/Light-ASD (MIT licence).

Public API:
    detector = LightASDDetector(model_path, device)
    detector.update(frame_bgr, face_results, timestamp, audio_buf)
    speaking: bool = detector.is_speaking_now(track_id)
    score:   float = detector.get_score(track_id)   # 0..1
"""

import time
import logging
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model architecture — verbatim from Light-ASD repo (MIT licence)
# ---------------------------------------------------------------------------

class _Audio_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.t_3 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.m_5 = nn.Conv2d(in_ch, out_ch, (5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.t_5 = nn.Conv2d(out_ch, out_ch, (1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.last = nn.Conv2d(out_ch, out_ch, (1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x3 = self.relu(self.bn_m_3(self.m_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_m_5(self.m_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class _Visual_Block(nn.Module):
    def __init__(self, in_ch, out_ch, down=False):
        super().__init__()
        self.relu = nn.ReLU()
        stride = (1, 2, 2) if down else (1, 1, 1)
        self.s_3 = nn.Conv3d(in_ch, out_ch, (1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn_s_3 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.t_3 = nn.Conv3d(out_ch, out_ch, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t_3 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.s_5 = nn.Conv3d(in_ch, out_ch, (1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False)
        self.bn_s_5 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.t_5 = nn.Conv3d(out_ch, out_ch, (5, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn_t_5 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.last = nn.Conv3d(out_ch, out_ch, (1, 1, 1), bias=False)
        self.bn_last = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x3 = self.relu(self.bn_s_3(self.s_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_s_5(self.s_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class _VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _Visual_Block(1, 32, down=True)
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _Visual_Block(32, 64)
        self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _Visual_Block(64, 128)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = x.transpose(1, 2)
        B, T, C, W, H = x.shape
        x = x.reshape(B * T, C, W, H)
        x = self.maxpool(x).view(B, T, C)
        return x


class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _Audio_Block(1, 32)
        self.pool1 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block2 = _Audio_Block(32, 64)
        self.pool2 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block3 = _Audio_Block(64, 128)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)
        return x


class _BGRU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fwd = nn.GRU(ch, ch, batch_first=True)
        self.bwd = nn.GRU(ch, ch, batch_first=True)
        self.gelu = nn.GELU()
        for m in self.modules():
            if isinstance(m, nn.GRU):
                nn.init.kaiming_normal_(m.weight_ih_l0)
                nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_(); m.bias_hh_l0.data.zero_()

    def forward(self, x):
        x, _ = self.fwd(x)
        x = self.gelu(x)
        x = torch.flip(x, [1])
        x, _ = self.bwd(x)
        x = torch.flip(x, [1])
        return self.gelu(x)


class _ASD_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()
        self.GRU = _BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        return self.visualEncoder(x)

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        return self.audioEncoder(x)

    def forward_audio_visual_backend(self, a, v):
        x = a + v
        x = self.GRU(x)
        return x.reshape(-1, 128)


# ---------------------------------------------------------------------------
# Real-time inference wrapper
# ---------------------------------------------------------------------------

class LightASDDetector:
    """Per-track audio-visual speaking score with hysteresis."""

    # Inference config
    WINDOW_SEC     = 1.0    # temporal context per inference call
    AUDIO_FPS      = 100    # MFCC frames per second (10ms hop)
    VIDEO_FPS      = 25     # model training FPS — crops are interpolated to this
    INFER_INTERVAL = 0.30   # seconds between inference runs per track

    # Hysteresis (same semantics as the pixel-diff version)
    SPEAK_ON_THRESH  = 0.55   # model score to START showing box
    SPEAK_OFF_THRESH = 0.35   # model score to STOP showing box
    SPEAK_ON_SEC     = 0.25   # sustained time above ON to activate
    SPEAK_OFF_SEC    = 0.50   # sustained time below OFF to deactivate

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = _ASD_Model().to(self.device).eval()
        self.fc = nn.Linear(128, 2).to(self.device)

        state = torch.load(model_path, map_location=self.device, weights_only=False)
        model_sd = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        self.model.load_state_dict(model_sd)
        if "lossAV.FC.weight" in state:
            self.fc.weight.data = state["lossAV.FC.weight"].to(self.device)
            self.fc.bias.data   = state["lossAV.FC.bias"].to(self.device)
        self.fc.eval()
        logger.info(f"LightASD loaded ({self.device}) from {model_path}")

        # Per-track state
        self._crop_buf:      dict[int, deque] = {}   # track_id → deque[(t, np.uint8 112x112)]
        self._score_buf:     dict[int, deque] = {}   # track_id → deque[(t, score)]
        self._last_infer:    dict[int, float] = {}   # track_id → last inference timestamp
        self._speak_state:   dict[int, bool]  = {}   # hysteresis on/off
        self._speak_since:   dict[int, float] = {}   # when state last changed
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Per-frame update (called from video loop)
    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray, face_results: list,
               timestamp: float, audio_buf: deque | None) -> None:
        """Extract mouth crops, schedule inference runs."""
        current_ids = set()
        for res in face_results:
            tid = res["track_id"]
            current_ids.add(tid)
            crop = self._extract_crop(frame_bgr, res["bbox"])
            if crop is None:
                continue
            buf = self._crop_buf.setdefault(tid, deque())
            buf.append((timestamp, crop))
            # Keep only the last WINDOW_SEC + 0.5s of crops
            cutoff = timestamp - self.WINDOW_SEC - 0.5
            while buf and buf[0][0] < cutoff:
                buf.popleft()

        # Trigger inference for tracks that are due
        if audio_buf is not None:
            for tid in list(current_ids):
                last = self._last_infer.get(tid, 0.0)
                if timestamp - last >= self.INFER_INTERVAL:
                    self._run_inference(tid, timestamp, audio_buf)

        # Clean up gone tracks
        for tid in list(self._crop_buf):
            if tid not in current_ids:
                self._crop_buf.pop(tid, None)
                self._score_buf.pop(tid, None)
                self._last_infer.pop(tid, None)
                self._speak_state.pop(tid, None)
                self._speak_since.pop(tid, None)

    # ------------------------------------------------------------------
    # Hysteresis query
    # ------------------------------------------------------------------

    def is_speaking_now(self, track_id: int) -> bool:
        now = time.time()
        score = self.get_score(track_id)
        speaking = self._speak_state.get(track_id, False)

        if speaking:
            if score < self.SPEAK_OFF_THRESH:
                since = self._speak_since.get(track_id, now)
                if now - since >= self.SPEAK_OFF_SEC:
                    self._speak_state[track_id] = False
                    self._speak_since[track_id] = now
                    return False
            else:
                self._speak_since[track_id] = now  # reset timer while above threshold
            return True
        else:
            if score >= self.SPEAK_ON_THRESH:
                since = self._speak_since.get(track_id, now)
                if now - since >= self.SPEAK_ON_SEC:
                    self._speak_state[track_id] = True
                    self._speak_since[track_id] = now
                    return True
            else:
                self._speak_since[track_id] = now  # reset timer while below threshold
            return False

    def get_score(self, track_id: int) -> float:
        """Mean speaking score over the last INFER_INTERVAL seconds."""
        sbuf = self._score_buf.get(track_id)
        if not sbuf:
            return 0.0
        now = time.time()
        recent = [s for t, s in sbuf if t >= now - self.INFER_INTERVAL * 2]
        return float(np.mean(recent)) if recent else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_crop(frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return None
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cs = max(h, w) // 2
        pad = cs + 10
        padded = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=110)
        face = padded[cy + pad - cs: cy + pad + cs, cx + pad - cs: cx + pad + cs]
        if face.size == 0:
            return None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224))
        return resized[56:168, 56:168]  # center 112×112

    @staticmethod
    def _compute_mfcc(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32), sr=sr,
            n_mfcc=13, n_fft=400, hop_length=160, center=False
        ).T  # (T, 13)
        return mfcc.astype(np.float32)

    def _run_inference(self, track_id: int, now: float, audio_buf: deque) -> None:
        self._last_infer[track_id] = now
        crop_buf = self._crop_buf.get(track_id)
        if not crop_buf:
            return

        t_start = now - self.WINDOW_SEC
        crops = [(t, c) for t, c in crop_buf if t >= t_start]
        if len(crops) < 3:
            return

        # Collect audio samples for the same window
        audio_samples = self._collect_audio(audio_buf, t_start, now)
        if audio_samples is None or len(audio_samples) < 800:  # < 50ms
            return

        # Interpolate crops to VIDEO_FPS
        T_v = self.VIDEO_FPS  # always 25 frames per 1-second window
        ts = np.array([t for t, _ in crops])
        ts_target = np.linspace(ts[0], ts[-1], T_v)
        interp_crops = np.stack([
            crops[int(np.clip(np.searchsorted(ts, t) - 1, 0, len(crops) - 1))][1]
            for t in ts_target
        ])  # (25, 112, 112) uint8

        mfcc = self._compute_mfcc(audio_samples)
        T_a_target = T_v * 4  # 100 frames for 1 second
        if mfcc.shape[0] < T_a_target:
            pad = T_a_target - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
        else:
            mfcc = mfcc[:T_a_target]

        try:
            with self._infer_lock, torch.no_grad():
                inputV = torch.FloatTensor(interp_crops).unsqueeze(0).to(self.device)
                inputA = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
                eV = self.model.forward_visual_frontend(inputV)
                eA = self.model.forward_audio_frontend(inputA)
                out = self.model.forward_audio_visual_backend(eA, eV)
                scores = F.softmax(self.fc(out), dim=-1)[:, 1].cpu().numpy()
            # Store one score per video frame, keyed by estimated frame time
            sbuf = self._score_buf.setdefault(track_id, deque())
            for i, s in enumerate(scores):
                t_frame = t_start + (i / T_v) * self.WINDOW_SEC
                sbuf.append((t_frame, float(s)))
            cutoff = now - self.WINDOW_SEC * 2
            while sbuf and sbuf[0][0] < cutoff:
                sbuf.popleft()
            logger.debug(f"LightASD track={track_id} mean={np.mean(scores):.2f}")
        except Exception as e:
            logger.debug(f"LightASD inference error: {e}")

    @staticmethod
    def _collect_audio(audio_buf: deque, t_start: float, t_end: float) -> np.ndarray | None:
        """Concatenate audio chunks overlapping [t_start, t_end]."""
        pieces = []
        for t_chunk_start, chunk in audio_buf:
            sr = 16000
            chunk_dur = len(chunk) / sr
            t_chunk_end = t_chunk_start + chunk_dur
            if t_chunk_end < t_start or t_chunk_start > t_end:
                continue
            # Slice relevant samples
            s0 = max(0, int((t_start - t_chunk_start) * sr))
            s1 = min(len(chunk), int((t_end - t_chunk_start) * sr))
            if s1 > s0:
                pieces.append(chunk[s0:s1])
        if not pieces:
            return None
        return np.concatenate(pieces).astype(np.float32)
