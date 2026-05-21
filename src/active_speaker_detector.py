"""
Active Speaker Detection (ASD).

Primary backend: Light-ASD (Liao et al., CVPR 2023) — audio-visual neural model.
Fallback backend: pixel-diff on mouth region (no model, always available).

get_active_speaker() / get_best_guess() — used by the audio thread to attribute
a speech segment to a face — prefer Light-ASD's neural scores and fall back to
pixel-diff only when Light-ASD has no data for the requested window.
The pixel-diff backend keeps running every frame so the fallback always works.
"""

import time
import logging
import cv2
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class ActiveSpeakerDetector:
    # Mouth region: 60-100% vertical, 15-85% horizontal of the face bbox
    MOUTH_TOP_RATIO   = 0.60
    MOUTH_SIDE_MARGIN = 0.15
    # Upper face region (forehead + eyes): used as motion baseline to detect occlusion.
    # If the mouth region moves no more than OCCLUSION_RATIO × upper face, it's not speech.
    UPPER_BOTTOM_RATIO = 0.50   # 0-50% vertical = forehead/eyes
    OCCLUSION_RATIO    = 1.8    # mouth must be 1.8× more active than upper face to count

    def __init__(self, buffer_seconds: float = 15.0, min_frames: int = 2,
                 dominant_ratio: float = 1.4, light_asd_model_path: str | None = None,
                 device: str = "cuda"):
        self._prev_mouth:  dict[int, np.ndarray] = {}
        self._prev_upper:  dict[int, np.ndarray] = {}
        self._activity:    dict[int, deque]       = {}
        self._buffer_seconds = buffer_seconds
        self._min_frames     = min_frames
        self._dominant_ratio = dominant_ratio

        # Pixel-diff hysteresis (fallback when Light-ASD is unavailable)
        self.SPEAK_ON_FLOOR  = 3.0
        self.SPEAK_ON_SEC    = 0.35
        self.SPEAK_OFF_SEC   = 0.6
        self._speak_state: dict[int, bool]  = {}
        self._speak_since: dict[int, float] = {}

        # Light-ASD neural backend (optional)
        self._light_asd = None
        if light_asd_model_path:
            try:
                from src.light_asd_detector import LightASDDetector
                self._light_asd = LightASDDetector(light_asd_model_path, device=device)
                logger.info("LightASD backend active")
            except Exception as e:
                logger.warning(f"LightASD unavailable, falling back to pixel-diff: {e}")

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, face_results: list,
               timestamp: float | None = None, audio_buf=None) -> None:
        """Called every frame with the face tracker results.

        Args:
            frame:        BGR camera frame.
            face_results: list of dicts with 'track_id' and 'bbox' (x1,y1,x2,y2).
            timestamp:    epoch in seconds (default: time.time()).
            audio_buf:    shared_state["asd_audio_buf"] deque — needed by Light-ASD.
        """
        if timestamp is None:
            timestamp = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        current_ids: set[int] = set()

        for res in face_results:
            track_id = res["track_id"]
            x1, y1, x2, y2 = (int(v) for v in res["bbox"])
            h, w = y2 - y1, x2 - x1
            if h <= 0 or w <= 0:
                continue

            my1 = y1 + int(h * self.MOUTH_TOP_RATIO)
            my2 = y2
            mx1 = x1 + int(w * self.MOUTH_SIDE_MARGIN)
            mx2 = x2 - int(w * self.MOUTH_SIDE_MARGIN)

            # Upper face region used as motion baseline (forehead/eyes)
            uy1 = y1
            uy2 = y1 + int(h * self.UPPER_BOTTOM_RATIO)

            mouth = gray[my1:my2, mx1:mx2]
            upper = gray[uy1:uy2, mx1:mx2]
            if mouth.size == 0:
                continue

            current_ids.add(track_id)

            # Pixel difference relative to previous frame
            prev_mouth = self._prev_mouth.get(track_id)
            prev_upper = self._prev_upper.get(track_id)
            if prev_mouth is not None and prev_mouth.shape == mouth.shape:
                mouth_diff = float(np.mean(np.abs(mouth.astype(np.float32) - prev_mouth.astype(np.float32))))
            else:
                mouth_diff = 0.0

            if prev_upper is not None and prev_upper.shape == upper.shape and upper.size > 0:
                upper_diff = float(np.mean(np.abs(upper.astype(np.float32) - prev_upper.astype(np.float32))))
            else:
                upper_diff = 0.0

            # Reject occlusion: hand/object covering mouth moves entire face region similarly.
            # Only count mouth activity when it's significantly higher than upper-face motion.
            if mouth_diff > 0 and mouth_diff < self.OCCLUSION_RATIO * (upper_diff + 0.5):
                diff = 0.0
            else:
                diff = mouth_diff

            self._prev_mouth[track_id] = mouth.copy()
            self._prev_upper[track_id] = upper.copy() if upper.size > 0 else upper

            if track_id not in self._activity:
                self._activity[track_id] = deque()
            self._activity[track_id].append((timestamp, diff))

            # Remove old entries
            cutoff = timestamp - self._buffer_seconds
            buf = self._activity[track_id]
            while buf and buf[0][0] < cutoff:
                buf.popleft()

        # Clean up tracks that disappeared from the scene
        for tid in list(self._prev_mouth):
            if tid not in current_ids:
                del self._prev_mouth[tid]
                self._prev_upper.pop(tid, None)

        # Light-ASD update (audio-visual neural backend)
        if self._light_asd is not None:
            self._light_asd.update(frame, face_results, timestamp, audio_buf)

    # ------------------------------------------------------------------
    # Query: who was speaking during [time_start, time_end]?
    # ------------------------------------------------------------------
    # Minimum mean pixel-diff to be considered "active" (not just background noise).
    # Typical values: non-speaking faces 0.3-1.0, speaking faces 3-15+.
    # Low-res or compressed video may have lower overall values.
    NOISE_FLOOR = 1.5

    def get_active_speaker(self, time_start: float, time_end: float) -> int | None:
        """Returns track_id of the most active speaker in the window, or None if uncertain.

        Prefers the Light-ASD neural backend; falls back to pixel-diff only when
        Light-ASD has no score data covering the requested window.

        Args:
            time_start: start of the audio segment (epoch seconds).
            time_end:   end of the audio segment (epoch seconds).
        """
        if self._light_asd is not None and self._light_asd.has_scores(time_start, time_end):
            return self._light_asd.get_active_speaker(time_start, time_end)
        # -- pixel-diff fallback --
        scores: dict[int, float] = {}
        for track_id, buf in self._activity.items():
            window = [s for t, s in buf if time_start <= t <= time_end]
            if len(window) >= self._min_frames:
                scores[track_id] = float(np.mean(window))

        if not scores:
            return None

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = sorted_items[0]

        scores_str = " | ".join(f"t{tid}:{s:.1f}" for tid, s in sorted_items)
        logger.debug(f"ASD scores [{time_end - time_start:.2f}s window]: {scores_str}")

        # Best face must be above noise floor to be considered speaking
        if best_score < self.NOISE_FLOOR:
            logger.debug(f"ASD=None: best score {best_score:.1f} < noise floor {self.NOISE_FLOOR}")
            return None

        if len(scores) == 1:
            return best_id

        _, second_score = sorted_items[1]

        # If second face is below noise floor, best is clearly the speaker
        if second_score < self.NOISE_FLOOR:
            logger.debug(f"ASD -> t{best_id} ({best_score:.1f}, second below noise)")
            return best_id

        # Both above noise floor — require dominant_ratio to avoid ties
        ratio = best_score / second_score
        if ratio < self._dominant_ratio:
            logger.debug(f"ASD=None: ratio {ratio:.2f} < {self._dominant_ratio} (best={best_score:.1f} second={second_score:.1f})")
            return None
        logger.debug(f"ASD -> t{best_id} ({best_score:.1f}, ratio={ratio:.2f})")
        return best_id

    def get_best_guess(self, time_start: float, time_end: float) -> int | None:
        """Fallback: returns the face with most mouth movement even below NOISE_FLOOR.

        Used when the verifier has no match and get_active_speaker() returned None.
        Only returns when the best face has score > 0 AND is clearly dominant (2x second).
        """
        if self._light_asd is not None and self._light_asd.has_scores(time_start, time_end):
            return self._light_asd.get_active_speaker(
                time_start, time_end,
                min_score=self._light_asd.GUESS_MIN_SCORE,
                margin=self._light_asd.GUESS_MARGIN,
            )
        # -- pixel-diff fallback --
        scores: dict[int, float] = {}
        for track_id, buf in self._activity.items():
            window = [s for t, s in buf if time_start <= t <= time_end]
            if len(window) >= self._min_frames:
                scores[track_id] = float(np.mean(window))

        if not scores:
            return None

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = sorted_items[0]

        if best_score <= 0:
            return None

        if len(scores) == 1:
            return best_id

        _, second_score = sorted_items[1]

        # Require best to be at least 1.5x the second
        if second_score > 0 and best_score / second_score < 1.5:
            return None

        logger.debug(f"ASD_GUESS -> t{best_id} ({best_score:.1f}, below noise but dominant)")
        return best_id

    def is_speaking_now(self, track_id: int) -> bool:
        """Real-time check: delegates to Light-ASD if available, else pixel-diff hysteresis."""
        if self._light_asd is not None:
            return self._light_asd.is_speaking_now(track_id)
        # -- pixel-diff fallback --
        buf = self._activity.get(track_id)
        now = time.time()

        currently_speaking = self._speak_state.get(track_id, False)

        if not buf:
            if currently_speaking:
                self._speak_state[track_id] = False
                self._speak_since[track_id] = now
            return False

        if currently_speaking:
            # Check if we should turn OFF: need SPEAK_OFF_SEC below NOISE_FLOOR
            window = [s for t, s in buf if t >= now - self.SPEAK_OFF_SEC]
            below_floor = len(window) >= self._min_frames and float(np.mean(window)) < self.NOISE_FLOOR
            if below_floor:
                self._speak_state[track_id] = False
                self._speak_since[track_id] = now
                return False
            return True
        else:
            # Check if we should turn ON: need SPEAK_ON_SEC above SPEAK_ON_FLOOR
            window = [s for t, s in buf if t >= now - self.SPEAK_ON_SEC]
            above_floor = len(window) >= self._min_frames and float(np.mean(window)) >= self.SPEAK_ON_FLOOR
            if above_floor:
                self._speak_state[track_id] = True
                self._speak_since[track_id] = now
                return True
            return False

    def get_scores(self, time_start: float, time_end: float) -> dict[int, float]:
        """Returns activity scores for all faces (for debug)."""
        return {
            tid: float(np.mean([s for t, s in buf if time_start <= t <= time_end]))
            for tid, buf in self._activity.items()
            if any(time_start <= t <= time_end for t, _ in buf)
        }
