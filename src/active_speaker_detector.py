"""
Active Speaker Detection (ASD) based on lip movement.

For each tracked face, computes pixel difference in the mouth region
between consecutive frames. The speaker with the highest average activity
during the audio segment is the active speaker candidate.

No additional models — uses only the already-processed camera frames.
"""

import time
import cv2
import numpy as np
from collections import deque


class ActiveSpeakerDetector:
    # Mouth region: 60-100% vertical, 15-85% horizontal of the face bbox
    MOUTH_TOP_RATIO  = 0.60
    MOUTH_SIDE_MARGIN = 0.15

    def __init__(self, buffer_seconds: float = 15.0, min_frames: int = 5,
                 dominant_ratio: float = 1.6):
        """
        Args:
            buffer_seconds: history window kept in memory.
            min_frames: minimum number of frames with data to consider a speaker.
            dominant_ratio: minimum ratio between the best and second score to
                            confirm the active speaker (avoids ties).
        """
        self._prev_mouth: dict[int, np.ndarray] = {}   # track_id → previous ROI
        self._activity:   dict[int, deque]        = {}  # track_id → deque[(t, score)]
        self._buffer_seconds = buffer_seconds
        self._min_frames     = min_frames
        self._dominant_ratio = dominant_ratio

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, face_results: list, timestamp: float | None = None) -> None:
        """Called every frame with the face tracker results.

        Args:
            frame: BGR or grayscale camera frame.
            face_results: list of dicts with 'track_id' and 'bbox' (x1,y1,x2,y2).
            timestamp: epoch in seconds (default: time.time()).
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

            mouth = gray[my1:my2, mx1:mx2]
            if mouth.size == 0:
                continue

            current_ids.add(track_id)

            # Pixel difference relative to previous frame
            prev = self._prev_mouth.get(track_id)
            if prev is not None and prev.shape == mouth.shape:
                diff = float(np.mean(np.abs(mouth.astype(np.float32) - prev.astype(np.float32))))
            else:
                diff = 0.0

            self._prev_mouth[track_id] = mouth.copy()

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

    # ------------------------------------------------------------------
    # Query: who was speaking during [time_start, time_end]?
    # ------------------------------------------------------------------
    def get_active_speaker(self, time_start: float, time_end: float) -> int | None:
        """Returns track_id of the most active speaker in the window, or None if uncertain.

        Args:
            time_start: start of the audio segment (epoch seconds).
            time_end:   end of the audio segment (epoch seconds).
        """
        scores: dict[int, float] = {}
        for track_id, buf in self._activity.items():
            window = [s for t, s in buf if time_start <= t <= time_end]
            if len(window) >= self._min_frames:
                scores[track_id] = float(np.mean(window))

        if not scores:
            return None
        if len(scores) == 1:
            return next(iter(scores))

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id,  best_score  = sorted_items[0]
        _,        second_score = sorted_items[1]

        # Only returns if clearly dominant
        if second_score > 0 and best_score / second_score < self._dominant_ratio:
            return None
        return best_id

    def get_scores(self, time_start: float, time_end: float) -> dict[int, float]:
        """Returns activity scores for all faces (for debug)."""
        return {
            tid: float(np.mean([s for t, s in buf if time_start <= t <= time_end]))
            for tid, buf in self._activity.items()
            if any(time_start <= t <= time_end for t, _ in buf)
        }
