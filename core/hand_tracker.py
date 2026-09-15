"""
core/hand_tracker.py — MediaPipe Hand Landmarker (Tasks API).

Performance fixes vs original:
  * Uses real wall-clock timestamps (time.perf_counter) instead of a fixed
    33 ms tick — eliminates the tracking desync that caused sluggishness.
  * Lower detection/tracking confidence thresholds for snappier response.
  * Colorful per-finger landmark visualization.

Hand point visualization:
  - Skeleton connections drawn as grey lines.
  - Every joint drawn as a small white dot.
  - Each FINGERTIP drawn as a larger colored dot (one color per finger).
  - Index fingertip gets an extra bright ring — it's the writing point.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

import config

# ── Model path ────────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent.parent / "hand_landmarker.task"

# ── Landmark indices ──────────────────────────────────────────────────────────
FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
FINGER_PIPS = [3, 6, 10, 14, 18]   # thumb IP, then PIP joints

# All 21 connection pairs for skeleton drawing
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),      # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),      # index
    (0, 9),  (9, 10), (10,11), (11,12),      # middle
    (0, 13),(13, 14),(14, 15),(15, 16),      # ring
    (0, 17),(17, 18),(18, 19),(19, 20),      # pinky
    (5, 9), (9, 13),(13, 17),               # palm arch
]

# Fingertip landmark index → finger index (0–4)
_TIP_TO_FINGER = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}


class HandTracker:
    """
    Wraps mediapipe.tasks HandLandmarker for single-hand VIDEO-mode detection.

    Usage::
        tracker = HandTracker()
        result  = tracker.process(frame)
        if result:
            landmarks, annotated_frame = result
    """

    def __init__(
        self,
        max_hands: int = config.MP_MAX_HANDS,
        detection_confidence: float = config.MP_DETECTION_CONFIDENCE,
        tracking_confidence: float = config.MP_TRACKING_CONFIDENCE,
        model_path: Path = _MODEL_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found: {model_path}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                f"-OutFile {model_path}"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        # ── Real-time timestamp baseline ─────────────────────────────────────
        # MediaPipe VIDEO mode requires strictly monotonic timestamps in ms.
        # Using wall-clock time keeps them in sync with actual frame rate.
        self._start_time = time.perf_counter()

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── main API ─────────────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        draw_skeleton: bool = True,
    ) -> tuple[list[tuple[float, float]], np.ndarray] | None:
        """
        Detect hand landmarks in ``frame`` (BGR).

        Returns (landmarks_px, annotated_frame) or None if no hand found.
        landmarks_px — list of 21 (x, y) pixel coordinate tuples.
        annotated_frame — frame copy with skeleton + landmark points drawn.
        """
        h, w = frame.shape[:2]

        # ── Real wall-clock timestamp (milliseconds) ─────────────────────────
        # This is the critical fix: previously a fixed +33 ms tick was used,
        # which desynchronises from actual processing time and makes MediaPipe
        # tracking sluggish or confused.
        ts_ms = int((time.perf_counter() - self._start_time) * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        landmarks_px: list[tuple[float, float]] = [
            (lm.x * w, lm.y * h) for lm in hand
        ]

        if draw_skeleton:
            annotated = frame.copy()
            self._draw_hand(annotated, landmarks_px)
        else:
            annotated = frame

        return landmarks_px, annotated

    # ── colorful hand visualization ───────────────────────────────────────────

    def _draw_hand(
        self,
        frame: np.ndarray,
        lm: list[tuple[float, float]],
    ) -> None:
        """
        Draw skeleton connections, joint dots, and color-coded fingertip dots.
        """
        # 1. Connections (grey lines, drawn first so dots appear on top)
        for a, b in _HAND_CONNECTIONS:
            p1 = (int(lm[a][0]), int(lm[a][1]))
            p2 = (int(lm[b][0]), int(lm[b][1]))
            cv2.line(frame, p1, p2,
                     config.CONNECTION_COLOR, config.CONNECTION_THICKNESS, cv2.LINE_AA)

        # 2. All 21 joint dots (small, white)
        for idx, (x, y) in enumerate(lm):
            if idx not in _TIP_TO_FINGER:
                cv2.circle(frame, (int(x), int(y)),
                           config.LANDMARK_RADIUS, config.LANDMARK_COLOR, -1, cv2.LINE_AA)
                # Thin dark border for contrast
                cv2.circle(frame, (int(x), int(y)),
                           config.LANDMARK_RADIUS, (40, 40, 40), 1, cv2.LINE_AA)

        # 3. Fingertip dots — larger, per-finger color
        for tip_idx, finger_i in _TIP_TO_FINGER.items():
            x, y = int(lm[tip_idx][0]), int(lm[tip_idx][1])
            color = config.FINGERTIP_COLORS[finger_i]

            # Outer glow ring
            cv2.circle(frame, (x, y),
                       config.FINGERTIP_RADIUS + 4, color, 1, cv2.LINE_AA)
            # Filled dot
            cv2.circle(frame, (x, y),
                       config.FINGERTIP_RADIUS, color, -1, cv2.LINE_AA)
            # White center highlight
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 4. Index fingertip extra emphasis — double ring (writing point)
        ix, iy = int(lm[8][0]), int(lm[8][1])
        cv2.circle(frame, (ix, iy),
                   config.FINGERTIP_RADIUS + 8,
                   config.FINGERTIP_COLORS[1], 1, cv2.LINE_AA)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def index_tip(landmarks: list[tuple[float, float]]) -> tuple[float, float]:
        return landmarks[8]

    @staticmethod
    def thumb_tip(landmarks: list[tuple[float, float]]) -> tuple[float, float]:
        return landmarks[4]

    @staticmethod
    def fingers_up(landmarks: list[tuple[float, float]]) -> list[bool]:
        """Return [thumb, index, middle, ring, pinky] — True = extended."""
        up = []
        # Thumb: horizontal heuristic
        thumb_up = abs(landmarks[4][0] - landmarks[3][0]) > abs(
            landmarks[3][0] - landmarks[2][0]
        ) * 0.5
        up.append(thumb_up)
        # Four fingers: tip.y < pip.y
        for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            up.append(landmarks[tip_idx][1] < landmarks[pip_idx][1] - 5)
        return up

    @staticmethod
    def pinch_distance(landmarks: list[tuple[float, float]]) -> float:
        tx, ty = landmarks[4]
        ix, iy = landmarks[8]
        return float(np.hypot(tx - ix, ty - iy))
