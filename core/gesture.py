"""
core/gesture.py — Gesture state machine (8 gestures).

States
------
IDLE        — No hand / unrecognised pose.
WRITING     — Index finger only.           → Draw strokes.
HOVER       — Index + middle (peace).      → Move cursor, no draw.
ERASE       — Open palm (all 5 fingers).   → Erase under fingertip.
CLEAR       — Fist held FIST_HOLD_FRAMES.  → Clear canvas.
COLOR_PICK  — Index + pinky (ILY sign).    → Show color palette.
BRUSH_SIZE  — Index + middle + ring.       → Pinch distance → brush size.
RECOGNIZE   — Thumb + pinky (shaka).       → Trigger OCR / math.
UNDO        — Thumb only extended.         → Undo last stroke.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import config
from core.hand_tracker import HandTracker


class GestureState(Enum):
    IDLE       = auto()
    WRITING    = auto()
    HOVER      = auto()
    ERASE      = auto()
    CLEAR      = auto()
    COLOR_PICK = auto()
    BRUSH_SIZE = auto()
    RECOGNIZE  = auto()
    UNDO       = auto()


# Frames a gesture must be held before committing
_DEBOUNCE: dict[GestureState, int] = {
    GestureState.IDLE:       2,
    GestureState.WRITING:    1,
    GestureState.HOVER:      1,
    GestureState.ERASE:      2,
    GestureState.CLEAR:      config.FIST_HOLD_FRAMES,
    GestureState.COLOR_PICK: 3,
    GestureState.BRUSH_SIZE: 2,
    GestureState.RECOGNIZE:  8,   # hold shaka briefly to avoid accidental trigger
    GestureState.UNDO:       5,   # hold thumb-up briefly
}


@dataclass
class GestureResult:
    state: GestureState
    cursor: tuple[float, float] = (0.0, 0.0)
    pinch_distance: float = 0.0
    fingers_up: list[bool] = field(default_factory=lambda: [False] * 5)


class GestureRecognizer:
    """
    Converts 21 smoothed landmark positions → GestureResult.
    Call update(landmarks) every frame; reset() when hand disappears.
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self._w = frame_width
        self._h = frame_height
        self._state = GestureState.IDLE
        self._pending_state = GestureState.IDLE
        self._pending_count = 0

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, landmarks: list[tuple[float, float]]) -> GestureResult:
        fingers = HandTracker.fingers_up(landmarks)
        pinch   = HandTracker.pinch_distance(landmarks)
        cursor  = HandTracker.index_tip(landmarks)
        raw     = self._classify(fingers, pinch)
        state   = self._debounce(raw)
        return GestureResult(state=state, cursor=cursor,
                             pinch_distance=pinch, fingers_up=fingers)

    def reset(self) -> GestureResult:
        self._pending_state = GestureState.IDLE
        self._pending_count = 0
        self._state = GestureState.IDLE
        return GestureResult(state=GestureState.IDLE)

    @property
    def state(self) -> GestureState:
        return self._state

    # ── classification ────────────────────────────────────────────────────────

    def _classify(self, fingers: list[bool], pinch: float) -> GestureState:
        thumb, index, middle, ring, pinky = fingers
        n_up = sum(fingers)

        # ── Fist (CLEAR) ─────────────────────────────────────────────────────
        if not index and not middle and not ring and not pinky and not thumb:
            return GestureState.CLEAR

        # ── Open palm (ERASE) ────────────────────────────────────────────────
        if index and middle and ring and pinky and n_up >= 4:
            return GestureState.ERASE

        # ── Shaka — thumb + pinky (RECOGNIZE) ───────────────────────────────
        if thumb and pinky and not index and not middle and not ring:
            return GestureState.RECOGNIZE

        # ── Thumb only (UNDO) ────────────────────────────────────────────────
        if thumb and not index and not middle and not ring and not pinky:
            return GestureState.UNDO

        # ── ILY sign — index + pinky (COLOR_PICK) ───────────────────────────
        if index and pinky and not middle and not ring:
            return GestureState.COLOR_PICK

        # ── 3 fingers — index + middle + ring (BRUSH_SIZE) ───────────────────
        if index and middle and ring and not pinky:
            return GestureState.BRUSH_SIZE

        # ── Peace — index + middle (HOVER) ───────────────────────────────────
        if index and middle and not ring and not pinky:
            return GestureState.HOVER

        # ── Index only (WRITING) ──────────────────────────────────────────────
        if index and not middle and not ring and not pinky:
            return GestureState.WRITING

        return GestureState.IDLE

    def _debounce(self, raw: GestureState) -> GestureState:
        if raw == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw
            self._pending_count = 1
        if self._pending_count >= _DEBOUNCE.get(raw, 3):
            self._state = raw
        return self._state
