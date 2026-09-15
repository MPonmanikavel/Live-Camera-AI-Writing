"""
ui/hud.py — Heads-up display bar.
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from core.gesture import GestureState

_STATE_LABEL: dict[GestureState, str] = {
    GestureState.IDLE:       "[ IDLE ]",
    GestureState.WRITING:    "[ WRITING ]",
    GestureState.HOVER:      "[ HOVER ]",
    GestureState.ERASE:      "[ ERASE ]",
    GestureState.CLEAR:      "[ CLEARING... ]",
    GestureState.COLOR_PICK: "[ COLOR PICK ]",
    GestureState.BRUSH_SIZE: "[ BRUSH SIZE ]",
    GestureState.RECOGNIZE:  "[ RECOGNIZING... ]",
    GestureState.UNDO:       "[ UNDO ]",
}

_STATE_COLOR: dict[GestureState, tuple[int, int, int]] = {
    GestureState.IDLE:       (100, 100, 100),
    GestureState.WRITING:    (50,  220,  50),
    GestureState.HOVER:      (50,  200, 255),
    GestureState.ERASE:      (50,   50, 255),
    GestureState.CLEAR:      (0,    60, 220),
    GestureState.COLOR_PICK: (200, 180,  50),
    GestureState.BRUSH_SIZE: (255, 140,  30),
    GestureState.RECOGNIZE:  (180,  50, 255),
    GestureState.UNDO:       (80,  200, 255),
}


def state_color(state: GestureState) -> tuple[int, int, int]:
    return _STATE_COLOR.get(state, (100, 100, 100))


class HUD:
    _LINE1 = "S:Save  C:Clear  F:Fullscreen  Z:Undo  1-8:Color  [/]:Brush  Q:Quit"
    _LINE2 = "Index=Write  Peace=Hover  Palm=Erase  Fist=Clear  ILY=Color  3Finger=Size  Shaka=OCR  Thumb=Undo"

    def draw(
        self,
        frame: np.ndarray,
        state: GestureState,
        fps: float,
        pen_color: tuple[int, int, int],
        brush_radius: int,
        mode_label: str = "LIVE",
        undo_available: bool = False,
    ) -> None:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = config.HUD_FONT_SCALE
        th   = config.HUD_THICKNESS
        accent = _STATE_COLOR.get(state, (100, 100, 100))

        # Background
        cv2.rectangle(frame, (0, 0), (w, h), config.HUD_BG_COLOR, -1)
        # Top accent line
        cv2.line(frame, (0, 0), (w, 0), accent, 2)

        # Row 1: state label + FPS + mode + swatch
        label = _STATE_LABEL.get(state, "IDLE")
        cv2.putText(frame, label, (8, 16), font, fs, accent, th + 1, cv2.LINE_AA)

        fps_str = f"FPS:{fps:.0f}  [{mode_label}]"
        cv2.putText(frame, fps_str, (8, 32), font, fs * 0.82,
                    config.HUD_TEXT_COLOR, th, cv2.LINE_AA)

        # Colour swatch
        sw_x = 230
        cv2.rectangle(frame, (sw_x, 4), (sw_x + 26, 22), pen_color, -1)
        cv2.rectangle(frame, (sw_x, 4), (sw_x + 26, 22), (160, 160, 160), 1)
        undo_col = (0, 220, 100) if undo_available else (80, 80, 80)
        cv2.putText(frame, f"sz:{brush_radius}  Z:undo",
                    (sw_x + 32, 16), font, fs * 0.78, undo_col, th, cv2.LINE_AA)

        # Row 1 shortcuts (right-aligned)
        (tw, _), _ = cv2.getTextSize(self._LINE1, font, fs * 0.68, th)
        cv2.putText(frame, self._LINE1, (w - tw - 6, 16),
                    font, fs * 0.68, config.HUD_TEXT_COLOR, th, cv2.LINE_AA)

        # Row 2: gesture guide
        cv2.putText(frame, self._LINE2, (6, 48),
                    font, fs * 0.58, (130, 130, 130), th, cv2.LINE_AA)
