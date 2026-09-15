"""
ui/renderer.py — Single-panel composite display with all feature overlays.

Layout:
  ┌──────────────────────────────────────┐
  │  Live webcam + strokes (full panel)  │
  │  [Palette UI when COLOR_PICK]        │
  │  [OCR result card when active]       │
  │  [Brush-size preview circle]         │
  │  [Eraser radius circle]              │
  ├──────────────────────────────────────┤
  │             HUD bar                  │
  └──────────────────────────────────────┘
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from core.gesture import GestureState
from ui.hud import HUD, state_color
from ui.palette_ui import PaletteUI


class Renderer:
    def __init__(self, cam_width: int, cam_height: int) -> None:
        self._w   = cam_width
        self._h   = cam_height
        self._hud = HUD()
        self._palette_ui = PaletteUI(cam_width)
        self._fullscreen = False

    # ── fullscreen toggle ─────────────────────────────────────────────────────

    def toggle_fullscreen(self) -> bool:
        self._fullscreen = not self._fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, prop)
        return self._fullscreen

    @property
    def is_fullscreen(self) -> bool:
        return self._fullscreen

    @property
    def palette_ui(self) -> PaletteUI:
        return self._palette_ui

    # ── main render ───────────────────────────────────────────────────────────

    def render(
        self,
        composite_frame: np.ndarray,
        webcam_frame: np.ndarray,
        gesture_result,
        fps: float,
        pen_color: tuple[int, int, int],
        brush_radius: int,
        color_idx: int,
        ocr_text: str = "",
        math_result: str = "",
        ocr_frames_left: int = 0,
        undo_available: bool = False,
    ) -> np.ndarray:
        h, w = self._h, self._w
        hud_h = config.HUD_HEIGHT
        display = np.zeros((h + hud_h, w, 3), dtype=np.uint8)

        # ── Main view ─────────────────────────────────────────────────────────
        main = cv2.resize(composite_frame, (w, h))
        display[:h] = main

        state  = gesture_result.state
        cursor = gesture_result.cursor
        accent = state_color(state)

        # ── Eraser radius ring ────────────────────────────────────────────────
        if state == GestureState.ERASE:
            cx, cy = int(cursor[0]), int(cursor[1])
            cv2.circle(display, (cx, cy), config.ERASER_RADIUS,
                       (50, 50, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "ERASE", (cx - 22, cy - config.ERASER_RADIUS - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 50, 255), 1, cv2.LINE_AA)

        # ── Brush size preview ────────────────────────────────────────────────
        elif state == GestureState.BRUSH_SIZE:
            cx, cy = int(cursor[0]), int(cursor[1])
            cv2.circle(display, (cx, cy), brush_radius,
                       (255, 140, 30), 2, cv2.LINE_AA)
            cv2.putText(display, f"SIZE: {brush_radius}",
                        (cx - 28, cy - brush_radius - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 140, 30), 1, cv2.LINE_AA)

        # ── Color palette overlay ─────────────────────────────────────────────
        if state == GestureState.COLOR_PICK:
            self._palette_ui.draw(display, cursor, color_idx)

        # ── Cursor crosshair ─────────────────────────────────────────────────
        if state not in (GestureState.IDLE, GestureState.ERASE,
                         GestureState.BRUSH_SIZE):
            self._draw_cursor(display, cursor, accent, brush_radius * 3)

        # ── OCR result card ───────────────────────────────────────────────────
        if ocr_frames_left > 0 and ocr_text:
            self._draw_ocr_card(display, ocr_text, math_result, ocr_frames_left)

        # ── RECOGNIZE spinner ─────────────────────────────────────────────────
        if state == GestureState.RECOGNIZE and ocr_frames_left == 0:
            self._draw_centered_text(display, "Hold shaka...  release to OCR",
                                     (180, 50, 255))

        # ── UNDO flash ────────────────────────────────────────────────────────
        if state == GestureState.UNDO:
            self._draw_centered_text(display, "UNDO", (80, 200, 255))

        # ── Bottom hint ───────────────────────────────────────────────────────
        cv2.putText(display, "F:Fullscreen  Q:Quit", (6, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 90, 90), 1, cv2.LINE_AA)

        # ── HUD bar ───────────────────────────────────────────────────────────
        mode_label = "FULLSCREEN" if self._fullscreen else "LIVE"
        self._hud.draw(
            display[h:], state, fps, pen_color,
            brush_radius, mode_label, undo_available,
        )

        return display

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_cursor(
        display: np.ndarray,
        cursor_px: tuple[float, float],
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        cx, cy = int(cursor_px[0]), int(cursor_px[1])
        dh, dw = display.shape[:2]
        if 0 <= cy < dh and 0 <= cx < dw:
            cv2.circle(display, (cx, cy), max(radius, 4), color, 2, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 3, color, -1, cv2.LINE_AA)
            cv2.line(display, (cx-radius-5, cy), (cx-radius+2, cy), color, 1, cv2.LINE_AA)
            cv2.line(display, (cx+radius-2, cy), (cx+radius+5, cy), color, 1, cv2.LINE_AA)
            cv2.line(display, (cx, cy-radius-5), (cx, cy-radius+2), color, 1, cv2.LINE_AA)
            cv2.line(display, (cx, cy+radius-2), (cx, cy+radius+5), color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_ocr_card(
        display: np.ndarray,
        ocr_text: str,
        math_result: str,
        frames_left: int,
    ) -> None:
        h, w = display.shape[:2]
        alpha = min(1.0, frames_left / 30.0)   # fade out last 30 frames

        # Card dimensions
        card_w, card_h = min(w - 40, 560), 100 if not math_result else 130
        cx = (w - card_w) // 2
        cy = h // 2 - card_h // 2

        overlay = display.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.88 * alpha, display, 1 - 0.88 * alpha, 0, display)
        cv2.rectangle(display, (cx, cy), (cx + card_w, cy + card_h), (180, 50, 255), 1)

        # Title
        cv2.putText(display, "RECOGNIZED TEXT",
                    (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 50, 255), 1, cv2.LINE_AA)

        # OCR text (may be long — truncate)
        display_text = ocr_text[:55] + ("..." if len(ocr_text) > 55 else "")
        cv2.putText(display, display_text,
                    (cx + 10, cy + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)

        # Math result
        if math_result:
            cv2.putText(display, f"MATH:  {math_result}",
                        (cx + 10, cy + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 255, 150), 1, cv2.LINE_AA)

        # Progress bar (fades out)
        bar_pct = frames_left / config.OCR_DISPLAY_FRAMES
        cv2.rectangle(display,
                      (cx, cy + card_h - 4), (cx + card_w, cy + card_h),
                      (60, 60, 60), -1)
        cv2.rectangle(display,
                      (cx, cy + card_h - 4),
                      (cx + int(card_w * bar_pct), cy + card_h),
                      (180, 50, 255), -1)

    @staticmethod
    def _draw_centered_text(
        display: np.ndarray,
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        h, w = display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.70, 1)
        x = (w - tw) // 2
        y = h // 4

        overlay = display.copy()
        cv2.rectangle(overlay, (x - 14, y - th - 10), (x + tw + 14, y + 10),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
        cv2.putText(display, text, (x, y), font, 0.70, color, 1, cv2.LINE_AA)
