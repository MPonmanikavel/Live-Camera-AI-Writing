"""
core/canvas.py — Stroke accumulation, compositing, and undo stack.
"""
from __future__ import annotations

import cv2
import numpy as np

import config


class Canvas:
    """Persistent drawing surface with undo support."""

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height
        self._image = self._blank()
        self._strokes: list[list[tuple[int, int]]] = []
        self._current_stroke: list[tuple[int, int]] = []
        self._undo_stack: list[np.ndarray] = []   # snapshots before each stroke
        self.color: tuple[int, int, int] = config.EXTENDED_PALETTE[config.DEFAULT_COLOR_INDEX][1]
        self.radius: int = config.DEFAULT_BRUSH_RADIUS
        self._prev_pt: tuple[int, int] | None = None

    # ── drawing ───────────────────────────────────────────────────────────────

    def begin_stroke(self, pt: tuple[float, float]) -> None:
        # Save state for undo before starting a new stroke
        self._push_undo()
        self._current_stroke = [self._to_px(pt)]
        self._prev_pt = self._to_px(pt)
        self._draw_dot(self._to_px(pt))

    def continue_stroke(self, pt: tuple[float, float]) -> None:
        cur_px = self._to_px(pt)
        self._current_stroke.append(cur_px)
        if self._prev_pt is not None:
            self._draw_segment(self._prev_pt, cur_px)
        self._prev_pt = cur_px

    def end_stroke(self) -> None:
        if self._current_stroke:
            self._strokes.append(self._current_stroke)
        self._current_stroke = []
        self._prev_pt = None

    def lift_pen(self) -> None:
        self._prev_pt = None

    def erase(self, pt: tuple[float, float], radius: int = config.ERASER_RADIUS) -> None:
        px = self._to_px(pt)
        cv2.circle(self._image, px, radius, config.CANVAS_BACKGROUND, -1, cv2.LINE_AA)
        self._prev_pt = None

    def clear(self) -> None:
        self._push_undo()
        self._image = self._blank()
        self._strokes.clear()
        self._current_stroke.clear()
        self._prev_pt = None

    def undo(self) -> bool:
        """Restore previous state. Returns True if something was undone."""
        if not self._undo_stack:
            return False
        self._image = self._undo_stack.pop()
        if self._strokes:
            self._strokes.pop()
        self._current_stroke = []
        self._prev_pt = None
        return True

    # ── compositing ───────────────────────────────────────────────────────────

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay strokes onto webcam frame.
        Stroke pixels → fully opaque. White background → transparent (webcam shows).
        """
        bg   = np.array(config.CANVAS_BACKGROUND, dtype=np.uint8)
        mask = np.any(self._image != bg, axis=2)    # True where strokes exist
        if frame.shape[:2] != (self._h, self._w):
            frame = cv2.resize(frame, (self._w, self._h))
        result = frame.copy()
        result[mask] = self._image[mask]
        return result

    def get_image(self) -> np.ndarray:
        return self._image.copy()

    # ── cursor overlay ────────────────────────────────────────────────────────

    def draw_cursor(self, target, pt, state_color, radius=10):
        px = self._to_px(pt)
        cv2.circle(target, px, radius, state_color, 2, cv2.LINE_AA)
        cv2.circle(target, px, 3,      state_color, -1, cv2.LINE_AA)

    @property
    def strokes(self):
        return list(self._strokes)

    @property
    def has_undo(self) -> bool:
        return len(self._undo_stack) > 0

    # ── internals ─────────────────────────────────────────────────────────────

    def _blank(self) -> np.ndarray:
        return np.full((self._h, self._w, 3), config.CANVAS_BACKGROUND, dtype=np.uint8)

    def _push_undo(self) -> None:
        if len(self._undo_stack) >= config.UNDO_STACK_MAX:
            self._undo_stack.pop(0)
        self._undo_stack.append(self._image.copy())

    def _to_px(self, pt: tuple[float, float]) -> tuple[int, int]:
        x = int(np.clip(pt[0], 0, self._w - 1))
        y = int(np.clip(pt[1], 0, self._h - 1))
        return (x, y)

    def _draw_dot(self, px: tuple[int, int]) -> None:
        cv2.circle(self._image, px, self.radius, self.color, -1, cv2.LINE_AA)

    def _draw_segment(self, p0: tuple[int, int], p1: tuple[int, int]) -> None:
        cv2.line(self._image, p0, p1, self.color, self.radius * 2, cv2.LINE_AA)
        cv2.circle(self._image, p0, self.radius, self.color, -1, cv2.LINE_AA)
        cv2.circle(self._image, p1, self.radius, self.color, -1, cv2.LINE_AA)
