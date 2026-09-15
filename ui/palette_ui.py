"""
ui/palette_ui.py — Floating color palette overlay.

Shown when the gesture state is COLOR_PICK (ILY sign: index + pinky).
Hover the index fingertip over a swatch for DWELL_FRAMES to select it.
A progress arc fills around the swatch as you dwell.
"""
from __future__ import annotations

import cv2
import numpy as np

import config

SWATCH_R   = 26     # swatch circle radius (px)
SWATCH_GAP = 16     # gap between swatches
PALETTE_Y  = 80     # center-Y of swatches from top


class PaletteUI:
    """Renders the 8-color palette and handles dwell-based color selection."""

    def __init__(self, frame_width: int) -> None:
        self._w = frame_width
        n = len(config.EXTENDED_PALETTE)
        total_w = n * SWATCH_R * 2 + (n - 1) * SWATCH_GAP
        self._start_x = (frame_width - total_w) // 2

        self._dwell_idx: int | None = None
        self._dwell_count: int = 0

    # ── centers ───────────────────────────────────────────────────────────────

    def swatch_centers(self) -> list[tuple[int, int]]:
        centers = []
        x = self._start_x + SWATCH_R
        for _ in config.EXTENDED_PALETTE:
            centers.append((x, PALETTE_Y))
            x += SWATCH_R * 2 + SWATCH_GAP
        return centers

    # ── hit test + dwell ──────────────────────────────────────────────────────

    def update(
        self,
        cursor: tuple[float, float],
        color_idx: int,
    ) -> int:
        """
        Call every frame when in COLOR_PICK mode.
        Returns the currently selected color index (may be updated by dwell).
        """
        hit = self._hit_test(cursor)

        if hit is not None:
            if hit == self._dwell_idx:
                self._dwell_count += 1
            else:
                self._dwell_idx   = hit
                self._dwell_count = 1
        else:
            self._dwell_idx   = None
            self._dwell_count = 0

        # Commit selection after full dwell
        if (
            self._dwell_idx is not None
            and self._dwell_count >= config.DWELL_FRAMES
        ):
            selected = self._dwell_idx
            self._dwell_idx   = None
            self._dwell_count = 0
            return selected

        return color_idx

    def reset_dwell(self) -> None:
        self._dwell_idx   = None
        self._dwell_count = 0

    # ── drawing ───────────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        cursor: tuple[float, float],
        selected_idx: int,
    ) -> None:
        centers = self.swatch_centers()
        n = len(centers)

        # ── Background bar ────────────────────────────────────────────────
        margin = 18
        x1 = self._start_x - margin
        y1 = PALETTE_Y - SWATCH_R - margin - 20
        x2 = centers[-1][0] + SWATCH_R + margin
        y2 = PALETTE_Y + SWATCH_R + margin + 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)

        # Title
        cv2.putText(frame, "COLOR PICKER  |  Hover to select",
                    (x1 + 6, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        hover_idx = self._hit_test(cursor)

        for i, (cx, cy) in enumerate(centers):
            name, color = config.EXTENDED_PALETTE[i]

            # Fill
            cv2.circle(frame, (cx, cy), SWATCH_R, color, -1, cv2.LINE_AA)

            # Selected ring
            if i == selected_idx:
                cv2.circle(frame, (cx, cy), SWATCH_R + 4, (255, 255, 255), 2, cv2.LINE_AA)

            # Hover highlight
            if i == hover_idx:
                cv2.circle(frame, (cx, cy), SWATCH_R + 7, (200, 200, 200), 1, cv2.LINE_AA)
                # Dwell progress arc
                progress = self._dwell_count / config.DWELL_FRAMES
                if progress > 0:
                    angle = int(360 * progress)
                    cv2.ellipse(
                        frame, (cx, cy),
                        (SWATCH_R + 7, SWATCH_R + 7),
                        -90, 0, angle,
                        (0, 255, 100), 2, cv2.LINE_AA,
                    )

            # Color name label below swatch
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(frame, name,
                        (cx - tw // 2, cy + SWATCH_R + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

    # ── internal ──────────────────────────────────────────────────────────────

    def _hit_test(self, cursor: tuple[float, float]) -> int | None:
        cx, cy = int(cursor[0]), int(cursor[1])
        for i, (sx, sy) in enumerate(self.swatch_centers()):
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist <= SWATCH_R + 8:
                return i
        return None
