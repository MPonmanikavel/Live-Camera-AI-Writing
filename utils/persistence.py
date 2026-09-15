"""
utils/persistence.py — Save and load drawings.

Drawings are saved to ~/AirWriting/ as:
  <timestamp>.png              — flat canvas image
  <timestamp>_strokes.json     — stroke log for future re-rendering
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config


def _ensure_dir() -> Path:
    d = config.SAVE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_drawing(
    canvas_image: np.ndarray,
    strokes: list[list[tuple[int, int]]],
) -> Path:
    """
    Save the canvas image and stroke data to the save directory.

    Returns the path to the saved PNG.
    """
    save_dir = _ensure_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = save_dir / f"{ts}.png"
    json_path = save_dir / f"{ts}_strokes.json"

    # Save PNG
    cv2.imwrite(str(png_path), canvas_image)

    # Save stroke log
    stroke_data = {
        "timestamp": ts,
        "canvas_size": canvas_image.shape[:2],   # (height, width)
        "strokes": [
            [list(pt) for pt in stroke]
            for stroke in strokes
        ],
        "brush": {
            "color": list(config.PALETTE[config.PALETTE_KEYS[config.DEFAULT_COLOR_INDEX]]),
            "radius": config.DEFAULT_BRUSH_RADIUS,
        },
    }
    json_path.write_text(json.dumps(stroke_data, indent=2), encoding="utf-8")

    return png_path


def list_drawings() -> list[Path]:
    """Return all saved PNG files, newest first."""
    save_dir = config.SAVE_DIR
    if not save_dir.exists():
        return []
    return sorted(save_dir.glob("*.png"), reverse=True)


def load_drawing(png_path: Path) -> np.ndarray | None:
    """Load a saved canvas image.  Returns BGR array or None."""
    img = cv2.imread(str(png_path))
    return img
