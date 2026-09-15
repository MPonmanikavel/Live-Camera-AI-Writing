"""
core/camera.py — Webcam capture wrapper.

Usage:
    with Camera(index=0, width=640, height=480) as cam:
        frame = cam.read()   # returns BGR numpy array or None
"""
from __future__ import annotations

import cv2
import numpy as np
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class Camera:
    """Thin wrapper around cv2.VideoCapture with context-manager support."""

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the camera device.  Raises CameraError on failure."""
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)   # CAP_DSHOW is faster on Windows
        if not cap.isOpened():
            # Fallback without backend hint
            cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise CameraError(
                f"Cannot open camera {self._index}.  "
                "Check the index or whether another application is using it."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # minimal internal buffering → lower latency
        self._cap = cap

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── frame access ─────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Read the next frame.  Returns BGR array or None if unavailable."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def width(self) -> int:
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
