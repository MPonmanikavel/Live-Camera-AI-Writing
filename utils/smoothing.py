"""
utils/smoothing.py — Exponential smoothing for (x, y) coordinate streams.

Reduces MediaPipe landmark jitter without introducing perceptible lag.
"""
from __future__ import annotations

import numpy as np


class ExponentialSmoother:
    """
    Per-channel weighted exponential moving average.

    new_value = α * raw + (1 - α) * previous
    α closer to 1.0  →  less smoothing, faster response
    α closer to 0.0  →  more smoothing, slower response
    """

    def __init__(self, alpha: float = 0.45, shape: tuple[int, ...] = (2,)) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._state: np.ndarray | None = None
        self._shape = shape

    def update(self, value: np.ndarray | tuple | list) -> np.ndarray:
        """Feed a new raw value; returns the smoothed estimate."""
        v = np.asarray(value, dtype=float)
        if self._state is None:
            self._state = v.copy()
        else:
            self._state = self._alpha * v + (1.0 - self._alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        """Clear history (e.g., when the hand disappears from frame)."""
        self._state = None

    @property
    def value(self) -> np.ndarray | None:
        return self._state.copy() if self._state is not None else None


class LandmarkSmoother:
    """
    Applies an ExponentialSmoother independently to each of the 21 hand landmarks.
    """

    def __init__(self, alpha: float = 0.45, n_landmarks: int = 21) -> None:
        self._smoothers = [ExponentialSmoother(alpha=alpha, shape=(2,)) for _ in range(n_landmarks)]

    def update(self, landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        landmarks: list of (x, y) pairs in pixel coordinates.
        Returns smoothed list of same shape.
        """
        out: list[tuple[float, float]] = []
        for i, pt in enumerate(landmarks):
            smoothed = self._smoothers[i].update(pt)
            out.append((float(smoothed[0]), float(smoothed[1])))
        return out

    def reset(self) -> None:
        for s in self._smoothers:
            s.reset()
