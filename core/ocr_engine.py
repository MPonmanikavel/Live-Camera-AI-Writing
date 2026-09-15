"""
core/ocr_engine.py — Lazy EasyOCR + SymPy math recognition.

EasyOCR is loaded only on the first recognition trigger so the app starts
instantly.  Recognition runs in a background thread to keep the UI alive.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

import numpy as np
import cv2

# ── lazy globals ──────────────────────────────────────────────────────────────
_reader = None
_lock   = threading.Lock()


def _get_reader():
    """Return (and lazily initialise) the EasyOCR reader."""
    global _reader
    with _lock:
        if _reader is None:
            try:
                import easyocr                              # type: ignore
                print("[OCR] Loading EasyOCR model...")
                _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                print("[OCR] Model ready.")
            except ImportError:
                raise ImportError(
                    "EasyOCR not installed.  Run: "
                    ".venv\\Scripts\\python.exe -m pip install easyocr"
                )
    return _reader


# ── public helpers ────────────────────────────────────────────────────────────

def _preprocess(canvas_img: np.ndarray) -> np.ndarray:
    """
    Prepare canvas for OCR:
      1. Convert to grayscale.
      2. Invert (dark strokes on white → white strokes on dark) — EasyOCR
         works better with light text on dark backgrounds.
      3. Light Gaussian denoise.
    """
    gray   = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    # Threshold: strokes are dark (<200), background is white (≥200)
    _, bw  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.GaussianBlur(bw, (3, 3), 0)
    return denoised


def recognize_text(canvas_img: np.ndarray) -> str:
    """
    Synchronously run OCR on canvas_img (BGR, white bg, dark strokes).
    Returns the recognised text string (may be empty).
    """
    reader = _get_reader()
    processed = _preprocess(canvas_img)
    results = reader.readtext(processed, detail=0, paragraph=True)
    return " ".join(results).strip()


def evaluate_math(text: str) -> str | None:
    """
    Try to evaluate text as a math expression using SymPy.
    Returns a result string like '= 42' or None if not parseable.
    """
    try:
        import sympy                                       # type: ignore

        # Normalise common OCR artefacts
        expr = (
            text
            .replace("×", "*").replace("÷", "/")
            .replace("^", "**").replace("²", "**2")
            .replace("x", "*")                            # OCR often reads × as x
        )
        # Keep only valid math chars
        expr = re.sub(r"[^0-9+\-*/()=.\s]", "", expr).strip()
        if not expr:
            return None

        if "=" in expr:
            # Try to verify/solve equation
            lhs, rhs = expr.split("=", 1)
            diff = sympy.simplify(f"({lhs.strip()}) - ({rhs.strip()})")
            return f"= {diff}  (balance)"
        else:
            result = sympy.simplify(sympy.sympify(expr))
            return f"= {result}"

    except Exception:
        return None


def recognize_async(
    canvas_img: np.ndarray,
    callback: Callable[[str, str | None], None],
) -> threading.Thread:
    """
    Run OCR + math in a daemon thread.  Calls callback(text, math_result)
    when done.  Returns the thread so the caller can join if needed.
    """
    img_copy = canvas_img.copy()

    def _worker():
        try:
            text = recognize_text(img_copy)
            math = evaluate_math(text) if text else None
            callback(text, math)
        except Exception as exc:
            callback(f"[OCR error: {exc}]", None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
