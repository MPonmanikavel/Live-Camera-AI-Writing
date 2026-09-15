"""
config.py — Central configuration for Live Air Writing AI.
"""
from pathlib import Path

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
TARGET_FPS: int = 30

# ── MediaPipe Hands ──────────────────────────────────────────────────────────
MP_MAX_HANDS: int = 1
MP_DETECTION_CONFIDENCE: float = 0.60
MP_TRACKING_CONFIDENCE: float = 0.50

# ── Gesture thresholds ───────────────────────────────────────────────────────
PINCH_THRESHOLD_NORM: float = 0.065
FIST_HOLD_FRAMES: int = 20
FINGER_UP_THRESHOLD: float = 0.02
# Dwell selection: frames index must hover over palette swatch to pick color
DWELL_FRAMES: int = 24                 # ~0.8 s at 30 fps

# ── Smoothing ────────────────────────────────────────────────────────────────
SMOOTHING_ALPHA: float = 0.70

# ── Canvas ───────────────────────────────────────────────────────────────────
CANVAS_BACKGROUND: tuple = (255, 255, 255)
ERASER_RADIUS: int = 35
DEFAULT_BRUSH_RADIUS: int = 6
BRUSH_RADIUS_MIN: int = 2
BRUSH_RADIUS_MAX: int = 20
UNDO_STACK_MAX: int = 20              # max undo history depth

# ── Extended Colour palette (BGR) ─────────────────────────────────────────────
EXTENDED_PALETTE: list[tuple[str, tuple]] = [
    ("Black",   (10,   10,  10)),
    ("White",   (240, 240, 240)),
    ("Red",     (30,   30, 220)),
    ("Green",   (30,  180,  50)),
    ("Blue",    (220,  60,  30)),
    ("Yellow",  (20,  220, 220)),
    ("Purple",  (200,  30, 150)),
    ("Orange",  (20,  140, 255)),
]
# Legacy dict for keyboard shortcuts 1-5
PALETTE: dict[str, tuple] = {name: color for name, color in EXTENDED_PALETTE[:5]}
PALETTE_KEYS = list(PALETTE.keys())
DEFAULT_COLOR_INDEX: int = 0

# ── Window / Display ─────────────────────────────────────────────────────────
WINDOW_TITLE: str = "Live Air Writing AI"
PANEL_GAP: int = 8
HUD_HEIGHT: int = 56               # slightly taller for more shortcuts

# ── Persistence ───────────────────────────────────────────────────────────────
SAVE_DIR: Path = Path.home() / "AirWriting"

# ── Skeleton / Hand point drawing ─────────────────────────────────────────────
FINGERTIP_COLORS: list[tuple] = [
    (0,   180, 255),   # Thumb  — orange
    (0,   255,  80),   # Index  — green  (writing finger)
    (255, 200,   0),   # Middle — blue
    (200,   0, 255),   # Ring   — magenta
    (0,   220, 255),   # Pinky  — yellow
]
LANDMARK_COLOR: tuple   = (200, 200, 200)
CONNECTION_COLOR: tuple = (120, 120, 120)
LANDMARK_RADIUS: int    = 5
FINGERTIP_RADIUS: int   = 10
CONNECTION_THICKNESS: int = 2

# ── HUD ──────────────────────────────────────────────────────────────────────
HUD_BG_COLOR: tuple    = (20, 20, 20)
HUD_TEXT_COLOR: tuple  = (220, 220, 220)
HUD_FONT_SCALE: float  = 0.48
HUD_THICKNESS: int     = 1

# ── OCR / Recognition overlay ────────────────────────────────────────────────
OCR_DISPLAY_FRAMES: int = 150          # frames to show result (~5 s at 30 fps)
OCR_CARD_ALPHA: float = 0.82
