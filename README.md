# Live Air Writing AI

Touchless, gesture-based writing via webcam. No physical contact required.

---

## Setup (one-time)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
python main.py                       # default webcam, 640×480
python main.py --camera 1            # secondary webcam
python main.py --width 1280 --height 720
python main.py --no-skeleton         # hide hand landmark overlay
```

---

## Gesture Controls

| Gesture | How to do it | Effect |
|---------|-------------|--------|
| ✍️ **Write** | Raise index finger only | Draw strokes |
| ✌️ **Hover** | Raise index + middle | Move cursor, no drawing |
| 🖐 **Erase** | Open palm (all fingers up) | Erase under fingertip |
| ✊ **Clear** | Hold fist for ~1 second | Clear entire canvas |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `S` | Save drawing to `~/AirWriting/` |
| `C` | Clear canvas instantly |
| `1` – `5` | Switch pen colour (Black, Red, Green, Blue, Yellow) |
| `[` / `]` | Decrease / increase brush size |
| `Q` / `ESC` | Quit |

---

## Output Files

Drawings are saved to `~/AirWriting/` as:
- `<timestamp>.png` — flat canvas image
- `<timestamp>_strokes.json` — stroke log for re-rendering

---

## Project Layout

```
air/
├── main.py              Entry point + event loop
├── config.py            All tuneable constants
├── requirements.txt
├── core/
│   ├── camera.py        Webcam capture
│   ├── hand_tracker.py  MediaPipe landmark detection
│   ├── gesture.py       Gesture state machine
│   └── canvas.py        Stroke rendering + compositing
├── ui/
│   ├── renderer.py      Two-panel display layout
│   └── hud.py           HUD bar (mode, FPS, controls)
└── utils/
    ├── smoothing.py     Landmark jitter removal
    └── persistence.py  Save / load drawings
```
