"""
main.py — Live Air Writing AI
Full-featured entry point.

Gestures
--------
  ✍  Index only          → WRITE
  ✌  Index + middle      → HOVER
  🖐  Open palm           → ERASE
  ✊  Fist (hold ~0.7 s)  → CLEAR
  🤟  Index + pinky       → COLOR PICK  (palette appears; dwell to select)
  🤘  Index+middle+ring   → BRUSH SIZE  (pinch dist → size 2-20)
  🤙  Thumb + pinky       → RECOGNIZE   (release to OCR + math)
  👍  Thumb only          → UNDO

Keyboard
--------
  S         save    |  C  clear  |  Z  undo   |  F  fullscreen  |  Q/ESC quit
  1-8       colour  |  [  smaller brush  |  ]  larger brush
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

import config
from core.camera import Camera, CameraError
from core.hand_tracker import HandTracker
from core.gesture import GestureRecognizer, GestureState
from core.canvas import Canvas
from core.ocr_engine import recognize_async
from ui.renderer import Renderer
from utils.smoothing import LandmarkSmoother
from utils.persistence import save_drawing


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Air Writing AI")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--width",  type=int, default=config.FRAME_WIDTH)
    p.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    p.add_argument("--no-skeleton", action="store_true")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _show_banner(w: int, h: int) -> np.ndarray:
    banner = np.zeros((h + config.HUD_HEIGHT, w, 3), dtype=np.uint8)
    cx, cy = w // 2, (h + config.HUD_HEIGHT) // 2
    for i, (text, fs, col) in enumerate([
        ("Live Air Writing AI",              0.85, (0, 220, 255)),
        ("Initialising camera...",           0.52, (160, 160, 160)),
        ("Show your hand to the webcam.",    0.42, (120, 120, 120)),
    ]):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(banner, text,
                    (cx - tw // 2, cy - 30 + i * int(th * 2.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    return banner


def _save_notification(frame: np.ndarray, path: Path) -> None:
    msg = f"Saved: {path.name}"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    h, w = frame.shape[:2]
    px, py = (w - tw) // 2, 36
    cv2.rectangle(frame, (px-10, py-th-6), (px+tw+10, py+8), (20, 20, 20), -1)
    cv2.putText(frame, msg, (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 255, 100), 1, cv2.LINE_AA)


def _brush_from_pinch(pinch_px: float,
                      lo: float = 20, hi: float = 200) -> int:
    """Map thumb-index distance (pixels) to brush radius range."""
    t = (pinch_px - lo) / max(hi - lo, 1)
    t = max(0.0, min(1.0, t))
    return int(config.BRUSH_RADIUS_MIN +
               t * (config.BRUSH_RADIUS_MAX - config.BRUSH_RADIUS_MIN))


# ── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    draw_skeleton = not args.no_skeleton

    # ── State ─────────────────────────────────────────────────────────────────
    color_idx    = config.DEFAULT_COLOR_INDEX
    brush_radius = config.DEFAULT_BRUSH_RADIUS
    prev_state   = GestureState.IDLE

    # OCR
    ocr_text:         str      = ""
    math_result:      str | None = None
    ocr_frames_left:  int      = 0
    ocr_running:      bool     = False
    recognize_armed:  bool     = False   # True once shaka is held, fires on release

    # Undo: only trigger once per thumb-up gesture
    undo_armed: bool = False

    # Save notification
    save_notif_frames: int       = 0
    save_notif_path:   Path|None = None

    # FPS
    fps_smooth = 30.0
    prev_time  = time.perf_counter()

    print("\n[AIR WRITING] Live Air Writing AI -- starting up")
    print(f"   Camera : {args.camera}  |  {args.width}x{args.height}")
    print(f"   Saves  : {config.SAVE_DIR}\n")

    try:
        with Camera(args.camera, args.width, args.height) as cam:
            w, h = cam.width, cam.height

            cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(config.WINDOW_TITLE, w, h + config.HUD_HEIGHT)
            cv2.imshow(config.WINDOW_TITLE, _show_banner(w, h))
            cv2.waitKey(1)

            canvas      = Canvas(w, h)
            gesture_rec = GestureRecognizer(w, h)
            smoother    = LandmarkSmoother(alpha=config.SMOOTHING_ALPHA)
            renderer    = Renderer(w, h)

            with HandTracker() as tracker:
                print("[READY] Show your hand to start.\n")

                while True:
                    # ── 1. Capture ────────────────────────────────────────────
                    frame = cam.read()
                    if frame is None:
                        continue
                    frame = cv2.flip(frame, 1)

                    # ── 2. Hand detection ─────────────────────────────────────
                    result = tracker.process(frame, draw_skeleton=draw_skeleton)

                    if result is None:
                        smoother.reset()
                        gesture = gesture_rec.reset()
                        if prev_state == GestureState.WRITING:
                            canvas.end_stroke()
                        # Fire OCR if we were armed and hand disappeared
                        if recognize_armed and not ocr_running:
                            _trigger_ocr(canvas, ocr_running_ref := [False],
                                         _on_ocr_done(ocr_results := {}))
                        recognize_armed = False
                        undo_armed      = False
                        annotated = frame
                    else:
                        raw_lm, annotated = result
                        landmarks = smoother.update(raw_lm)
                        gesture   = gesture_rec.update(landmarks)
                        state     = gesture.state
                        cursor    = gesture.cursor

                        # ── Set pen color ──────────────────────────────────────
                        canvas.color  = config.EXTENDED_PALETTE[color_idx][1]
                        canvas.radius = brush_radius

                        # ── BRUSH SIZE: pinch distance → radius ───────────────
                        if state == GestureState.BRUSH_SIZE:
                            brush_radius = _brush_from_pinch(gesture.pinch_distance)

                        # ── COLOR PICK: palette dwell ─────────────────────────
                        if state == GestureState.COLOR_PICK:
                            color_idx = renderer.palette_ui.update(cursor, color_idx)
                        else:
                            renderer.palette_ui.reset_dwell()

                        # ── RECOGNIZE: arm on entry, fire on exit ─────────────
                        if state == GestureState.RECOGNIZE:
                            recognize_armed = True
                        elif recognize_armed and not ocr_running:
                            # Gesture released — trigger OCR
                            ocr_running = True
                            canvas_snap = canvas.get_image()

                            def _on_done(text: str, math: str | None) -> None:
                                nonlocal ocr_text, math_result, ocr_frames_left, ocr_running
                                ocr_text        = text if text else "(nothing recognized)"
                                math_result     = math if math else ""
                                ocr_frames_left = config.OCR_DISPLAY_FRAMES
                                ocr_running     = False
                                print(f"[OCR] {ocr_text}")
                                if math:
                                    print(f"[MATH] {math}")

                            recognize_async(canvas_snap, _on_done)
                            recognize_armed = False

                        # ── UNDO: fire once per gesture entry ─────────────────
                        if state == GestureState.UNDO and not undo_armed:
                            if canvas.undo():
                                print("[UNDO]")
                            undo_armed = True
                        elif state != GestureState.UNDO:
                            undo_armed = False

                        # ── WRITE / ERASE / CLEAR ─────────────────────────────
                        if state == GestureState.WRITING:
                            if prev_state != GestureState.WRITING:
                                canvas.begin_stroke(cursor)
                            else:
                                canvas.continue_stroke(cursor)

                        elif state == GestureState.ERASE:
                            canvas.erase(cursor)
                            if prev_state == GestureState.WRITING:
                                canvas.end_stroke()

                        elif state == GestureState.CLEAR:
                            canvas.clear()
                            if prev_state == GestureState.WRITING:
                                canvas.end_stroke()

                        elif state in (GestureState.HOVER, GestureState.IDLE,
                                       GestureState.COLOR_PICK,
                                       GestureState.BRUSH_SIZE,
                                       GestureState.RECOGNIZE,
                                       GestureState.UNDO):
                            if prev_state == GestureState.WRITING:
                                canvas.end_stroke()
                            canvas.lift_pen()

                        prev_state = state

                    # ── 3. FPS ────────────────────────────────────────────────
                    now = time.perf_counter()
                    fps_smooth = 0.1 * (1.0 / max(now - prev_time, 1e-6)) + 0.9 * fps_smooth
                    prev_time  = now

                    # ── 4. Composite + render ─────────────────────────────────
                    composite  = canvas.composite(annotated)
                    pen_color  = config.EXTENDED_PALETTE[color_idx][1]

                    if ocr_frames_left > 0:
                        ocr_frames_left -= 1

                    display = renderer.render(
                        composite_frame  = composite,
                        webcam_frame     = annotated,
                        gesture_result   = gesture,
                        fps              = fps_smooth,
                        pen_color        = pen_color,
                        brush_radius     = brush_radius,
                        color_idx        = color_idx,
                        ocr_text         = ocr_text,
                        math_result      = math_result or "",
                        ocr_frames_left  = ocr_frames_left,
                        undo_available   = canvas.has_undo,
                    )

                    if save_notif_frames > 0 and save_notif_path:
                        _save_notification(display, save_notif_path)
                        save_notif_frames -= 1

                    cv2.imshow(config.WINDOW_TITLE, display)

                    # ── 5. Keyboard ───────────────────────────────────────────
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord('q'), ord('Q'), 27):
                        print("Quitting.")
                        break

                    elif key in (ord('f'), ord('F')):
                        fs = renderer.toggle_fullscreen()
                        print(f"[MODE] {'FULLSCREEN' if fs else 'WINDOWED'}")

                    elif key in (ord('s'), ord('S')):
                        path = save_drawing(canvas.get_image(), canvas.strokes)
                        print(f"[SAVED] {path}")
                        save_notif_path   = path
                        save_notif_frames = 90

                    elif key in (ord('c'), ord('C')):
                        canvas.clear()
                        print("[CLEAR]")

                    elif key in (ord('z'), ord('Z')):
                        if canvas.undo():
                            print("[UNDO]")

                    # Colors 1-8
                    elif ord('1') <= key <= ord('8'):
                        color_idx = key - ord('1')
                        name = config.EXTENDED_PALETTE[color_idx][0]
                        print(f"[COLOR] {name}")

                    elif key == ord('['):
                        brush_radius = max(config.BRUSH_RADIUS_MIN, brush_radius - 1)
                        print(f"[BRUSH] {brush_radius}")

                    elif key == ord(']'):
                        brush_radius = min(config.BRUSH_RADIUS_MAX, brush_radius + 1)
                        print(f"[BRUSH] {brush_radius}")

    except CameraError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
