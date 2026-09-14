"""Webcam preview and recording with mounting guidance.

    python tools/record_webcam.py --source 0
    python tools/record_webcam.py --source 0 --backend dshow --capture 1920x1080 --fourcc MJPG

The preview shows the detected board and placement hints: resolution on the board (px/mm),
viewing tilt, and whether the whole board is in the frame.
Keys: r start/stop recording (recordings/webcam_<date>.avi, MJPG) | s save a photo | q quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import BoardState, DartboardDetector, geometry as g  # noqa: E402
from darts_vader.board.overlay import draw_board, draw_panel  # noqa: E402
from darts_vader.camera import BACKENDS, open_camera  # noqa: E402

OK_COLOR, WARN_COLOR = (0, 220, 0), (0, 165, 255)


def placement_hints(state: BoardState, shape) -> list[tuple[str, tuple[int, int, int]]]:
    """Mounting hints as (text, colour) pairs."""
    h, w = shape[:2]
    ring = state.to_image(g.model_points(g.R_DOUBLE_OUT, np.arange(0, 360, 5))).astype(np.float32)
    (_, _), (a, b), _ = cv2.fitEllipse(ring)
    major, minor = max(a, b), min(a, b)
    tilt = float(np.degrees(np.arccos(np.clip(minor / major, 0, 1))))
    px_mm = major / 2 / g.R_DOUBLE_OUT
    outer = state.to_image(g.model_points(g.R_BOARD, np.arange(0, 360, 5)))
    inside = bool(np.all((outer[:, 0] >= 0) & (outer[:, 0] < w) & (outer[:, 1] >= 0) & (outer[:, 1] < h)))
    hints = [(f"resolution on the board {px_mm:.1f} px/mm", OK_COLOR if px_mm >= 2.0 else WARN_COLOR),
             (f"viewing tilt {tilt:.0f} deg", OK_COLOR if 20 <= tilt <= 50 else WARN_COLOR),
             ("whole board in frame" if inside else "board CUT OFF: widen the framing", OK_COLOR if inside else WARN_COLOR)]
    if px_mm < 2.0:
        hints.append(("move the webcam closer or raise the resolution", WARN_COLOR))
    if tilt < 20:
        hints.append(("almost frontal view: darts may hide their own tips", WARN_COLOR))
    elif tilt > 50:
        hints.append(("very steep view: the board gets squashed", WARN_COLOR))
    return hints


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--backend", choices=tuple(BACKENDS), default="msmf")
    ap.add_argument("--capture", default="3264x2448", help="resolution requested from the webcam")
    ap.add_argument("--fourcc", default="", help="pixel format requested from the webcam, e.g. MJPG with dshow")
    ap.add_argument("--detect-every", type=int, default=5, help="run board detection once every N frames")
    ap.add_argument("--out", default="recordings")
    args = ap.parse_args()

    try:
        cap = open_camera(args.source, args.backend, args.capture, args.fourcc or None)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = DartboardDetector()
    state, writer, rec_path, i = None, None, None, 0
    t_last, fps = time.perf_counter(), 0.0
    cv2.namedWindow("webcam", cv2.WINDOW_AUTOSIZE)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("no frames from the webcam")
            break
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6) if fps else 1.0 / max(now - t_last, 1e-6)
        t_last = now
        if writer is not None:
            writer.write(frame)
        if i % args.detect_every == 0:
            state = detector.process(frame)
        i += 1

        scale = min(1.0, 900 / frame.shape[0], 1600 / frame.shape[1])
        vis = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        lines = [(f"{frame.shape[1]}x{frame.shape[0]}  {fps:.0f} fps", 0.55, (255, 255, 255), 1)]
        if state is not None:
            draw_board(vis, state.scaled(scale))
            lines += [(text, 0.6, color, 2) for text, color in placement_hints(state, frame.shape)]
        else:
            lines.append(("board not detected: frame it fully and light it well", 0.6, (80, 80, 255), 2))
        if writer is not None:
            lines.append(("REC", 0.6, (80, 80, 255), 2))
        draw_panel(vis, lines, 8, 8)
        draw_panel(vis, [("r record/stop | s photo | q quit", 0.5, (220, 220, 220), 1)], 8, vis.shape[0] - 34)
        cv2.imshow("webcam", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            if writer is None:
                rec_path = out / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                rec_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 15.0
                writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"MJPG"), rec_fps, frame.shape[1::-1])
                print(f"recording to {rec_path} ({rec_fps:.0f} fps)")
            else:
                writer.release()
                writer = None
                print(f"recording saved: {rec_path}")
        elif key == ord("s"):
            path = out / f"photo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"photo saved: {path}")
    if writer is not None:
        writer.release()
        print(f"recording saved: {rec_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
