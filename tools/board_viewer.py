"""Live dartboard detection viewer: draws the detected sectors over a webcam, stream or video.

    python tools/board_viewer.py --source 0                          # webcam
    python tools/board_viewer.py --source http://192.168.1.20:8080/video
    python tools/board_viewer.py --source match.mp4 --save annotated.mp4
    python tools/board_viewer.py --source synthetic                  # virtual camera, no board needed

Keys:  q/Esc quit | Space pause | r reset | w rectified view | m colour mask | s screenshot
Mouse: hover a segment to see its score | left click prints it | right click marks the 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import DartboardDetector, DetectorConfig  # noqa: E402
from darts_vader.board.overlay import draw_board, put_text, rectified_view  # noqa: E402
from darts_vader.board.synthetic import SyntheticVideo  # noqa: E402
from darts_vader.camera import BACKENDS, FrameSource  # noqa: E402

WINDOW = "DARTS VADER - board viewer"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index, stream URL, video file or 'synthetic'")
    ap.add_argument("--backend", choices=tuple(BACKENDS), default="msmf")
    ap.add_argument("--capture", help="resolution requested from the webcam, e.g. 1920x1080")
    ap.add_argument("--process-width", type=int, default=960, help="frames are processed at this width")
    ap.add_argument("--display-height", type=int, default=900)
    ap.add_argument("--no-smooth", action="store_true", help="disable temporal smoothing")
    ap.add_argument("--save", help="write the annotated video to this .mp4 file")
    args = ap.parse_args()

    source = SyntheticVideo() if args.source == "synthetic" else args.source
    frames = FrameSource(source, args.backend, args.capture)
    cfg = DetectorConfig(process_width=args.process_width)
    if args.no_smooth:
        cfg.smoothing = 0.0
    detector = DartboardDetector(cfg)

    mouse = {"pos": None, "left": None, "right": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse["pos"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            mouse["left"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse["right"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    frame, state, writer = None, None, None
    paused, show_rect, show_mask = False, False, False
    proc_ms, fps, last_t, seen_loops = 0.0, 0.0, None, 0
    while True:
        if not paused or frame is None:
            ok, new_frame = frames.read()
            if not ok:
                if frames.error:
                    print(f"source error: {frames.error}")
                break
            if frames.loops != seen_loops:  # a video file restarted
                seen_loops = frames.loops
                detector.reset()
            frame = new_frame
            t0 = time.perf_counter()
            state = detector.process(frame)
            dt = (time.perf_counter() - t0) * 1000
            proc_ms = 0.9 * proc_ms + 0.1 * dt if proc_ms else dt
            if last_t is not None:
                now_fps = 1.0 / max(1e-6, t0 - last_t)
                fps = 0.9 * fps + 0.1 * now_fps if fps else now_fps
            last_t = t0

        h, w = frame.shape[:2]
        scale = min(1.0, args.display_height / h, 1600 / w)
        vis = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()

        if mouse["right"] is not None and state is not None:
            detector.set_sector20(np.array(mouse["right"]) / scale)
            state = detector.state
        mouse["right"] = None

        if state is not None:
            shown = state.scaled(scale)
            draw_board(vis, shown, mouse["pos"])
            if mouse["left"] is not None:
                hit = shown.hit(mouse["left"])
                print(f"{hit.label:>5} = {hit.score}")
            status = f"{'TRACKING' if state.tracked else 'LOCKED'}  conf {state.confidence:.2f}  rms {state.rms_mm:.2f} mm"
            color = (0, 255, 0)
        else:
            status, color = "SEARCHING FOR THE BOARD...", (0, 0, 255)
        mouse["left"] = None
        put_text(vis, status, (10, 25), 0.55, color)
        put_text(vis, f"{proc_ms:.0f} ms/frame  |  {fps:.0f} fps  |  {w}x{h}", (10, 50), 0.55, (255, 255, 255))
        if paused:
            put_text(vis, "PAUSED", (10, 75), 0.6, (0, 255, 255))

        cv2.imshow(WINDOW, vis)
        if show_rect and state is not None:
            cv2.imshow("rectified", rectified_view(frame, state))
        if show_mask and "mask" in detector.debug:
            cv2.imshow("colour mask", detector.debug["mask"])
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, vis.shape[1::-1])
            writer.write(vis)

        key = cv2.waitKey(30 if paused else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            detector.reset()
            state = None
        elif key == ord("w"):
            show_rect = not show_rect
            if not show_rect:
                cv2.destroyWindow("rectified")
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("colour mask")
        elif key == ord("s"):
            Path("screenshots").mkdir(exist_ok=True)
            path = Path("screenshots") / f"{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), vis)
            print(f"saved {path}")

    frames.stop()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
