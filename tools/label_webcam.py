"""Label dart tips on webcam photos taken at the end of each turn.

    python tools/label_webcam.py
    python tools/label_webcam.py --session 20260914_150000        # resume an existing session

1. The board is locked automatically. Click the sector of the 20 and press Enter to confirm:
   photos are disabled until the orientation is confirmed.
2. Throw three darts and press Space: the photo is frozen on screen.
3. The model proposes tips (numbered yellow circles): press a to accept them all; right-click
   removes a proposal or a saved tip. For missing or inaccurate tips click near the tip: a zoom
   window opens, click where the tip enters the board. A green bar confirms every saved tip.
4. Enter saves the turn and goes back to the live view (pull the darts and throw again).

Frozen photo: Enter save turn | a accept proposals | right-click remove | u undo last tip |
              k discard photo | q quit
Live view:    Space take photo | r re-lock the board and ask for the 20 again | q quit

Every photo re-fits the board on the photo itself while keeping the confirmed orientation, so
tips stay accurate even if the webcam moves slightly. Everything is saved immediately to
<out>/<session>/ (photo_XXXX.jpg + labels.json, see darts_vader/labels.py).
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
from darts_vader.board.overlay import draw_board, draw_panel, label_box  # noqa: E402
from darts_vader.camera import BACKENDS, FrameSource  # noqa: E402
from darts_vader.labels import DISCARDED, LABELLED, NO_TIPS, PENDING, LabelSession, tip_record  # noqa: E402

ZOOM_HALF, ZOOM = 60, 5
GREEN, RED, ORANGE, GRAY = (0, 160, 0), (0, 0, 200), (0, 140, 255), (90, 90, 90)
PROPOSAL_THRESHOLD = 0.3
DUPLICATE_MM = 8.0  # a proposal this close to a saved tip is the same dart
SEARCHING, CALIBRATING, LIVE, PHOTO = "searching", "calibrating", "live", "photo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--backend", choices=tuple(BACKENDS), default="msmf")
    ap.add_argument("--capture", default="3264x2448")
    ap.add_argument("--fourcc", default="")
    ap.add_argument("--out", default="webcam_labels")
    ap.add_argument("--session", help="session name (to resume an existing one)")
    ap.add_argument("--display-height", type=int, default=900)
    ap.add_argument("--model", default="models/tipnet.pt", help="model used for tip proposals (empty = none)")
    args = ap.parse_args()

    tip_model, tip_view, device, predict_tips = None, "camera", "cpu", None
    if args.model and Path(args.model).exists():
        import torch

        from darts_vader.tips.model import load_tipnet, predict_tips

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tip_model, tip_view = load_tipnet(args.model, device)
        print(f"tip proposals from {args.model}")

    session = LabelSession(Path(args.out) / (args.session or time.strftime("%Y%m%d_%H%M%S")),
                           dict(source=args.source, backend=args.backend, capture=args.capture))
    if session.photos:
        print(f"resuming {session.path}: {len(session.photos)} photos")

    frames = FrameSource(args.source, args.backend, args.capture, args.fourcc or None)
    banner = {"text": "locking the board", "color": GRAY, "until": time.monotonic() + 5}

    def notify(text: str, color=GREEN, seconds: float = 4.0) -> None:
        banner.update(text=text, color=color, until=time.monotonic() + seconds)
        print(("[OK] " if color == GREEN else "[--] ") + text, flush=True)

    detector = DartboardDetector()
    board: BoardState | None = None  # orientation confirmed by the user
    candidate: BoardState | None = None  # locked board, orientation still to confirm
    lock_count = lost = 0
    mode = SEARCHING
    photo = None
    ui = {"main": None, "right": None, "zoom": None, "zoom_click": None, "zoom_mouse": None}

    def on_main(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["main"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            ui["right"] = (x, y)

    def on_zoom(event, x, y, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            ui["zoom_mouse"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            ui["zoom_click"] = (x, y)

    def close_zoom() -> None:
        ui["zoom"], ui["zoom_click"] = None, None
        try:
            cv2.destroyWindow("zoom")
        except cv2.error:
            pass

    cv2.namedWindow("webcam", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("webcam", on_main)
    while True:
        ok, frame = frames.read()
        if not ok:
            print(f"no frames from the webcam{': ' + frames.error if frames.error else ''}")
            break
        base = photo["frame"] if mode == PHOTO else frame
        scale = min(1.0, args.display_height / base.shape[0], 1600 / base.shape[1])

        # ------------------------------------------------------------ board lock and tracking
        if mode == SEARCHING:
            detected = detector.process(frame)
            lock_count = lock_count + 1 if detected is not None and detected.confidence >= 0.7 else 0
            if lock_count >= 5:
                candidate, mode, lost = detected, CALIBRATING, 0
                notify("CLICK THE SECTOR OF THE 20, then press Enter to confirm", ORANGE, 30)
        elif mode in (CALIBRATING, LIVE):
            # the board is tracked frame by frame (small webcam shifts or rotations) while the
            # chosen orientation is kept: the 20 can never jump to another sector
            detected = detector.process(frame)
            if detected is not None:
                if mode == CALIBRATING:
                    candidate = detected.aligned_to(candidate)
                else:
                    board = detected.aligned_to(board)
                lost = 0
            else:
                lost += 1

        # ------------------------------------------------------------ clicks
        if ui["main"] is not None:
            click = np.array(ui["main"]) / scale
            if mode == CALIBRATING:
                theta = g.polar(*candidate.to_model([click])[0])[1]
                candidate = candidate.rotated(g.sector_index(theta))
                notify("20 set where you clicked: check the numbers and press Enter", ORANGE, 30)
            elif mode == PHOTO:
                ui["zoom"], ui["zoom_mouse"] = dict(center=click), None
                cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback("zoom", on_zoom)
        ui["main"] = None

        if ui["right"] is not None and mode == PHOTO:  # remove the closest proposal or saved tip
            shown, pos, entry = photo["board"].scaled(scale), np.array(ui["right"], float), photo["entry"]
            items = [("proposal", i, t["tip_mm"]) for i, t in enumerate(photo["proposals"])] + \
                    [("tip", i, t["tip_mm"]) for i, t in enumerate(entry["tips"])]
            dist = [np.linalg.norm(shown.to_image([m])[0] - pos) for _, _, m in items]
            if dist and min(dist) <= 25:
                kind, i, _ = items[int(np.argmin(dist))]
                if kind == "proposal":
                    notify(f"proposal {photo['proposals'].pop(i)['score']} removed", ORANGE)
                else:
                    tip = entry["tips"].pop(i)
                    if not entry["tips"]:
                        entry["status"] = PENDING
                    session.save()
                    notify(f"tip {tip['score']} removed - saved", ORANGE)
            else:
                notify("nothing to remove near the click", ORANGE, 2)
        ui["right"] = None

        if ui["zoom_click"] is not None and ui["zoom"] is not None and mode == PHOTO:
            cx, cy = np.round(ui["zoom"]["center"]).astype(int)
            tip_px = np.array([cx - ZOOM_HALF, cy - ZOOM_HALF], float) + np.array(ui["zoom_click"]) / ZOOM
            photo_board, entry = photo["board"], photo["entry"]
            tip_mm = photo_board.to_model([tip_px])[0]
            hit = g.score_model_point(*tip_mm, photo_board.rings)
            entry["tips"].append(tip_record(tip_mm, photo_board, hit.label, "click"))
            # a proposal close to the clicked tip is no longer needed
            photo["proposals"] = [p for p in photo["proposals"]
                                  if np.linalg.norm(np.array(p["tip_mm"]) - tip_mm) > DUPLICATE_MM]
            entry["status"] = LABELLED
            session.save()
            notify(f"TIP {len(entry['tips'])} SAVED: {hit.label} - click the next one or press Enter to close the turn")
            close_zoom()

        # ------------------------------------------------------------ drawing
        vis = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        current = photo["board"] if mode == PHOTO else (candidate if mode == CALIBRATING else board)
        if current is not None:
            shown = current.scaled(scale)
            draw_board(vis, shown)
            if mode == PHOTO:
                for k, t in enumerate(photo["proposals"], start=1):
                    p = shown.to_image([t["tip_mm"]])[0].round().astype(int)
                    cv2.circle(vis, tuple(p.tolist()), 13, (0, 255, 255), 2, cv2.LINE_AA)
                    label_box(vis, f"{k}? {t['score']}", (p[0] + 34, p[1] + 26), 0.55, (0, 255, 255), 1)
                for t in photo["entry"]["tips"]:
                    p = shown.to_image([t["tip_mm"]])[0].round().astype(int)
                    cv2.circle(vis, tuple(p.tolist()), 9, (0, 255, 0), 2, cv2.LINE_AA)
                    label_box(vis, t["score"], (p[0] + 30, p[1] - 22), 0.6, (0, 255, 0), 2)
        status = {
            SEARCHING: "LOCKING THE BOARD: keep it fully in frame",
            CALIBRATING: "ORIENTATION: click the sector of the 20, then Enter | r re-lock",
            LIVE: "LIVE: throw three darts and press Space | r re-lock | q quit",
            PHOTO: "PHOTO: a accept proposals | click missing tip | right-click remove | Enter save | k discard",
        }[mode]
        lines = []
        if mode in (CALIBRATING, LIVE) and lost >= 10:
            lines.append(("BOARD LOST: frame it again (the confirmed orientation is kept)", 0.55, (80, 80, 255), 2))
        tips_total = sum(len(p["tips"]) for p in session.photos)
        lines += [(status, 0.55, (0, 255, 255), 1),
                  (f"labelled turns {session.count(LABELLED)}   tips {tips_total}", 0.6, (255, 255, 255), 2)]
        draw_panel(vis, lines, 8, 8)
        if time.monotonic() < banner["until"]:
            h = vis.shape[0]
            cv2.rectangle(vis, (0, h - 46), (vis.shape[1], h), banner["color"], -1)
            cv2.putText(vis, banner["text"], (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("webcam", vis)

        if ui["zoom"] is not None:
            if cv2.getWindowProperty("zoom", cv2.WND_PROP_VISIBLE) < 1:
                ui["zoom"] = None
            else:
                src = photo["frame"]
                H, W = src.shape[:2]
                cx, cy = np.round(ui["zoom"]["center"]).astype(int)
                x0, y0 = cx - ZOOM_HALF, cy - ZOOM_HALF
                crop = np.zeros((2 * ZOOM_HALF, 2 * ZOOM_HALF, 3), np.uint8)
                sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + 2 * ZOOM_HALF), min(H, y0 + 2 * ZOOM_HALF)
                crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = src[sy0:sy1, sx0:sx1]
                big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
                if ui["zoom_mouse"] is not None:
                    mx, my = ui["zoom_mouse"]
                    cv2.line(big, (mx, 0), (mx, big.shape[0]), (0, 255, 255), 1)
                    cv2.line(big, (0, my), (big.shape[1], my), (0, 255, 255), 1)
                draw_panel(big, [("click where the tip enters the board", 0.5, (0, 255, 255), 1)], 4, 4, pad=5)
                cv2.imshow("zoom", big)

        # ------------------------------------------------------------ keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r") and mode in (LIVE, CALIBRATING):
            board, candidate, lock_count, mode = None, None, 0, SEARCHING
            detector.reset()
            notify("re-locking the board", ORANGE)
        elif mode == CALIBRATING and key in (13, 10):
            board, mode = candidate, LIVE
            notify("orientation confirmed: throw three darts and press Space", GREEN, 6)
        elif mode == LIVE and key == ord(" "):
            snap = frame.copy()
            fresh = DartboardDetector().process(snap)
            photo_board = fresh.aligned_to(board) if fresh is not None and fresh.confidence >= 0.6 else board
            entry = session.add_photo(snap, photo_board)
            proposals = []
            if tip_model is not None:
                tips = predict_tips(tip_model, snap, photo_board, tip_view, PROPOSAL_THRESHOLD, max_tips=3, device=device)
                proposals = [dict(tip_mm=[round(float(x), 2), round(float(y), 2)], confidence=round(float(c), 3),
                                  score=g.score_model_point(x, y, photo_board.rings).label)
                             for x, y, c in tips if np.hypot(x, y) <= g.R_BOARD]
            photo, mode = dict(frame=snap, entry=entry, board=photo_board, proposals=proposals,
                               n_proposed=len(proposals), warned=False), PHOTO
            notify(f"photo {entry['index']}: {len(proposals)} proposals - a accepts, right-click removes, "
                   "click adds; then Enter", ORANGE, 8)
        elif mode == PHOTO:
            entry = photo["entry"]
            if key in (13, 10) and len(entry["tips"]) < photo["n_proposed"] and not photo["warned"]:
                photo["warned"] = True
                notify(f"WARNING: the model sees {photo['n_proposed']} darts but you saved {len(entry['tips'])}. "
                       "Check the green circles, then press Enter again to confirm", RED, 10)
            elif key in (13, 10):
                if not entry["tips"]:
                    entry["status"] = NO_TIPS
                session.save()
                close_zoom()
                mode = LIVE
                notify(f"turn {entry['index']} saved: {' '.join(t['score'] for t in entry['tips']) or 'no tips'}", GREEN, 5)
            elif key == ord("u"):
                if entry["tips"]:
                    tip = entry["tips"].pop()
                    if not entry["tips"]:
                        entry["status"] = PENDING
                    session.save()
                    notify(f"removed tip {tip['score']}", ORANGE)
                else:
                    notify("no tip to undo", ORANGE)
            elif key == ord("a"):
                added = []
                for t in photo["proposals"]:
                    if any(np.linalg.norm(np.array(t["tip_mm"]) - np.array(s["tip_mm"])) <= DUPLICATE_MM
                           for s in entry["tips"]):
                        continue
                    entry["tips"].append(tip_record(t["tip_mm"], photo["board"], t["score"], "proposal", t["confidence"]))
                    added.append(t["score"])
                photo["proposals"] = []
                if entry["tips"]:
                    entry["status"] = LABELLED
                session.save()
                notify(f"accepted {len(added)} proposals: {' '.join(added) or '-'} - saved; Enter closes the turn", GREEN)
            elif key == ord("k"):
                session.discard(entry)
                close_zoom()
                mode = LIVE
                notify(f"photo {entry['index']} discarded", ORANGE)

    if session.photos:
        session.save()
    frames.stop()
    cv2.destroyAllWindows()
    print(f"session {session.dir}: {session.count(LABELLED)} labelled turns, "
          f"{sum(len(p['tips']) for p in session.photos if p['status'] != DISCARDED)} tips")


if __name__ == "__main__":
    main()
