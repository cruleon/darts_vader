"""Sanity check + benchmark: does the ONNX export find the same tips as the original torch
model, and how much faster/smaller is it on CPU? Reads a few labelled webcam photos (never
writes to the main project). Run after export_onnx.py.

    python lite/benchmark.py [--session webcam_labels/20260915_173022_301] [--n 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from darts_vader.board.detector import BoardState  # noqa: E402
from infer_onnx import TipNetOnnx  # noqa: E402


def load_photos(session_dir: Path, n: int):
    data = json.loads((session_dir / "labels.json").read_text())
    legacy = {"etichettata": "labelled"}
    photos = [p for p in data["photos"] if legacy.get(p.get("status"), p.get("status")) == "labelled"]
    for p in photos[:n]:
        frame = cv2.imread(str(session_dir / p["image"]))
        H = np.array(p["board_H"], float)
        board = BoardState(H, 1.0, 0.0, 0, False, rings=tuple(p["rings"]))
        yield p["image"], frame, board


def time_it(fn, *args, reps: int) -> tuple[float, object]:
    fn(*args)  # warm-up (session/model init, first-call overhead)
    t0 = time.perf_counter()
    result = None
    for _ in range(reps):
        result = fn(*args)
    return (time.perf_counter() - t0) / reps, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="a webcam_labels/<session> folder; default: the newest one")
    ap.add_argument("--n", type=int, default=6, help="photos to check")
    ap.add_argument("--reps", type=int, default=5, help="timing repeats per photo")
    args = ap.parse_args()

    if args.session:
        session_dir = ROOT / args.session
    else:
        sessions = sorted((ROOT / "webcam_labels").iterdir(), key=lambda p: p.stat().st_mtime)
        session_dir = [s for s in sessions if (s / "labels.json").exists()][-1]
    print(f"session: {session_dir.name}")

    view = (Path(__file__).parent / "model" / "view.txt").read_text().strip()
    onnx_fp32 = TipNetOnnx(Path(__file__).parent / "model" / "tipnet.onnx", view=view)
    onnx_int8 = TipNetOnnx(Path(__file__).parent / "model" / "tipnet_int8.onnx", view=view)

    import torch  # only needed for this reference comparison, not for the onnx path itself
    from darts_vader.tips.model import load_tipnet
    from darts_vader.tips.model import predict_tips as torch_predict_tips
    torch_model, _ = load_tipnet(ROOT / "models" / "tipnet.pt", device="cpu")

    t_torch = t_fp32 = t_int8 = 0.0
    n_photos = 0
    max_diff_fp32 = max_diff_int8 = 0.0
    for name, frame, board in load_photos(session_dir, args.n):
        n_photos += 1
        dt, ref = time_it(lambda: torch_predict_tips(torch_model, frame, board, view, device="cpu"), reps=args.reps)
        t_torch += dt
        dt, out_fp32 = time_it(lambda: onnx_fp32.predict_tips(frame, board), reps=args.reps)
        t_fp32 += dt
        dt, out_int8 = time_it(lambda: onnx_int8.predict_tips(frame, board), reps=args.reps)
        t_int8 += dt

        d1 = _match_diff(ref, out_fp32)
        d2 = _match_diff(ref, out_int8)
        max_diff_fp32 = max(max_diff_fp32, d1)
        max_diff_int8 = max(max_diff_int8, d2)
        print(f"  {name}: torch {len(ref)} tips, onnx-fp32 {len(out_fp32)} tips (max diff {d1:.2f} mm), "
              f"onnx-int8 {len(out_int8)} tips (max diff {d2:.2f} mm)")

    print()
    print(f"avg latency over {n_photos} photos, {args.reps} reps each (CPU):")
    print(f"  torch (fp32):  {t_torch / n_photos * 1000:.1f} ms")
    print(f"  onnx (fp32):   {t_fp32 / n_photos * 1000:.1f} ms")
    print(f"  onnx (int8):   {t_int8 / n_photos * 1000:.1f} ms")
    print(f"largest tip-position gap vs the torch model: fp32 {max_diff_fp32:.2f} mm, int8 {max_diff_int8:.2f} mm")


def _match_diff(ref: np.ndarray, other: np.ndarray) -> float:
    """Largest distance (mm) between each torch tip and its nearest onnx tip, in the same order
    torch found them. 0 if either is empty and the other isn't (a tip appeared/disappeared)."""
    if len(ref) != len(other):
        return float("nan")
    if len(ref) == 0:
        return 0.0
    diffs = []
    for x, y, _ in ref:
        d = np.hypot(other[:, 0] - x, other[:, 1] - y)
        diffs.append(float(d.min()))
    return max(diffs)


if __name__ == "__main__":
    main()
