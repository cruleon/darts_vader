"""Convert the DeepDarts dataset into TipNet training samples.

For every selected image the script
- detects the board and checks that the four labelled calibration points fall where expected
  (outer double wire at the 5|20, 17|3, 8|11 and 13|6 boundaries);
- saves the image in the chosen view (see darts_vader/tips/views.py);
- writes the tips in board millimetres and in pixels of the saved image, their scores and the
  ``px_to_mm`` matrix of the saved image.

    python tools/build_deepdarts.py --out dataset_cam --view camera
    python tools/build_deepdarts.py --out dataset_rect --view rect

Sessions are split into train/val/test so that a session never appears in two splits;
otherwise almost identical images would end up in both training and test.
Output: <out>/images/<split>/*.jpg and <out>/deepdarts.jsonl (one line per image).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import DartboardDetector, geometry as g  # noqa: E402
from darts_vader.tips.views import VIEWS, render_view, view_geometry  # noqa: E402

CAL_ANGLES = (-9.0, 171.0, 261.0, 81.0)
CAL_RADIUS = 171.5  # measured: the points are clicked on the outer double wire, ~1.5 mm beyond 170 mm
MAX_CAL_ERR_MM = 5.0


def session_date(name: str) -> datetime:
    _, month, day, year = name.split("_")[:4]
    return datetime(int(year), int(month), int(day))


def assign_splits(sessions) -> dict[str, str]:
    """About 70/15/15 by session, spreading val and test over the whole collection period."""
    ordered = sorted(sessions, key=lambda s: (session_date(s), s))
    return {s: "val" if i % 7 == 3 else "test" if i % 7 == 6 else "train" for i, s in enumerate(ordered)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="training_data/labels.pkl")
    ap.add_argument("--images", default="training_data/cropped_images/800")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", choices=VIEWS, default="camera")
    ap.add_argument("--d1-per-session", type=int, default=40, help="d1 images per session (0 = none)")
    ap.add_argument("--limit", type=int, default=0, help="maximum images per session (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_pickle(args.labels)
    df["subset"] = df["img_folder"].str[:2]
    parts = [df[df["subset"] == "d2"]]
    if args.d1_per_session > 0:
        d1 = df[df["subset"] == "d1"].sample(frac=1.0, random_state=args.seed)
        parts.append(d1.groupby("img_folder").head(args.d1_per_session))
    selected = pd.concat(parts)
    if args.limit:
        selected = selected.groupby("img_folder").head(args.limit)

    splits = {}
    for subset in ("d1", "d2"):
        splits.update(assign_splits(df.loc[df["subset"] == subset, "img_folder"].unique()))

    out = Path(args.out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
    stats, tips_per_split = Counter(), Counter()
    t0 = time.perf_counter()
    with open(out / "deepdarts.jsonl", "w", encoding="utf-8") as f:
        for n, row in enumerate(selected.itertuples(index=False), start=1):
            img = cv2.imread(str(Path(args.images) / row.img_folder / row.img_name))
            if img is None:
                stats["unreadable image"] += 1
                continue
            state = DartboardDetector().process(img)
            if state is None:
                stats["board not found"] += 1
                continue
            h, w = img.shape[:2]
            pts_mm = state.to_model(np.array(row.xy, float) * [w, h])
            cal_err = np.linalg.norm(pts_mm[:4] - g.model_points(CAL_RADIUS, CAL_ANGLES), axis=1)
            if cal_err.max() > MAX_CAL_ERR_MM:
                stats["inconsistent calibration"] += 1
                continue
            tips_mm = pts_mm[4:]
            image, px_to_mm = render_view(img, state, args.view)
            tips_px = g.apply_homography(np.linalg.inv(px_to_mm), tips_mm) if len(tips_mm) else np.zeros((0, 2))
            split = splits[row.img_folder]
            name = f"{row.img_folder}__{Path(row.img_name).stem}.jpg"
            cv2.imwrite(str(out / "images" / split / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            record = dict(
                image=f"images/{split}/{name}", source=f"deepdarts_{row.subset}", session=row.img_folder, split=split,
                view=args.view, tilt=round(view_geometry(px_to_mm)[1], 1),
                tips_mm=np.round(tips_mm, 2).tolist(), tips_px=np.round(tips_px, 1).tolist(),
                scores=[g.score_model_point(*t, state.rings).label for t in tips_mm],
                px_to_mm=np.round(px_to_mm, 9).tolist(),
                board_rms_mm=round(float(state.rms_mm), 3), cal_err_mm=round(float(cal_err.mean()), 2),
            )
            f.write(json.dumps(record) + "\n")
            stats[split] += 1
            tips_per_split[split] += len(tips_mm)
            if n % 250 == 0:
                print(f"  {n}/{len(selected)} images ({time.perf_counter() - t0:.0f} s)", flush=True)

    print(f"{len(selected)} images selected, processed in {time.perf_counter() - t0:.0f} s ({args.view} view)")
    used_sessions = set(selected["img_folder"])
    for split in ("train", "val", "test"):
        sessions = {s for s, sp in splits.items() if sp == split and s in used_sessions}
        print(f"  {split:5s}: {stats[split]} images, {tips_per_split[split]} tips, {len(sessions)} sessions")
    for reason in ("board not found", "inconsistent calibration", "unreadable image"):
        if stats[reason]:
            print(f"  skipped ({reason}): {stats[reason]}")
    print(f"annotations: {out / 'deepdarts.jsonl'}")


if __name__ == "__main__":
    main()
