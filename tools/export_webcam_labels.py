"""Export labelled webcam sessions (tools/label_webcam.py or the web app) as TipNet samples.

    python tools/export_webcam_labels.py webcam_labels/20260914_150000 --out dataset_cam --split train
    python tools/export_webcam_labels.py webcam_labels/20260914_160000 --out dataset_cam --split test

Every labelled photo is saved in the chosen view together with all the tips on the board.
Output: <out>/images/<split>/webcam_<session>__pXXXX.jpg and <out>/webcam_<session>.jsonl, in the
same format as tools/build_deepdarts.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import geometry as g  # noqa: E402
from darts_vader.labels import LABELLED, LabelSession  # noqa: E402
from darts_vader.tips.views import VIEWS, render_view, view_geometry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (contains labels.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--view", choices=VIEWS, default="camera")
    args = ap.parse_args()

    session = LabelSession(Path(args.session))
    if not session.path.exists():
        raise SystemExit(f"no labels.json in {session.dir}")
    name = session.dir.name
    out = Path(args.out)
    (out / "images" / args.split).mkdir(parents=True, exist_ok=True)
    written = skipped = tips_total = 0
    with open(out / f"webcam_{name}.jsonl", "w", encoding="utf-8") as f:
        for entry in session.photos:
            img = cv2.imread(str(session.dir / entry["image"])) if entry["status"] == LABELLED else None
            if img is None:
                skipped += 1
                continue
            board = session.board(entry)
            image, px_to_mm = render_view(img, board, args.view)
            tips_mm = np.array([t["tip_mm"] for t in entry["tips"]], float).reshape(-1, 2)
            file = f"webcam_{name}__p{entry['index']:04d}.jpg"
            cv2.imwrite(str(out / "images" / args.split / file), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            record = dict(
                image=f"images/{args.split}/{file}", source="webcam", session=f"webcam_{name}", split=args.split,
                photo=entry["index"], view=args.view, tilt=round(view_geometry(px_to_mm)[1], 1),
                tips_mm=np.round(tips_mm, 2).tolist(),
                tips_px=np.round(g.apply_homography(np.linalg.inv(px_to_mm), tips_mm), 1).tolist() if len(tips_mm) else [],
                scores=[g.score_model_point(*t, board.rings).label for t in tips_mm],
                px_to_mm=np.round(px_to_mm, 9).tolist(),
            )
            f.write(json.dumps(record) + "\n")
            written += 1
            tips_total += len(tips_mm)
    print(f"session {name}: exported {written} photos ({tips_total} tips, {args.split}, {args.view} view); "
          f"skipped {skipped} unlabelled or discarded -> {out / f'webcam_{name}.jsonl'}")


if __name__ == "__main__":
    main()
