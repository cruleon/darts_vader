"""Create the orientation-normalised copy (``camera_canon`` view) of a ``camera`` view dataset.

Every crop is rotated about its centre so that the side of the board closest to the camera is at
the bottom (see darts_vader/tips/views.py). Homographies are stored in the samples (``px_to_mm``),
so boards do not need to be detected again.

    python tools/make_canonical_dataset.py --src dataset_cam --dst dataset_canon

Run it again after exporting new sessions to --src: --dst is rebuilt from scratch.
Output: the same jsonl files and image layout, with view="camera_canon" and the ``tilt`` field.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from darts_vader.board import geometry as g  # noqa: E402
from darts_vader.tips.views import canonical_rotation, view_geometry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="dataset_cam")
    ap.add_argument("--dst", default="dataset_canon")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0, total = time.perf_counter(), 0
    for jsonl in sorted(src.glob("*.jsonl")):
        records = [json.loads(line) for line in open(jsonl, encoding="utf-8")]
        with open(dst / jsonl.name, "w", encoding="utf-8") as f:
            for r in records:
                if r.get("view", "rect") != "camera":
                    raise SystemExit(f"{jsonl.name}: camera view samples required (found {r.get('view')!r})")
                img = cv2.imread(str(src / r["image"]))
                size = img.shape[1]
                P = np.array(r["px_to_mm"], float)
                _, tilt = view_geometry(P)
                R = canonical_rotation(P, size)
                out = dst / r["image"]
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), cv2.warpAffine(img, R[:2], (size, size), flags=cv2.INTER_LINEAR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                tips_px = np.array(r["tips_px"], float).reshape(-1, 2)
                r.update(view="camera_canon", tilt=round(tilt, 1), px_to_mm=np.round(P @ np.linalg.inv(R), 9).tolist(),
                         tips_px=np.round(g.apply_homography(R, tips_px), 1).tolist() if len(tips_px) else [])
                f.write(json.dumps(r) + "\n")
        total += len(records)
        print(f"  {jsonl.name}: {len(records)} samples ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"done: {total} samples in {dst}")


if __name__ == "__main__":
    main()
