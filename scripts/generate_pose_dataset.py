"""Genera un dataset sintetico in formato YOLO-pose per addestrare
``MLTipDetector`` (vedi ``scripts/train_pose_model.py``), senza hardware
reale ne' dati annotati a mano.

Tutto il dataset e' sintetico: nessuna validazione su dati reali finche'
non esiste un setup fisico (marker stampati, bersaglio, webcam) su cui
raccogliere un piccolo set di fine-tuning — stesso principio con cui il
progetto ha gia' rimandato la webcam finche' non serviva davvero.

Uso:
    uv run python scripts/generate_pose_dataset.py --num-samples 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dartvision.config import load_config
from dartvision.synthetic.pose_dataset import render_training_sample, to_yolo_pose_label

DATASET_YAML_TEMPLATE = """\
# Generato da scripts/generate_pose_dataset.py — dataset 100% sintetico.
path: {out_dir}
train: images/train
val: images/val
names:
  0: dart
kpt_shape: [1, 3]
"""


def _write_split(
    config, rng: np.random.Generator, out_dir: Path, split: str, num_samples: int, crop_size: int,
    max_darts_per_crop: int,
) -> None:
    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        sample = render_training_sample(
            config, rng, crop_size=crop_size, max_darts_per_crop=max_darts_per_crop
        )
        name = f"{split}_{i:06d}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), sample.image)

        lines = [to_yolo_pose_label(label, crop_size) for label in sample.labels]
        (labels_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--out-dir", default="data/pose_dataset")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--max-darts-per-crop", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    num_val = max(1, round(args.num_samples * args.val_fraction))
    num_train = max(1, args.num_samples - num_val)

    _write_split(config, rng, out_dir, "train", num_train, args.crop_size, args.max_darts_per_crop)
    _write_split(config, rng, out_dir, "val", num_val, args.crop_size, args.max_darts_per_crop)

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        DATASET_YAML_TEMPLATE.format(out_dir=out_dir.resolve()), encoding="utf-8"
    )

    print(f"Dataset scritto in {out_dir}: {num_train} train, {num_val} val.")
    print(f"dataset.yaml: {dataset_yaml}")


if __name__ == "__main__":
    main()
