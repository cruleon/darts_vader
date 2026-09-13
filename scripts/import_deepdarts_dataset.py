"""Converte il dataset reale DeepDarts (immagini + labels.pkl, scaricati
manualmente) nello stesso formato YOLO-pose usato da
``scripts/generate_pose_dataset.py``, cosi' i due dataset si possono
mescolare in training (Ultralytics accetta piu' percorsi in
``train:``/``val:`` nel dataset.yaml).

Il dataset NON e' incluso in questo repository: licenza CC-BY, va
scaricato manualmente con un account IEEE DataPort gratuito da
https://ieee-dataport.org/open-access/deepdarts-dataset (file
`labels_pkl.zip` e `cropped_images.zip`). Citazione:
William McNally, "DeepDarts Dataset", IEEE Dataport, 2021,
doi:10.21227/05e7-xs69 — vedi anche https://arxiv.org/abs/2105.09880.

Uso:
    uv run python scripts/import_deepdarts_dataset.py \
        --labels-pkl training_data_IEEE/labels_pkl/labels.pkl \
        --images-dir training_data_IEEE/cropped_images/800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dartvision.config import load_config
from dartvision.synthetic.deepdarts_import import load_rows, training_samples_from_row
from dartvision.synthetic.pose_dataset import to_yolo_pose_label

DATASET_YAML_TEMPLATE = """\
# Generato da scripts/import_deepdarts_dataset.py — dati REALI (DeepDarts,
# McNally 2021, CC-BY: https://ieee-dataport.org/open-access/deepdarts-dataset).
path: {out_dir}
train: images/train
val: images/val
names:
  0: dart
kpt_shape: [1, 3]
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--labels-pkl", default="training_data_IEEE/labels_pkl/labels.pkl")
    parser.add_argument("--images-dir", default="training_data_IEEE/cropped_images/800")
    parser.add_argument("--out-dir", default="data/deepdarts_pose_dataset")
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--box-size-mm", type=float, default=45.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="Processa solo le prime N righe (debug/smoke test)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    rows = load_rows(args.labels_pkl)
    if args.limit is not None:
        rows = rows[: args.limit]

    # Split per IMMAGINE, non per ritaglio: altrimenti piu' crop della
    # stessa foto finirebbero sia in train che in val (data leakage).
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    num_val = max(1, round(len(indices) * args.val_fraction))
    val_indices = set(indices[:num_val].tolist())

    counts = {"train": 0, "val": 0}
    skipped_missing_image = 0
    skipped_bad_homography = 0

    for i, row in enumerate(rows):
        split = "val" if i in val_indices else "train"
        image_path = images_dir / row.img_folder / row.img_name
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_missing_image += 1
            continue

        try:
            samples = training_samples_from_row(
                image, row, config, rng, crop_size=args.crop_size, box_size_mm=args.box_size_mm
            )
        except cv2.error:
            skipped_bad_homography += 1
            continue

        images_out = out_dir / "images" / split
        labels_out = out_dir / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for j, sample in enumerate(samples):
            name = f"{row.img_folder}_{Path(row.img_name).stem}_{j}"
            cv2.imwrite(str(images_out / f"{name}.jpg"), sample.image)
            lines = [to_yolo_pose_label(label, args.crop_size) for label in sample.labels]
            (labels_out / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(DATASET_YAML_TEMPLATE.format(out_dir=out_dir.resolve()), encoding="utf-8")

    print(
        f"Righe lette: {len(rows)}, immagini mancanti: {skipped_missing_image}, "
        f"omografie fallite: {skipped_bad_homography}"
    )
    print(f"Campioni scritti in {out_dir}: {counts['train']} train, {counts['val']} val.")
    print(f"dataset.yaml: {dataset_yaml}")


if __name__ == "__main__":
    main()
