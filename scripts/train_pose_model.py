"""Addestra il modello YOLO-pose usato da ``MLTipDetector`` sul dataset
sintetico generato da ``scripts/generate_pose_dataset.py``.

NON e' pensato per girare su questa macchina di sviluppo (nessuna GPU
NVIDIA disponibile qui): va eseguito sulla macchina con GPU, poi si
copia manualmente il file dei pesi risultante (``best.pt``) nel path
indicato da ``ml_detection.model_path`` in ``config/board_config.yaml``
(default ``models/dart_tip_pose.pt``).

Richiede l'extra opzionale 'ml' (non installato di default):
    uv sync --extra ml

Uso:
    uv run python scripts/train_pose_model.py --data data/pose_dataset/dataset.yaml \
        --epochs 150 --device 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pose_dataset/dataset.yaml")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Checkpoint di partenza (pretrained).")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=320, help="Deve combaciare con --crop-size usato in generate_pose_dataset.py.")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=25, help="Early stopping (epoche senza miglioramento).")
    parser.add_argument(
        "--device", default="0", help="'0' per la prima GPU CUDA, 'cpu' per forzare la CPU."
    )
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--name", default="dartvision_tip")
    parser.add_argument(
        "--copy-to",
        default="models/dart_tip_pose.pt",
        help="Dove copiare best.pt al termine (path atteso da ml_detection.model_path).",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics non installato: esegui 'uv sync --extra ml' prima del training."
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining completato. Pesi migliori: {best_weights}")

    if args.copy_to:
        destination = Path(args.copy_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, destination)
        print(f"Copiati in: {destination}")


if __name__ == "__main__":
    main()
