"""Genera un video mp4 sintetico (bersaglio + marker ArUco + freccette finte)
per sviluppare e verificare la pipeline SENZA hardware reale.

Il bersaglio e i marker sono disegnati su un unico "piano verita'"
(coerente con la geometria di ``config/board_config.yaml``), poi
proiettati con un'unica homography verso una vista camera plausibile:
essendo bersaglio e marker fisicamente coplanari nel setup reale,
questo e' esattamente il modello che la calibrazione dovra' invertire.

Uso:
    uv run python scripts/generate_synthetic_video.py
    uv run python scripts/generate_synthetic_video.py --dart 0,0 --dart 20,-140
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dartvision.config import AppConfig, load_config
from dartvision.synthetic.board_renderer import paste_markers, render_plane_image

BACKGROUND_COLOR = (70, 70, 70)
COLOR_DART = (20, 20, 20)


def draw_dart(plane_img: np.ndarray, config: AppConfig, x_mm: float, y_mm: float) -> None:
    cx, cy = config.rectified_plane.mm_to_px(x_mm, y_mm)
    radius_px = max(2, round(2.5 / config.rectified_plane.mm_per_px))
    cv2.circle(plane_img, (round(cx), round(cy)), radius_px, COLOR_DART, -1)
    cv2.circle(plane_img, (round(cx), round(cy)), radius_px + 1, (0, 0, 0), 1)


def build_plane_to_camera_homography(plane_size: int, camera_size: tuple[int, int]) -> np.ndarray:
    """Homography fissa che simula una webcam che guarda il bersaglio di lato/dall'alto."""
    w, h = camera_size
    src = np.float32(
        [[0, 0], [plane_size, 0], [plane_size, plane_size], [0, plane_size]]
    )
    dst = np.float32(
        [
            [0.24 * w, 0.08 * h],
            [0.86 * w, 0.14 * h],
            [0.95 * w, 0.92 * h],
            [0.12 * w, 0.85 * h],
        ]
    )
    return cv2.getPerspectiveTransform(src, dst)


def render_camera_frame(
    plane_img: np.ndarray,
    homography: np.ndarray,
    camera_size: tuple[int, int],
    noise_std: float,
) -> np.ndarray:
    w, h = camera_size
    warped = cv2.warpPerspective(plane_img, homography, (w, h))
    coverage_mask = cv2.warpPerspective(
        np.full(plane_img.shape[:2], 255, dtype=np.uint8), homography, (w, h)
    )

    frame = np.full((h, w, 3), BACKGROUND_COLOR, dtype=np.uint8)
    frame[coverage_mask > 0] = warped[coverage_mask > 0]

    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame


def _parse_mm_pair(value: str) -> tuple[float, float]:
    x_str, y_str = value.split(",")
    return float(x_str), float(y_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--output", default="data/videos/synthetic_calibration.mp4")
    parser.add_argument("--num-frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument(
        "--dart",
        action="append",
        default=[],
        metavar="X_MM,Y_MM",
        help="Coordinate (mm, origine=centro bersaglio) di un impatto da "
        "far comparire durante il video. Ripetibile fino a 3 volte.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    camera_size = (args.camera_width, args.camera_height)

    plane_img = render_plane_image(config)
    paste_markers(plane_img, config)
    homography = build_plane_to_camera_homography(config.rectified_plane.size_px, camera_size)

    darts_mm = [_parse_mm_pair(v) for v in args.dart]
    dart_schedule: dict[int, tuple[float, float]] = {}
    if darts_mm:
        start = args.num_frames // 4
        step = max(1, (args.num_frames - start) // (len(darts_mm) + 1))
        for i, coords in enumerate(darts_mm):
            dart_schedule[start + step * (i + 1)] = coords

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, camera_size)

    darts_log = []
    for frame_idx in range(args.num_frames):
        if frame_idx in dart_schedule:
            x_mm, y_mm = dart_schedule[frame_idx]
            draw_dart(plane_img, config, x_mm, y_mm)
            darts_log.append({"appears_at_frame": frame_idx, "x_mm": x_mm, "y_mm": y_mm})

        frame = render_camera_frame(plane_img, homography, camera_size, args.noise_std)
        writer.write(frame)

    writer.release()

    ground_truth = {
        "config_path": str(args.config),
        "camera_size": list(camera_size),
        "plane_size_px": config.rectified_plane.size_px,
        "plane_to_camera_homography": homography.tolist(),
        "darts": darts_log,
        "note": (
            "plane_to_camera_homography e' fornita SOLO come verita' di "
            "riferimento per validare la calibrazione nei test; la pipeline "
            "reale la ricalcola sempre da zero dai marker rilevati."
        ),
    }
    gt_path = output_path.with_suffix(".ground_truth.json")
    gt_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    print(f"Video sintetico scritto in: {output_path}")
    print(f"Ground truth scritta in:   {gt_path}")
    if darts_log:
        print(f"Freccette simulate: {darts_log}")


if __name__ == "__main__":
    main()
