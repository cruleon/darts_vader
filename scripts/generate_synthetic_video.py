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

from dartvision.calibration.aruco_detector import resolve_dictionary
from dartvision.config import AppConfig, load_config

BACKGROUND_COLOR = (70, 70, 70)
COLOR_BLACK = (35, 35, 35)
COLOR_CREAM = (215, 215, 230)
COLOR_RED = (40, 40, 195)
COLOR_GREEN = (60, 130, 60)
COLOR_DART = (20, 20, 20)


def render_plane_image(config: AppConfig) -> np.ndarray:
    """Disegna bersaglio + marker sul piano "verita'" (vista dall'alto)."""
    plane = config.rectified_plane
    board = config.board
    size = plane.size_px
    img = np.full((size, size, 3), 60, dtype=np.uint8)

    center = tuple(round(v) for v in plane.mm_to_px(0.0, 0.0))
    mm_per_px = plane.mm_per_px

    def r_px(r_mm: float) -> int:
        return max(1, round(r_mm / mm_per_px))

    sector_width = 360.0 / board.sector_count
    ring_bands = [
        (board.double_outer_radius_mm, True),
        (board.double_inner_radius_mm, False),
        (board.triple_outer_radius_mm, True),
        (board.triple_inner_radius_mm, False),
    ]

    for i in range(board.sector_count):
        center_angle = board.sector0_offset_deg + i * sector_width
        start_cv = (center_angle - sector_width / 2) - 90
        end_cv = (center_angle + sector_width / 2) - 90
        parity = i % 2
        for radius_mm, is_multiplier_ring in ring_bands:
            if is_multiplier_ring:
                color = COLOR_RED if parity == 0 else COLOR_GREEN
            else:
                color = COLOR_BLACK if parity == 0 else COLOR_CREAM
            cv2.ellipse(
                img, center, (r_px(radius_mm), r_px(radius_mm)), 0,
                start_cv, end_cv, color, -1,
            )

    cv2.circle(img, center, r_px(board.outer_bull_radius_mm), COLOR_GREEN, -1)
    cv2.circle(img, center, r_px(board.inner_bull_radius_mm), COLOR_RED, -1)
    return img


def _paste_clipped(dst: np.ndarray, patch: np.ndarray, top_left_x: int, top_left_y: int) -> None:
    """Incolla ``patch`` su ``dst`` ritagliando ai bordi se necessario."""
    h, w = patch.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    x0, y0 = max(0, top_left_x), max(0, top_left_y)
    x1, y1 = min(dst_w, top_left_x + w), min(dst_h, top_left_y + h)
    if x0 >= x1 or y0 >= y1:
        return

    src_x0, src_y0 = x0 - top_left_x, y0 - top_left_y
    dst[y0:y1, x0:x1] = patch[src_y0 : src_y0 + (y1 - y0), src_x0 : src_x0 + (x1 - x0)]


def paste_markers(plane_img: np.ndarray, config: AppConfig) -> None:
    dictionary = resolve_dictionary(config.aruco.dictionary)
    mm_per_px = config.rectified_plane.mm_per_px
    side_px = max(8, round(config.aruco.marker_length_mm / mm_per_px))
    quiet = max(2, side_px // 6)

    for marker in config.aruco.markers:
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker.id, side_px)
        padded = cv2.copyMakeBorder(
            marker_img, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
        )
        padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

        cx, cy = config.rectified_plane.mm_to_px(marker.x_mm, marker.y_mm)
        full_side = padded.shape[0]
        top_left_x = round(cx - full_side / 2)
        top_left_y = round(cy - full_side / 2)
        _paste_clipped(plane_img, padded_bgr, top_left_x, top_left_y)


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
