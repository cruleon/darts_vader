"""Orchestrazione pura per generare campioni di training YOLO-pose.

Il modello (``MLTipDetector``) lavora sempre su ritagli del piano gia'
raddrizzato: il dataset va quindi generato direttamente in quel dominio,
non nella vista camera simulata da
``scripts/generate_synthetic_video.py`` (che serve a un altro scopo,
validare calibrazione+frame-diff).

Le funzioni qui sono pure (nessun I/O su disco): l'I/O vive in
``scripts/generate_pose_dataset.py``, cosi' questo modulo resta
testabile rapidamente con immagini piccole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from dartvision.config import AppConfig
from dartvision.synthetic import domain_randomization
from dartvision.synthetic.board_renderer import render_plane_image
from dartvision.synthetic.dart_sprite import place_dart

BoxPx = tuple[float, float, float, float]
PointPx = tuple[float, float]


@dataclass(frozen=True)
class DartLabel:
    bbox_px: BoxPx
    tip_px: PointPx


@dataclass(frozen=True)
class TrainingSample:
    image: np.ndarray
    labels: list[DartLabel] = field(default_factory=list)


def sample_dart_placements(
    rng: np.random.Generator, config: AppConfig, n_darts: int
) -> list[tuple[float, float, float]]:
    """Posizioni/angoli (mm, mm, gradi) di ``n_darts`` freccette.

    Con probabilita' 70% (tranne la prima) una freccetta viene piazzata
    vicino a una gia' presente, per generare abbastanza esempi di
    freccette ravvicinate/sovrapposte — il caso che l'euristica
    geometrica classica gestisce peggio, e che il modello ML deve
    imparare a risolvere.
    """
    board = config.board
    max_radius = board.double_outer_radius_mm
    placements: list[tuple[float, float, float]] = []

    for i in range(n_darts):
        if i == 0 or rng.random() > 0.7:
            radius = max_radius * math.sqrt(rng.uniform(0.0, 1.0))
            angle = rng.uniform(0.0, 360.0)
            x_mm = radius * math.sin(math.radians(angle))
            y_mm = -radius * math.cos(math.radians(angle))
        else:
            base_x, base_y, _ = placements[rng.integers(0, len(placements))]
            offset_radius = rng.uniform(5.0, 35.0)
            offset_angle = rng.uniform(0.0, 360.0)
            x_mm = base_x + offset_radius * math.sin(math.radians(offset_angle))
            y_mm = base_y - offset_radius * math.cos(math.radians(offset_angle))

        dart_angle_deg = rng.uniform(0.0, 360.0)
        placements.append((x_mm, y_mm, dart_angle_deg))

    return placements


def _extract_crop(plane_img: np.ndarray, top_left: np.ndarray, size_px: int) -> np.ndarray:
    """Ritaglia un quadrato ``size_px`` da ``plane_img`` a partire da
    ``top_left``, riempiendo con lo sfondo neutro del piano se il
    ritaglio esce dai bordi (es. freccetta vicina al bordo del piano)."""
    h, w = plane_img.shape[:2]
    x0, y0 = round(top_left[0]), round(top_left[1])

    canvas = np.full((size_px, size_px, 3), 60, dtype=np.uint8)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x0 + size_px), min(h, y0 + size_px)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return canvas

    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    canvas[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = plane_img[
        src_y0:src_y1, src_x0:src_x1
    ]
    return canvas


def _clip_bbox_to_crop(bbox: BoxPx, crop_size: int) -> BoxPx | None:
    x_min, y_min, x_max, y_max = bbox
    x_min_c, y_min_c = max(0.0, x_min), max(0.0, y_min)
    x_max_c, y_max_c = min(float(crop_size), x_max), min(float(crop_size), y_max)
    if x_max_c <= x_min_c or y_max_c <= y_min_c:
        return None
    return (x_min_c, y_min_c, x_max_c, y_max_c)


def crop_sample_from_plane(
    plane_img: np.ndarray,
    placed: list[tuple[PointPx, BoxPx]],
    target_tip_px: PointPx,
    rng: np.random.Generator,
    crop_size: int = 320,
    apply_domain_randomization: bool = True,
) -> TrainingSample:
    """Ritaglia ``plane_img`` attorno a ``target_tip_px`` (la freccetta che
    "innesca" il rilevamento), etichettando tutte le freccette di
    ``placed`` (tip_px, bbox_px) visibili nel ritaglio.

    Il ritaglio ha lo stesso rumore geometrico che avra' il vero ROI
    proposer (offset/scala jittered) — cosi' il modello non si allena su
    ritagli piu' "puliti" del reale. La scala jittered raddoppia anche
    da augmentation per la distanza camera-bersaglio (piu' vicina/
    lontana in setup diversi). Condivisa tra dataset sintetico
    (``render_training_sample``) e import di dataset reali
    (``dartvision.synthetic.deepdarts_import``), cosi' entrambi
    producono campioni nello stesso formato con la stessa logica di
    ritaglio."""
    target = np.array(target_tip_px)
    scale = rng.uniform(0.8, 1.3)
    raw_size = max(8, round(crop_size * scale))
    offset = rng.uniform(-0.25, 0.25, size=2) * raw_size
    top_left = target - raw_size / 2 + offset

    raw_crop = _extract_crop(plane_img, top_left, raw_size)
    crop = cv2.resize(raw_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    scale_factor = crop_size / raw_size

    labels: list[DartLabel] = []
    for tip_px, bbox_px in placed:
        tip_local = (np.array(tip_px) - top_left) * scale_factor
        if not (0.0 <= tip_local[0] <= crop_size and 0.0 <= tip_local[1] <= crop_size):
            continue  # punta fuori dal ritaglio: niente da supervisionare

        bbox_local = (
            (bbox_px[0] - top_left[0]) * scale_factor,
            (bbox_px[1] - top_left[1]) * scale_factor,
            (bbox_px[2] - top_left[0]) * scale_factor,
            (bbox_px[3] - top_left[1]) * scale_factor,
        )
        clipped = _clip_bbox_to_crop(bbox_local, crop_size)
        if clipped is None:
            continue

        labels.append(DartLabel(bbox_px=clipped, tip_px=(float(tip_local[0]), float(tip_local[1]))))

    if apply_domain_randomization:
        crop = domain_randomization.randomize(crop, rng)
    return TrainingSample(image=crop, labels=labels)


def render_training_sample(
    config: AppConfig,
    rng: np.random.Generator,
    crop_size: int = 320,
    max_darts_per_crop: int = 3,
) -> TrainingSample:
    """Genera un campione di training: un ritaglio del piano raddrizzato
    con 1-``max_darts_per_crop`` freccette e le loro etichette
    (bbox+punta, in coordinate LOCALI al ritaglio)."""
    plane_img = render_plane_image(config)
    n_darts = int(rng.integers(1, max_darts_per_crop + 1))
    placements = sample_dart_placements(rng, config, n_darts)
    placed = [place_dart(plane_img, config, x_mm, y_mm, angle, rng) for x_mm, y_mm, angle in placements]

    return crop_sample_from_plane(plane_img, placed, placed[-1][0], rng, crop_size)


def to_yolo_pose_label(label: DartLabel, crop_size: int) -> str:
    """Formato riga YOLO-pose: ``class cx cy w h kpx kpy kpv``, tutto
    normalizzato in [0, 1] rispetto a ``crop_size``. Un solo keypoint
    (la punta), sempre marcato visibile (``kpv=2``): il dataset e'
    sintetico e la punta e' sempre disegnata per costruzione."""
    x_min, y_min, x_max, y_max = label.bbox_px
    cx = (x_min + x_max) / 2 / crop_size
    cy = (y_min + y_max) / 2 / crop_size
    w = (x_max - x_min) / crop_size
    h = (y_max - y_min) / crop_size
    kpx = label.tip_px[0] / crop_size
    kpy = label.tip_px[1] / crop_size
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpx:.6f} {kpy:.6f} 2"
