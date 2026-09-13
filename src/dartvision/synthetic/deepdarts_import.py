"""Conversione del dataset reale DeepDarts (McNally et al., CVSports 2021)
nello stesso formato YOLO-pose usato per il dataset sintetico
(``dartvision.synthetic.pose_dataset``).

DeepDarts annota, per ogni immagine, 4 punti di calibrazione fissi (ai
confini di settore standard del bersaglio — 5/20, 17/3, 8/11, 13/6, sul
bordo esterno dell'anello doppio) piu' 0-3 punte di freccetta, tutto
normalizzato [0,1] rispetto al ritaglio quadrato "cropped_images/800"
fornito dal dataset (vedi ``dataset/annotate.py`` nel loro repository:
``on_click`` registra ``x/w, y/h`` sull'immagine gia' ritagliata da
``crop_board``, non sull'immagine originale).

Poiche' quei 4 punti sono a posizioni fisiche NOTE sul bersaglio (stesse
specifiche BDO gia' usate in ``config/board_config.yaml``: raggio =
``double_outer_radius_mm``), possiamo calcolare un'omografia diretta
verso il NOSTRO piano raddrizzato — stessa idea del ``Calibrator``
(ArUco), ma con corrispondenze note anziche' rilevate — e proiettarci
sopra anche le punte reali. Verificato empiricamente: bersaglio e punte
proiettati con questa omografia si allineano esattamente con
``render_plane_image`` e con le freccette visibili nelle foto originali.

A differenza del dataset sintetico, qui non abbiamo un bounding box
"vero" per ogni freccetta (solo il punto di punta): usiamo un riquadro
fisso di dimensione fisica ragionevole centrato sulla punta, la stessa
scelta fatta internamente dagli stessi autori di DeepDarts
(``cfg.train.bbox_size``, una frazione fissa dell'immagine). Il
keypoint (la punta) resta l'unico segnale che conta; il box serve solo
da impalcatura per l'architettura YOLO-pose.

Il dataset NON e' incluso in questo repository (licenza CC-BY, richiede
un account IEEE DataPort gratuito):
https://ieee-dataport.org/open-access/deepdarts-dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dartvision.config import AppConfig
from dartvision.synthetic.pose_dataset import (
    BoxPx,
    PointPx,
    TrainingSample,
    crop_sample_from_plane,
)

# Angolo (gradi, senso orario da "in alto" — stessa convenzione di
# dartvision.synthetic.dart_sprite._direction_vector) del confine di
# settore corrispondente a ciascun punto di calibrazione DeepDarts,
# nell'ordine cal_1..cal_4 delle loro annotazioni. Derivato da
# dataset/annotate.py:transform() del repository DeepDarts (i punti
# cal_1/cal_2 sono diametralmente opposti, offset di 9 gradi dal
# verticale; cal_3/cal_4 idem dall'orizzontale) e verificato
# empiricamente proiettando bersagli e punte reali sul nostro piano.
CALIBRATION_POINT_ANGLES_DEG = (351.0, 171.0, 261.0, 81.0)

DEFAULT_DART_BOX_SIZE_MM = 45.0


@dataclass(frozen=True)
class DeepDartsRow:
    img_folder: str
    img_name: str
    cal_points_norm: np.ndarray  # (4, 2), normalizzate [0,1] sul crop 800x800
    dart_points_norm: np.ndarray  # (N, 2), N in 0..3, stessa normalizzazione


def load_rows(labels_pkl_path: str | Path) -> list[DeepDartsRow]:
    """Legge ``labels.pkl`` (formato pandas del repository DeepDarts) e
    ritorna solo le righe con tutti e 4 i punti di calibrazione
    presenti (scarta righe malformate/incomplete)."""
    df = pd.read_pickle(labels_pkl_path)
    rows: list[DeepDartsRow] = []
    for record in df.itertuples(index=False):
        xy = np.asarray(record.xy, dtype=np.float64)
        if xy.shape[0] < 4:
            continue
        rows.append(
            DeepDartsRow(
                img_folder=record.img_folder,
                img_name=record.img_name,
                cal_points_norm=xy[:4],
                dart_points_norm=xy[4:],
            )
        )
    return rows


def _direction_vector(angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    return np.array([np.sin(rad), -np.cos(rad)])


def calibration_points_plane_px(config: AppConfig) -> np.ndarray:
    """Posizione (px, piano raddrizzato) dei 4 punti di calibrazione
    DeepDarts, nell'ordine cal_1..cal_4, per la geometria di ``config``."""
    radius_mm = config.board.double_outer_radius_mm
    points = [
        config.rectified_plane.mm_to_px(*(_direction_vector(angle_deg) * radius_mm))
        for angle_deg in CALIBRATION_POINT_ANGLES_DEG
    ]
    return np.array(points, dtype=np.float32)


def compute_homography(
    cal_points_norm: np.ndarray, image_size: tuple[int, int], config: AppConfig
) -> np.ndarray:
    """Omografia che mappa pixel dell'immagine DeepDarts (il crop
    800x800, non l'originale) verso pixel del nostro piano raddrizzato,
    dai 4 punti di calibrazione noti."""
    w, h = image_size
    src = (cal_points_norm * [w, h]).astype(np.float32)
    dst = calibration_points_plane_px(config)
    return cv2.getPerspectiveTransform(src, dst)


def dart_bbox_px(
    tip_px: PointPx, config: AppConfig, box_size_mm: float = DEFAULT_DART_BOX_SIZE_MM
) -> BoxPx:
    """Riquadro quadrato fisso (px, piano raddrizzato) centrato su
    ``tip_px``: DeepDarts non annota un bbox reale, solo il keypoint."""
    half_px = (box_size_mm / config.rectified_plane.mm_per_px) / 2.0
    x, y = tip_px
    return (x - half_px, y - half_px, x + half_px, y + half_px)


def warp_row_to_plane(
    image: np.ndarray,
    row: DeepDartsRow,
    config: AppConfig,
    box_size_mm: float = DEFAULT_DART_BOX_SIZE_MM,
) -> tuple[np.ndarray, list[tuple[PointPx, BoxPx]]]:
    """Raddrizza ``image`` (crop DeepDarts) sul nostro piano e proietta
    le punte annotate con la stessa omografia.

    Ritorna ``(piano_raddrizzato, [(tip_px, bbox_px), ...])``.
    """
    h, w = image.shape[:2]
    homography = compute_homography(row.cal_points_norm, (w, h), config)

    plane_size = config.rectified_plane.size_px
    plane_img = cv2.warpPerspective(image, homography, (plane_size, plane_size))

    darts_px: list[tuple[PointPx, BoxPx]] = []
    if len(row.dart_points_norm) > 0:
        dart_px_img = (row.dart_points_norm * [w, h]).astype(np.float32)
        dart_px_plane = cv2.perspectiveTransform(dart_px_img.reshape(-1, 1, 2), homography).reshape(-1, 2)
        for x, y in dart_px_plane:
            tip = (float(x), float(y))
            darts_px.append((tip, dart_bbox_px(tip, config, box_size_mm)))

    return plane_img, darts_px


def training_samples_from_row(
    image: np.ndarray,
    row: DeepDartsRow,
    config: AppConfig,
    rng: np.random.Generator,
    crop_size: int = 320,
    box_size_mm: float = DEFAULT_DART_BOX_SIZE_MM,
) -> list[TrainingSample]:
    """Un campione di training per ciascuna freccetta della riga (quella
    che "innesca" il ritaglio), etichettando tutte le freccette
    visibili in quel ritaglio. Niente domain randomization: sono gia'
    foto reali di camera, non serve simularne la variabilita'."""
    plane_img, darts_px = warp_row_to_plane(image, row, config, box_size_mm)
    return [
        crop_sample_from_plane(
            plane_img, darts_px, target_tip_px, rng, crop_size, apply_domain_randomization=False
        )
        for target_tip_px, _ in darts_px
    ]
