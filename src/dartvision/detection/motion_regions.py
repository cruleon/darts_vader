"""Rilevamento di regioni "cambiate" tra due frame grezzi della camera.

Logica condivisa tra ``FrameDiffDetector`` (frame-differencing puro) e
``MLTipDetector`` (frame-differencing come proposta economica di ROI,
poi affinata da un modello ML): isolare le regioni di un frame che sono
cambiate rispetto al precedente, e proiettarle sul piano raddrizzato.

Estratta in un modulo a se' stante (invece di restare privata dentro
``FrameDiffDetector``) proprio perche' entrambi i detector ne hanno
bisogno, con soglie di area diverse ma la stessa maschera/contorni.
"""

from __future__ import annotations

import cv2
import numpy as np


def changed_region_mask(
    prev_frame: np.ndarray, curr_frame: np.ndarray, diff_threshold: int, morph_kernel: np.ndarray
) -> np.ndarray:
    """Maschera binaria delle regioni cambiate tra ``prev_frame`` e ``curr_frame``."""
    prev_gray = _to_blurred_gray(prev_frame)
    curr_gray = _to_blurred_gray(curr_frame)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # Apertura: rimuove speckle isolati (rumore sensore/luce).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
    # Chiusura: richiude piccoli buchi nella sagoma (es. riflessi sul fusto).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
    return mask


def find_candidate_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Contorni esterni della maschera, cosi' come li ritorna OpenCV."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def project_hull_to_plane(contour: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Hull convesso di ``contour`` proiettato sul piano raddrizzato."""
    hull = cv2.convexHull(contour).reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(hull, homography)
    return projected.reshape(-1, 2)


def _to_blurred_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(gray, (5, 5), 0)
