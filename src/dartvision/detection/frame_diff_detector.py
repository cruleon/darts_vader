"""Rilevamento impatto per differenza tra frame.

Il confronto avviene sui pixel GREZZI della camera (non su frame
ri-raddrizzati): e' piu' economico e non introduce il micro-jitter che
si avrebbe ri-calcolando l'homography ad ogni frame e ri-proiettando
l'intera immagine. Solo i contorni candidati che sopravvivono al primo
filtro grezzo vengono proiettati sul piano raddrizzato, dove le soglie
di area sono espresse in unita' fisiche stabili (px^2 del piano),
indipendenti da quanto la webcam sia vicina o lontana dal bersaglio.
"""

from __future__ import annotations

import cv2
import numpy as np

from dartvision.config import DetectionConfig, RectifiedPlaneConfig
from dartvision.detection.interface import ImpactDetector
from dartvision.detection.tip_extraction import select_tip_point
from dartvision.detection.types import Impact


class FrameDiffDetector(ImpactDetector):
    def __init__(self, detection: DetectionConfig, rectified_plane: RectifiedPlaneConfig) -> None:
        self._config = detection
        self._plane = rectified_plane
        k = detection.morph_kernel_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def detect(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray, homography: np.ndarray
    ) -> Impact | None:
        mask = self._changed_region_mask(prev_frame, curr_frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_area: float | None = None
        best_hull_plane: np.ndarray | None = None
        for contour in contours:
            hull_plane = self._project_hull_to_plane(contour, homography)
            area = cv2.contourArea(hull_plane.astype(np.float32))

            if area < self._config.min_impact_area_px or area > self._config.max_impact_area_px:
                continue
            # Tra piu' candidati validi nello stesso confronto, quello con
            # l'area maggiore e' il piu' plausibile: gli altri sono in
            # genere piccoli residui di rumore rimasti dopo la pulizia
            # morfologica.
            if best_area is None or area > best_area:
                best_area = area
                best_hull_plane = hull_plane

        if best_hull_plane is None or best_area is None:
            return None

        center_plane = (self._plane.size_px / 2.0, self._plane.size_px / 2.0)
        x_px, y_px = select_tip_point(best_hull_plane, center_plane)
        x_mm, y_mm = self._plane.px_to_mm(x_px, y_px)

        return Impact(x_mm=x_mm, y_mm=y_mm, x_px=x_px, y_px=y_px, area_px=best_area)

    def _changed_region_mask(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        prev_gray = self._to_blurred_gray(prev_frame)
        curr_gray = self._to_blurred_gray(curr_frame)

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, self._config.diff_threshold, 255, cv2.THRESH_BINARY)

        # Apertura: rimuove speckle isolati (rumore sensore/luce).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        # Chiusura: richiude piccoli buchi nella sagoma (es. riflessi sul fusto).
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    @staticmethod
    def _to_blurred_gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def _project_hull_to_plane(contour: np.ndarray, homography: np.ndarray) -> np.ndarray:
        hull = cv2.convexHull(contour).reshape(-1, 1, 2).astype(np.float32)
        projected = cv2.perspectiveTransform(hull, homography)
        return projected.reshape(-1, 2)
