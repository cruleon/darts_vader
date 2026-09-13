"""Calcolo dell'homography immagine -> piano del bersaglio raddrizzato.

L'homography viene ricalcolata ad ogni frame (o ad ogni volta che serve)
a partire dai marker ArUco effettivamente visti: questo e' cio' che
rende il sistema robusto a una webcam che cambia posizione tra una
sessione e l'altra, senza nessuna calibrazione "fissa" salvata su disco.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from dartvision.calibration.aruco_detector import ArucoDetector, DetectedMarkers
from dartvision.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationResult:
    """Esito del tentativo di calibrazione su un frame.

    ``success=False`` non e' un errore di programma: significa solo che
    per questo frame non e' stato possibile calcolare un'homography
    affidabile (es. marker occlusi). Il chiamante deve scartare il
    frame e riprovare con il successivo, mai interrompersi.
    """

    success: bool
    homography: np.ndarray | None
    marker_ids_used: tuple[int, ...]
    num_markers_detected: int
    message: str = ""


class Calibrator:
    """Rileva i marker e calcola l'homography immagine -> piano raddrizzato."""

    def __init__(self, config: AppConfig, aruco_detector: ArucoDetector | None = None) -> None:
        self._config = config
        self._detector = aruco_detector or ArucoDetector(config.aruco)

        # Punto atteso (in pixel, sul piano raddrizzato) per ciascun id
        # di marker configurato: pre-calcolato una volta sola.
        plane = config.rectified_plane
        self._plane_point_by_id: dict[int, tuple[float, float]] = {
            marker.id: plane.mm_to_px(marker.x_mm, marker.y_mm)
            for marker in config.aruco.markers
        }

    def detect_markers(self, frame: np.ndarray) -> DetectedMarkers:
        return self._detector.detect(frame)

    def calibrate(self, frame: np.ndarray) -> CalibrationResult:
        detected = self.detect_markers(frame)
        return self.calibrate_from_detection(detected)

    def calibrate_from_detection(self, detected: DetectedMarkers) -> CalibrationResult:
        matched_ids = sorted(set(detected.ids) & set(self._plane_point_by_id))
        min_required = self._config.calibration.min_markers_required

        if len(matched_ids) < min_required:
            message = (
                f"Marker insufficienti per calibrare: rilevati "
                f"{len(detected.ids)} ({detected.ids}), di cui validi "
                f"{len(matched_ids)}, richiesti almeno {min_required}."
            )
            logger.warning(message)
            return CalibrationResult(
                success=False,
                homography=None,
                marker_ids_used=tuple(matched_ids),
                num_markers_detected=len(detected.ids),
                message=message,
            )

        image_pts = np.array(
            [detected.center(mid) for mid in matched_ids], dtype=np.float32
        )
        plane_pts = np.array(
            [self._plane_point_by_id[mid] for mid in matched_ids], dtype=np.float32
        )

        method = cv2.RANSAC if len(matched_ids) > 4 else 0
        homography, _mask = cv2.findHomography(image_pts, plane_pts, method=method)

        if homography is None:
            message = (
                f"cv2.findHomography non ha prodotto una soluzione valida "
                f"con i marker {matched_ids} (probabilmente quasi collineari)."
            )
            logger.warning(message)
            return CalibrationResult(
                success=False,
                homography=None,
                marker_ids_used=tuple(matched_ids),
                num_markers_detected=len(detected.ids),
                message=message,
            )

        return CalibrationResult(
            success=True,
            homography=homography,
            marker_ids_used=tuple(matched_ids),
            num_markers_detected=len(detected.ids),
        )

    def rectify(self, frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Applica l'homography per ottenere la vista dall'alto del bersaglio."""
        size = self._config.rectified_plane.size_px
        return cv2.warpPerspective(frame, homography, (size, size))
