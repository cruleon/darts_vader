"""Rilevamento impatto con un modello ML (YOLO-pose).

Design ibrido: riusa la stessa maschera "regione cambiata" di
``FrameDiffDetector`` (``dartvision.detection.motion_regions``) come
proposta economica di ROI sui frame grezzi, ma affida a un modello di
keypoint detection la localizzazione precisa della punta dentro
ciascuna ROI, raddrizzata sul piano. Cosi' il modello gira solo sui
rari frame con qualcosa di cambiato, e solo su piccoli ritagli,
restando sostenibile anche su CPU; e ogni freccetta rilevata ha la
propria punta, risolvendo nativamente il caso di freccette
ravvicinate/sovrapposte che confonde l'euristica geometrica di
``FrameDiffDetector`` (vedi ``tip_extraction.select_tip_point``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dartvision.config import DetectionConfig, MLDetectionConfig, RectifiedPlaneConfig
from dartvision.detection.interface import ImpactDetector
from dartvision.detection.motion_regions import (
    changed_region_mask,
    find_candidate_contours,
    project_hull_to_plane,
)
from dartvision.detection.types import Impact

# (punta_x, punta_y), confidenza — quello che un "modello" deve esporre
# per essere usabile da MLTipDetector (vedi PoseModel piu' sotto).
Detection = tuple[tuple[float, float], float]


class ModelNotAvailableError(RuntimeError):
    """Il modello ML non e' disponibile: libreria mancante o pesi assenti.

    A differenza del fallback per-frame di ``Calibrator`` (normale, un
    frame mal illuminato non deve interrompere la sessione), un modello
    mancante e' un errore di SETUP: non va mai mascherato da un
    ``None`` silenzioso ripetuto ad ogni frame, quindi si solleva alla
    costruzione, non al primo ``detect()``.
    """


def _load_yolo_class():
    """Import lazy di ultralytics: isolato in una funzione a se' per
    poter essere monkeypatchato nei test senza richiedere la libreria
    installata."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelNotAvailableError(
            "L'estensione 'ml' non e' installata: uv sync --extra ml"
        ) from exc
    return YOLO


class UltralyticsPoseModel:
    """Adapter minimo sopra ``ultralytics.YOLO``: isola in un solo punto
    l'unica dipendenza specifica della libreria di inferenza."""

    def __init__(self, model_path: str, device: str) -> None:
        yolo_class = _load_yolo_class()
        self._model = yolo_class(model_path)
        self._device = device

    def predict(self, crop: np.ndarray, confidence_threshold: float) -> list[Detection]:
        results = self._model.predict(
            crop, device=self._device, conf=confidence_threshold, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for kp, conf in zip(keypoints_xy, confidences, strict=True):
                detections.append(((float(kp[0][0]), float(kp[0][1])), float(conf)))
        return detections


class MLTipDetector(ImpactDetector):
    def __init__(
        self,
        ml_detection: MLDetectionConfig,
        detection: DetectionConfig,
        rectified_plane: RectifiedPlaneConfig,
        model: object | None = None,
    ) -> None:
        self._config = ml_detection
        self._detection = detection
        self._plane = rectified_plane
        k = detection.morph_kernel_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        if model is not None:
            self._model = model
            return

        if not Path(ml_detection.model_path).is_file():
            raise ModelNotAvailableError(
                f"Nessun modello addestrato in '{ml_detection.model_path}': "
                "vedi scripts/train_pose_model.py"
            )
        device = self._resolve_device(ml_detection.device)
        self._model = UltralyticsPoseModel(ml_detection.model_path, device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray, homography: np.ndarray
    ) -> Impact | None:
        mask = changed_region_mask(
            prev_frame, curr_frame, self._detection.diff_threshold, self._kernel
        )
        contours = find_candidate_contours(mask)
        if not contours:
            return None

        candidates: list[tuple[np.ndarray, float]] = []
        for contour in contours:
            hull_plane = project_hull_to_plane(contour, homography)
            area = cv2.contourArea(hull_plane.astype(np.float32))
            if area < self._config.candidate_min_area_px or area > self._config.candidate_max_area_px:
                continue
            candidates.append((hull_plane, area))

        if not candidates:
            return None

        plane_size = self._plane.size_px
        rectified = cv2.warpPerspective(curr_frame, homography, (plane_size, plane_size))

        best_confidence: float | None = None
        best_tip_px: tuple[float, float] | None = None
        best_area: float | None = None

        for hull_plane, area in candidates:
            roi = self._roi_from_hull(hull_plane, plane_size)
            if roi is None:
                continue
            x0, y0, x1, y1 = roi
            crop = rectified[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            for (kp_x, kp_y), confidence in self._model.predict(crop, self._config.confidence_threshold):
                if confidence < self._config.confidence_threshold:
                    continue
                if best_confidence is None or confidence > best_confidence:
                    best_confidence = confidence
                    best_tip_px = (kp_x + x0, kp_y + y0)
                    best_area = area

        if best_tip_px is None or best_confidence is None or best_area is None:
            return None

        x_mm, y_mm = self._plane.px_to_mm(*best_tip_px)
        return Impact(
            x_mm=x_mm,
            y_mm=y_mm,
            x_px=best_tip_px[0],
            y_px=best_tip_px[1],
            area_px=best_area,
            confidence=best_confidence,
        )

    def _roi_from_hull(self, hull_plane: np.ndarray, plane_size: int) -> tuple[int, int, int, int] | None:
        x_min, y_min = hull_plane.min(axis=0)
        x_max, y_max = hull_plane.max(axis=0)
        pad = self._config.roi_padding_px

        x0 = max(0, int(np.floor(x_min - pad)))
        y0 = max(0, int(np.floor(y_min - pad)))
        x1 = min(plane_size, int(np.ceil(x_max + pad)))
        y1 = min(plane_size, int(np.ceil(y_max + pad)))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
