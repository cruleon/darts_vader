from __future__ import annotations

import cv2
import numpy as np
import pytest

from dartvision.detection import ml_tip_detector
from dartvision.detection.ml_tip_detector import MLTipDetector, ModelNotAvailableError

IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)


class _FakePoseModel:
    """Modello finto scriptato: ritorna sempre le stesse detection,
    indipendentemente dal crop ricevuto. Evita ogni dipendenza da
    ultralytics/GPU nei test."""

    def __init__(self, detections_by_call: list[list[tuple[tuple[float, float], float]]]):
        self._detections_by_call = list(detections_by_call)
        self.calls: list[np.ndarray] = []

    def predict(self, crop, confidence_threshold: float):
        self.calls.append(crop)
        if not self._detections_by_call:
            return []
        return self._detections_by_call.pop(0)


def _blank_frame(app_config) -> np.ndarray:
    size = app_config.rectified_plane.size_px
    return np.full((size, size, 3), 120, dtype=np.uint8)


def _with_circle(frame: np.ndarray, center_px: tuple[int, int], radius_px: int) -> np.ndarray:
    frame = frame.copy()
    cv2.circle(frame, center_px, radius_px, (20, 20, 20), -1)
    return frame


def _detector(app_config, model) -> MLTipDetector:
    return MLTipDetector(
        app_config.ml_detection, app_config.detection, app_config.rectified_plane, model=model
    )


def test_no_change_between_frames_returns_none(app_config):
    detector = _detector(app_config, model=_FakePoseModel([]))
    prev = _blank_frame(app_config)
    curr = prev.copy()

    assert detector.detect(prev, curr, IDENTITY_HOMOGRAPHY) is None


def test_candidate_below_threshold_is_never_sent_to_model(app_config):
    # Un puntino di 1px di raggio e' sotto candidate_min_area_px: il
    # modello non deve nemmeno essere interpellato.
    model = _FakePoseModel([[((0.0, 0.0), 0.99)]])
    detector = _detector(app_config, model=model)
    prev = _blank_frame(app_config)
    center = int(app_config.rectified_plane.size_px / 2.0)
    curr = _with_circle(prev, (center, center), radius_px=1)

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is None
    assert model.calls == []


def test_detection_above_threshold_maps_to_plane_mm(app_config):
    center = int(app_config.rectified_plane.size_px / 2.0)
    dart_center_px = (center + 100, center)
    # Il modello ritorna un keypoint in coordinate LOCALI al crop: (5, 5)
    # dall'angolo top-left della ROI ritagliata attorno al candidato.
    model = _FakePoseModel([[((5.0, 5.0), 0.8)]])
    detector = _detector(app_config, model=model)

    prev = _blank_frame(app_config)
    curr = _with_circle(prev, dart_center_px, radius_px=10)

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is not None
    assert impact.confidence == pytest.approx(0.8)
    expected_x_mm, expected_y_mm = app_config.rectified_plane.px_to_mm(impact.x_px, impact.y_px)
    assert impact.x_mm == pytest.approx(expected_x_mm)
    assert impact.y_mm == pytest.approx(expected_y_mm)


def test_detection_below_confidence_threshold_is_rejected(app_config):
    low_confidence = app_config.ml_detection.confidence_threshold - 0.1
    model = _FakePoseModel([[((5.0, 5.0), max(low_confidence, 0.0))]])
    detector = _detector(app_config, model=model)

    prev = _blank_frame(app_config)
    center = int(app_config.rectified_plane.size_px / 2.0)
    curr = _with_circle(prev, (center + 100, center), radius_px=10)

    assert detector.detect(prev, curr, IDENTITY_HOMOGRAPHY) is None


def test_picks_most_confident_detection_across_candidates(app_config):
    center = int(app_config.rectified_plane.size_px / 2.0)
    curr = _with_circle(_blank_frame(app_config), (center - 200, center), radius_px=10)
    curr = _with_circle(curr, (center + 200, center), radius_px=10)
    prev = _blank_frame(app_config)

    # Un candidato per ogni blob: il secondo ha confidenza piu' alta.
    model = _FakePoseModel([[((5.0, 5.0), 0.5)], [((5.0, 5.0), 0.9)]])
    detector = _detector(app_config, model=model)

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is not None
    assert impact.confidence == pytest.approx(0.9)


def test_missing_weights_file_raises_model_not_available_error(app_config, tmp_path):
    ml_config = app_config.ml_detection
    bad_config = type(ml_config)(
        model_path=str(tmp_path / "does_not_exist.pt"),
        confidence_threshold=ml_config.confidence_threshold,
        device=ml_config.device,
        roi_padding_px=ml_config.roi_padding_px,
        candidate_min_area_px=ml_config.candidate_min_area_px,
        candidate_max_area_px=ml_config.candidate_max_area_px,
    )

    with pytest.raises(ModelNotAvailableError, match="Nessun modello addestrato"):
        MLTipDetector(bad_config, app_config.detection, app_config.rectified_plane)


def test_missing_ultralytics_library_raises_model_not_available_error(app_config, tmp_path, monkeypatch):
    weights_path = tmp_path / "fake_weights.pt"
    weights_path.write_bytes(b"not a real model, just needs to exist")

    def _raise_not_available():
        raise ModelNotAvailableError("libreria non installata (simulato nel test)")

    monkeypatch.setattr(ml_tip_detector, "_load_yolo_class", _raise_not_available)

    ml_config = app_config.ml_detection
    config_with_real_path = type(ml_config)(
        model_path=str(weights_path),
        confidence_threshold=ml_config.confidence_threshold,
        device=ml_config.device,
        roi_padding_px=ml_config.roi_padding_px,
        candidate_min_area_px=ml_config.candidate_min_area_px,
        candidate_max_area_px=ml_config.candidate_max_area_px,
    )

    with pytest.raises(ModelNotAvailableError, match="simulato nel test"):
        MLTipDetector(config_with_real_path, app_config.detection, app_config.rectified_plane)
