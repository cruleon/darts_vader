from __future__ import annotations

import cv2
import numpy as np
import pytest

from dartvision.detection.frame_diff_detector import FrameDiffDetector

IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)


@pytest.fixture
def detector(app_config):
    return FrameDiffDetector(app_config.detection, app_config.rectified_plane)


def _blank_frame(app_config) -> np.ndarray:
    size = app_config.rectified_plane.size_px
    return np.full((size, size, 3), 120, dtype=np.uint8)


def _with_circle(frame: np.ndarray, center_px: tuple[int, int], radius_px: int) -> np.ndarray:
    frame = frame.copy()
    cv2.circle(frame, center_px, radius_px, (20, 20, 20), -1)
    return frame


def test_detects_single_dart_sized_blob(app_config, detector):
    prev = _blank_frame(app_config)
    center_of_plane = app_config.rectified_plane.size_px / 2.0  # 500
    dart_center_px = (int(center_of_plane) + 100, int(center_of_plane))
    curr = _with_circle(prev, dart_center_px, radius_px=6)

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is not None
    # Il punto scelto e' sul bordo del blob rivolto verso il centro,
    # non il suo centroide.
    expected_x_px = dart_center_px[0] - 6
    assert impact.x_px == pytest.approx(expected_x_px, abs=2.5)
    assert impact.y_px == pytest.approx(dart_center_px[1], abs=2.5)

    expected_x_mm, expected_y_mm = app_config.rectified_plane.px_to_mm(
        impact.x_px, impact.y_px
    )
    assert impact.x_mm == pytest.approx(expected_x_mm, abs=1e-6)
    assert impact.y_mm == pytest.approx(expected_y_mm, abs=1e-6)


def test_ignores_speckle_noise_below_min_area(app_config, detector):
    prev = _blank_frame(app_config)
    center_px = int(app_config.rectified_plane.size_px / 2.0)
    curr = _with_circle(prev, (center_px, center_px), radius_px=1)

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is None


def test_ignores_large_blob_above_max_area(app_config, detector):
    prev = _blank_frame(app_config)
    curr = prev.copy()
    size = app_config.rectified_plane.size_px
    # Un rettangolo grande quanto una mano che entra nel campo.
    cv2.rectangle(curr, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), (20, 20, 20), -1)

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is None


def test_no_change_between_frames_returns_none(app_config, detector):
    prev = _blank_frame(app_config)
    curr = prev.copy()

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is None


def test_picks_largest_candidate_when_multiple_present(app_config, detector):
    prev = _blank_frame(app_config)
    center = int(app_config.rectified_plane.size_px / 2.0)

    curr = _with_circle(prev, (center - 200, center), radius_px=5)  # piccolo, area ~78
    curr = _with_circle(curr, (center + 200, center), radius_px=12)  # grande, area ~452

    impact = detector.detect(prev, curr, IDENTITY_HOMOGRAPHY)

    assert impact is not None
    # Deve aver scelto il blob grande (a destra), il cui punto piu'
    # vicino al centro e' a center+200-12, non quello piccolo a sinistra.
    assert impact.x_px == pytest.approx(center + 200 - 12, abs=1.5)
