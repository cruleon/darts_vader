from __future__ import annotations

import cv2
import numpy as np

from dartvision.calibration.aruco_detector import DetectedMarkers
from dartvision.calibration.homography import Calibrator


def _apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _fake_corners_around(center: tuple[float, float], half_side: float = 5.0) -> np.ndarray:
    """4 angoli sintetici la cui media e' esattamente ``center``."""
    cx, cy = center
    return np.array(
        [
            [cx - half_side, cy - half_side],
            [cx + half_side, cy - half_side],
            [cx + half_side, cy + half_side],
            [cx - half_side, cy + half_side],
        ],
        dtype=np.float32,
    )


def _known_plane_to_image_homography() -> np.ndarray:
    """Homography 'verita'' plausibile, plane -> image, non degenere."""
    plane_size = 800
    src = np.float32([[0, 0], [plane_size, 0], [plane_size, plane_size], [0, plane_size]])
    dst = np.float32([[150, 80], [1100, 40], [1180, 700], [80, 650]])
    return cv2.getPerspectiveTransform(src, dst)


def test_calibrate_recovers_correct_homography(app_config):
    calibrator = Calibrator(app_config)
    h_true_plane_to_image = _known_plane_to_image_homography()

    detected_corners = {}
    for marker in app_config.aruco.markers:
        plane_pt = np.array(
            [app_config.rectified_plane.mm_to_px(marker.x_mm, marker.y_mm)], dtype=np.float32
        )
        image_pt = _apply_homography(h_true_plane_to_image, plane_pt)[0]
        detected_corners[marker.id] = _fake_corners_around(tuple(image_pt))

    detected = DetectedMarkers(corners_by_id=detected_corners)
    result = calibrator.calibrate_from_detection(detected)

    assert result.success
    assert result.num_markers_detected == len(app_config.aruco.markers)
    assert set(result.marker_ids_used) == {m.id for m in app_config.aruco.markers}

    # L'homography stimata (image -> plane) deve essere l'inversa di quella
    # 'verita'' (plane -> image): verifichiamolo sui punti dei marker stessi.
    image_pts = np.array(
        [_apply_homography(h_true_plane_to_image,
                            np.array([app_config.rectified_plane.mm_to_px(m.x_mm, m.y_mm)], dtype=np.float32))[0]
         for m in app_config.aruco.markers],
        dtype=np.float32,
    )
    expected_plane_pts = np.array(
        [app_config.rectified_plane.mm_to_px(m.x_mm, m.y_mm) for m in app_config.aruco.markers],
        dtype=np.float32,
    )
    recovered_plane_pts = _apply_homography(result.homography, image_pts)

    assert np.allclose(recovered_plane_pts, expected_plane_pts, atol=1.0)


def test_calibrate_fails_gracefully_with_too_few_markers(app_config):
    calibrator = Calibrator(app_config)
    only_marker = app_config.aruco.markers[0]
    detected = DetectedMarkers(
        corners_by_id={only_marker.id: _fake_corners_around((100.0, 100.0))}
    )

    result = calibrator.calibrate_from_detection(detected)

    assert not result.success
    assert result.homography is None
    assert result.num_markers_detected == 1
    assert "insufficienti" in result.message.lower()


def test_calibrate_ignores_unknown_marker_ids(app_config):
    calibrator = Calibrator(app_config)
    h_true = _known_plane_to_image_homography()

    detected_corners = {}
    for marker in app_config.aruco.markers:
        plane_pt = np.array(
            [app_config.rectified_plane.mm_to_px(marker.x_mm, marker.y_mm)], dtype=np.float32
        )
        image_pt = _apply_homography(h_true, plane_pt)[0]
        detected_corners[marker.id] = _fake_corners_around(tuple(image_pt))

    # marker "estraneo", non presente in configurazione: non deve rompere nulla
    detected_corners[9999] = _fake_corners_around((5.0, 5.0))

    detected = DetectedMarkers(corners_by_id=detected_corners)
    result = calibrator.calibrate_from_detection(detected)

    assert result.success
    assert 9999 not in result.marker_ids_used
    assert result.num_markers_detected == len(app_config.aruco.markers) + 1
