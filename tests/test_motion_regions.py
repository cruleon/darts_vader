from __future__ import annotations

import cv2
import numpy as np
import pytest

from dartvision.detection.motion_regions import (
    changed_region_mask,
    find_candidate_contours,
    project_hull_to_plane,
)

IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _blank_frame(size: int = 200) -> np.ndarray:
    return np.full((size, size, 3), 120, dtype=np.uint8)


def _with_circle(frame: np.ndarray, center_px: tuple[int, int], radius_px: int) -> np.ndarray:
    frame = frame.copy()
    cv2.circle(frame, center_px, radius_px, (20, 20, 20), -1)
    return frame


def test_no_change_produces_empty_mask_and_no_contours():
    prev = _blank_frame()
    curr = prev.copy()

    mask = changed_region_mask(prev, curr, diff_threshold=25, morph_kernel=KERNEL)

    assert mask.sum() == 0
    assert find_candidate_contours(mask) == []


def test_changed_region_produces_one_contour():
    prev = _blank_frame()
    curr = _with_circle(prev, (100, 100), radius_px=10)

    mask = changed_region_mask(prev, curr, diff_threshold=25, morph_kernel=KERNEL)
    contours = find_candidate_contours(mask)

    assert len(contours) == 1


def test_project_hull_to_plane_identity_homography_matches_contour_bounds():
    prev = _blank_frame()
    curr = _with_circle(prev, (100, 100), radius_px=10)

    mask = changed_region_mask(prev, curr, diff_threshold=25, morph_kernel=KERNEL)
    contours = find_candidate_contours(mask)
    hull_plane = project_hull_to_plane(contours[0], IDENTITY_HOMOGRAPHY)

    assert hull_plane.shape[1] == 2
    # Il centro della nuvola di punti dell'hull deve stare vicino al
    # centro del cerchio disegnato (identity homography: nessuna
    # trasformazione applicata).
    centroid = hull_plane.mean(axis=0)
    assert centroid[0] == pytest.approx(100, abs=2)
    assert centroid[1] == pytest.approx(100, abs=2)
