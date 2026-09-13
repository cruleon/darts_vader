from __future__ import annotations

import cv2
import numpy as np
import pytest

from dartvision.synthetic.board_renderer import render_plane_image
from dartvision.synthetic.deepdarts_import import (
    CALIBRATION_POINT_ANGLES_DEG,
    DeepDartsRow,
    calibration_points_plane_px,
    compute_homography,
    dart_bbox_px,
    training_samples_from_row,
    warp_row_to_plane,
)


def _cal_points_norm_matching_plane(app_config, image_size: int) -> np.ndarray:
    """Costruisce punti di calibrazione "sintetici" che, per costruzione,
    corrispondono esattamente alle posizioni attese sul piano (a meno di
    scala/offset identici), cosi' l'omografia risultante e' prevedibile
    (quasi-identita', a meno del fattore di scala piano/immagine)."""
    plane_px = calibration_points_plane_px(app_config)
    plane_size = app_config.rectified_plane.size_px
    return plane_px / plane_size  # normalizzato [0,1] come farebbe DeepDarts


def test_calibration_points_are_diametrically_opposed_pairs(app_config):
    points = calibration_points_plane_px(app_config)
    center = app_config.rectified_plane.size_px / 2.0

    # cal_1/cal_2 e cal_3/cal_4 devono essere simmetrici rispetto al
    # centro del piano (stesso raggio, direzioni opposte).
    midpoint_12 = (points[0] + points[1]) / 2
    midpoint_34 = (points[2] + points[3]) / 2
    assert midpoint_12 == pytest.approx([center, center], abs=1e-3)
    assert midpoint_34 == pytest.approx([center, center], abs=1e-3)


def test_calibration_points_lie_on_double_outer_radius(app_config):
    points = calibration_points_plane_px(app_config)
    center = np.array([app_config.rectified_plane.size_px / 2.0] * 2)
    expected_radius_px = app_config.board.double_outer_radius_mm / app_config.rectified_plane.mm_per_px

    for point in points:
        assert np.linalg.norm(point - center) == pytest.approx(expected_radius_px, rel=1e-6)


def test_homography_maps_calibration_points_back_onto_plane(app_config):
    image_size = 800
    cal_points_norm = _cal_points_norm_matching_plane(app_config, image_size)

    homography = compute_homography(cal_points_norm, (image_size, image_size), app_config)

    src_px = (cal_points_norm * image_size).astype(np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src_px, homography).reshape(-1, 2)
    expected = calibration_points_plane_px(app_config)
    assert projected == pytest.approx(expected, abs=1e-2)


def test_dart_bbox_is_centered_square_of_requested_physical_size(app_config):
    tip_px = (100.0, 200.0)
    bbox = dart_bbox_px(tip_px, app_config, box_size_mm=50.0)

    x_min, y_min, x_max, y_max = bbox
    expected_side_px = 50.0 / app_config.rectified_plane.mm_per_px
    assert (x_max - x_min) == pytest.approx(expected_side_px)
    assert (y_max - y_min) == pytest.approx(expected_side_px)
    assert (x_min + x_max) / 2 == pytest.approx(tip_px[0])
    assert (y_min + y_max) / 2 == pytest.approx(tip_px[1])


def test_warp_row_to_plane_projects_dart_near_expected_position(app_config):
    image_size = 800
    cal_points_norm = _cal_points_norm_matching_plane(app_config, image_size)

    # Una "freccetta" piazzata esattamente al centro del piano, espressa
    # nello stesso sistema normalizzato dell'immagine sorgente.
    center_norm = np.array([[0.5, 0.5]])
    row = DeepDartsRow(
        img_folder="fake",
        img_name="fake.jpg",
        cal_points_norm=cal_points_norm,
        dart_points_norm=center_norm,
    )
    image = render_plane_image_at_size(app_config, image_size)

    plane_img, darts_px = warp_row_to_plane(image, row, app_config)

    assert plane_img.shape[:2] == (app_config.rectified_plane.size_px, app_config.rectified_plane.size_px)
    assert len(darts_px) == 1
    tip_px, _ = darts_px[0]
    plane_center = app_config.rectified_plane.size_px / 2.0
    assert tip_px[0] == pytest.approx(plane_center, abs=1.0)
    assert tip_px[1] == pytest.approx(plane_center, abs=1.0)


def render_plane_image_at_size(app_config, size: int):
    plane_img = render_plane_image(app_config)
    return cv2.resize(plane_img, (size, size))


def test_training_samples_from_row_returns_one_sample_per_dart(app_config):
    image_size = 800
    cal_points_norm = _cal_points_norm_matching_plane(app_config, image_size)
    dart_points_norm = np.array([[0.5, 0.5], [0.55, 0.45]])
    row = DeepDartsRow(
        img_folder="fake", img_name="fake.jpg",
        cal_points_norm=cal_points_norm, dart_points_norm=dart_points_norm,
    )
    image = render_plane_image_at_size(app_config, image_size)
    rng = np.random.default_rng(0)

    samples = training_samples_from_row(image, row, app_config, rng, crop_size=128)

    assert len(samples) == 2
    for sample in samples:
        assert sample.image.shape == (128, 128, 3)
        assert len(sample.labels) >= 1


def test_no_darts_yields_no_training_samples(app_config):
    image_size = 800
    cal_points_norm = _cal_points_norm_matching_plane(app_config, image_size)
    row = DeepDartsRow(
        img_folder="fake", img_name="fake.jpg",
        cal_points_norm=cal_points_norm, dart_points_norm=np.zeros((0, 2)),
    )
    image = render_plane_image_at_size(app_config, image_size)
    rng = np.random.default_rng(0)

    samples = training_samples_from_row(image, row, app_config, rng)

    assert samples == []


def test_calibration_point_angles_are_four_distinct_boundaries():
    assert len(set(CALIBRATION_POINT_ANGLES_DEG)) == 4
    assert all(0.0 <= a < 360.0 for a in CALIBRATION_POINT_ANGLES_DEG)
