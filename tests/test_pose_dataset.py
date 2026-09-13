from __future__ import annotations

import numpy as np
import pytest

from dartvision.synthetic.pose_dataset import (
    DartLabel,
    render_training_sample,
    sample_dart_placements,
    to_yolo_pose_label,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_sample_dart_placements_returns_requested_count(app_config, rng):
    placements = sample_dart_placements(rng, app_config, n_darts=3)

    assert len(placements) == 3
    for x_mm, y_mm, angle_deg in placements:
        assert isinstance(x_mm, float)
        assert isinstance(y_mm, float)
        assert 0.0 <= angle_deg < 360.0


def test_sample_dart_placements_stays_within_playable_radius(app_config, rng):
    max_radius = app_config.board.double_outer_radius_mm
    placements = sample_dart_placements(rng, app_config, n_darts=20)

    for x_mm, y_mm, _ in placements:
        # Le freccette "clustered" possono uscire leggermente dal raggio
        # di gioco (offset fino a 35mm dalla base): margine tollerato.
        assert (x_mm**2 + y_mm**2) ** 0.5 <= max_radius + 35.0


def test_render_training_sample_produces_crop_of_requested_size(app_config, rng):
    sample = render_training_sample(app_config, rng, crop_size=128, max_darts_per_crop=2)

    assert sample.image.shape == (128, 128, 3)
    assert len(sample.labels) >= 1  # almeno la freccetta target e' nel ritaglio


def test_render_training_sample_labels_are_within_crop_bounds(app_config, rng):
    crop_size = 160
    sample = render_training_sample(app_config, rng, crop_size=crop_size, max_darts_per_crop=3)

    for label in sample.labels:
        x_min, y_min, x_max, y_max = label.bbox_px
        assert 0.0 <= x_min < x_max <= crop_size
        assert 0.0 <= y_min < y_max <= crop_size
        tip_x, tip_y = label.tip_px
        assert 0.0 <= tip_x <= crop_size
        assert 0.0 <= tip_y <= crop_size


def test_to_yolo_pose_label_is_normalized_and_well_formed():
    label = DartLabel(bbox_px=(10.0, 20.0, 30.0, 60.0), tip_px=(15.0, 25.0))

    line = to_yolo_pose_label(label, crop_size=100)
    parts = line.split()

    assert len(parts) == 8
    assert parts[0] == "0"
    cx, cy, w, h, kpx, kpy, kpv = (float(p) for p in parts[1:])
    assert cx == pytest.approx(0.20)
    assert cy == pytest.approx(0.40)
    assert w == pytest.approx(0.20)
    assert h == pytest.approx(0.40)
    assert kpx == pytest.approx(0.15)
    assert kpy == pytest.approx(0.25)
    assert kpv == 2.0
