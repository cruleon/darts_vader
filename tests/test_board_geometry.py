from __future__ import annotations

import pytest

from dartvision.scoring.board_geometry import (
    cartesian_to_polar,
    polar_to_cartesian,
    sector_center_angle,
    sector_number_at_angle,
)


def test_cartesian_to_polar_top_is_zero_degrees():
    angle, radius = cartesian_to_polar(0.0, -100.0)
    assert angle == pytest.approx(0.0)
    assert radius == pytest.approx(100.0)


def test_cartesian_to_polar_right_is_90_degrees():
    angle, radius = cartesian_to_polar(100.0, 0.0)
    assert angle == pytest.approx(90.0)
    assert radius == pytest.approx(100.0)


def test_cartesian_to_polar_bottom_is_180_degrees():
    angle, _radius = cartesian_to_polar(0.0, 100.0)
    assert angle == pytest.approx(180.0)


def test_cartesian_to_polar_left_is_270_degrees():
    angle, _radius = cartesian_to_polar(-100.0, 0.0)
    assert angle == pytest.approx(270.0)


def test_polar_cartesian_roundtrip():
    for angle in [0, 37, 90, 123.4, 200, 359]:
        for radius in [1.0, 50.0, 169.9]:
            x, y = polar_to_cartesian(angle, radius)
            back_angle, back_radius = cartesian_to_polar(x, y)
            assert back_angle == pytest.approx(angle % 360, abs=1e-6)
            assert back_radius == pytest.approx(radius, abs=1e-6)


def test_sector_20_is_at_top(app_config):
    # Con sector0_offset_deg=0.0 e sector_order[0]=20, il 20 e' esattamente in alto.
    sector = sector_number_at_angle(0.0, app_config.board)
    assert sector == 20


def test_sector_order_matches_official_dartboard(app_config):
    expected_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    sector_width = 360.0 / app_config.board.sector_count
    for i, expected_sector in enumerate(expected_order):
        angle = i * sector_width  # centro del settore i-esimo
        assert sector_number_at_angle(angle, app_config.board) == expected_sector


def test_sector_center_angle_is_inverse_of_sector_number_at_angle(app_config):
    for sector in app_config.board.sector_order:
        angle = sector_center_angle(sector, app_config.board)
        assert sector_number_at_angle(angle, app_config.board) == sector


def test_sector_boundaries_are_exclusive_on_the_upper_side(app_config):
    sector_width = 360.0 / app_config.board.sector_count
    # Appena prima del confine tra settore 0 e 1 -> ancora settore 0 (il 20).
    assert sector_number_at_angle(sector_width / 2 - 0.01, app_config.board) == 20
    # Esattamente al confine -> gia' il settore successivo (l'1).
    assert sector_number_at_angle(sector_width / 2, app_config.board) == 1
