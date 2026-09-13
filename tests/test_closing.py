from __future__ import annotations

import pytest

from dartvision.scoring.board_geometry import polar_to_cartesian, sector_center_angle
from dartvision.scoring.closing import (
    closing_distance_mm,
    double_target_point_mm,
    required_double_sector,
)


@pytest.mark.parametrize(
    "remaining,expected_sector",
    [
        (40, 20),
        (36, 18),
        (2, 1),
        (32, 16),
    ],
)
def test_required_double_sector_known_values(remaining, expected_sector):
    assert required_double_sector(remaining) == expected_sector


@pytest.mark.parametrize("remaining", [41, 1, 0, 42, 170, 501, 50])
def test_required_double_sector_none_when_not_a_single_double_checkout(remaining):
    assert required_double_sector(remaining) is None


def test_closing_distance_zero_at_exact_target(app_config):
    board = app_config.board
    remaining = 40  # richiede il doppio 20
    target_x, target_y = double_target_point_mm(20, board)

    distance = closing_distance_mm(target_x, target_y, remaining, board)

    assert distance == pytest.approx(0.0, abs=1e-9)


def test_closing_distance_measures_offset_from_target(app_config):
    board = app_config.board
    remaining = 40
    target_x, target_y = double_target_point_mm(20, board)

    distance = closing_distance_mm(target_x + 5.0, target_y, remaining, board)

    assert distance == pytest.approx(5.0, abs=1e-6)


def test_closing_distance_none_when_remaining_not_a_single_double(app_config):
    board = app_config.board
    assert closing_distance_mm(0.0, 0.0, 501, board) is None
    assert closing_distance_mm(0.0, 0.0, 50, board) is None
    assert closing_distance_mm(0.0, 0.0, 41, board) is None


def test_double_target_point_is_on_sector_axis(app_config):
    board = app_config.board
    x, y = double_target_point_mm(6, board)
    angle = sector_center_angle(6, board)
    expected_x, expected_y = polar_to_cartesian(
        angle, (board.double_inner_radius_mm + board.double_outer_radius_mm) / 2
    )
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(expected_y)
