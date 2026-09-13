from __future__ import annotations

from dartvision.scoring.board_geometry import polar_to_cartesian, sector_center_angle
from dartvision.scoring.score import (
    RING_BULL,
    RING_BULLSEYE,
    RING_DOUBLE,
    RING_MISS,
    RING_SINGLE_INNER,
    RING_SINGLE_OUTER,
    RING_TRIPLE,
    score_point,
)


def test_center_is_bullseye_50(app_config):
    result = score_point(0.0, 0.0, app_config.board)
    assert result.ring == RING_BULLSEYE
    assert result.sector is None
    assert result.points == 50


def test_outer_bull_is_25(app_config):
    board = app_config.board
    mid_radius = (board.inner_bull_radius_mm + board.outer_bull_radius_mm) / 2
    result = score_point(0.0, -mid_radius, board)
    assert result.ring == RING_BULL
    assert result.points == 25


def test_triple_20_scores_60(app_config):
    board = app_config.board
    angle = sector_center_angle(20, board)
    mid_radius = (board.triple_inner_radius_mm + board.triple_outer_radius_mm) / 2
    x, y = polar_to_cartesian(angle, mid_radius)

    result = score_point(x, y, board)

    assert result.sector == 20
    assert result.ring == RING_TRIPLE
    assert result.multiplier == 3
    assert result.points == 60


def test_double_3_scores_6(app_config):
    board = app_config.board
    angle = sector_center_angle(3, board)
    mid_radius = (board.double_inner_radius_mm + board.double_outer_radius_mm) / 2
    x, y = polar_to_cartesian(angle, mid_radius)

    result = score_point(x, y, board)

    assert result.sector == 3
    assert result.ring == RING_DOUBLE
    assert result.multiplier == 2
    assert result.points == 6


def test_single_inner_area(app_config):
    board = app_config.board
    angle = sector_center_angle(19, board)
    mid_radius = (board.outer_bull_radius_mm + board.triple_inner_radius_mm) / 2
    x, y = polar_to_cartesian(angle, mid_radius)

    result = score_point(x, y, board)

    assert result.sector == 19
    assert result.ring == RING_SINGLE_INNER
    assert result.multiplier == 1
    assert result.points == 19


def test_single_outer_area(app_config):
    board = app_config.board
    angle = sector_center_angle(11, board)
    mid_radius = (board.triple_outer_radius_mm + board.double_inner_radius_mm) / 2
    x, y = polar_to_cartesian(angle, mid_radius)

    result = score_point(x, y, board)

    assert result.sector == 11
    assert result.ring == RING_SINGLE_OUTER
    assert result.multiplier == 1
    assert result.points == 11


def test_outside_double_ring_is_miss(app_config):
    board = app_config.board
    result = score_point(0.0, -(board.double_outer_radius_mm + 5.0), board)
    assert result.ring == RING_MISS
    assert result.sector is None
    assert result.multiplier == 0
    assert result.points == 0


def test_exactly_on_double_outer_edge_still_counts(app_config):
    board = app_config.board
    angle = sector_center_angle(20, board)
    x, y = polar_to_cartesian(angle, board.double_outer_radius_mm)
    result = score_point(x, y, board)
    assert result.ring == RING_DOUBLE
    assert result.sector == 20
