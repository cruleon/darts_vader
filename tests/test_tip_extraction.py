from __future__ import annotations

import numpy as np
import pytest

from dartvision.detection.tip_extraction import select_tip_point


def test_select_tip_point_picks_closest_to_center():
    center = (100.0, 100.0)
    points = np.array(
        [
            [100.0, 80.0],   # distanza 20 dal centro
            [130.0, 100.0],  # distanza 30
            [100.0, 100.0],  # distanza 0 -> deve vincere questo
            [200.0, 200.0],
        ]
    )

    x, y = select_tip_point(points, center)

    assert (x, y) == (100.0, 100.0)


def test_select_tip_point_on_circle_boundary_offsets_towards_center():
    """Per un blob circolare, il punto scelto e' sul bordo rivolto verso
    il centro del bersaglio, non il centroide del blob (approssima la
    punta, non il corpo della freccetta)."""
    center = (0.0, 0.0)
    blob_center = np.array([50.0, 0.0])
    radius = 6.0
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle_points = blob_center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

    x, y = select_tip_point(circle_points, center)

    assert x == pytest.approx(50.0 - radius, abs=0.5)
    assert y == pytest.approx(0.0, abs=0.5)


def test_select_tip_point_rejects_empty_input():
    with pytest.raises(ValueError):
        select_tip_point(np.empty((0, 2)), (0.0, 0.0))
