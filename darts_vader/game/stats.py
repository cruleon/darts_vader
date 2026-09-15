"""Match statistics computed from the confirmed turns of an x01 game."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .x01 import BUST, WIN


@dataclass
class DartRecord:
    label: str
    score: int
    number: int
    multiplier: int
    x_mm: float
    y_mm: float
    at_double: bool = False  # the score left before this dart could be finished with a double
    finish_distance_mm: float | None = None  # distance from that finishing double (double-out games)


@dataclass
class TurnRecord:
    player: int
    leg: int
    start: int  # score left before the turn
    points: int
    outcome: str
    darts_counted: int  # darts counted for averages (3 unless the leg was won earlier)
    darts: list[DartRecord] = field(default_factory=list)


def _three_dart_average(points: int, darts: int) -> float:
    return round(3 * points / darts, 2) if darts else 0.0


def _percent(part: int, total: int) -> float | None:
    return round(100 * part / total, 1) if total else None


def player_stats(index: int, name: str, turns: list[TurnRecord], legs_won: int, double_out: bool) -> dict:
    own = [t for t in turns if t.player == index]
    darts = [d for t in own for d in t.darts]
    by_leg: dict[int, list[TurnRecord]] = defaultdict(list)
    for t in own:
        by_leg[t.leg].append(t)
    first_nine = [t for leg in by_leg.values() for t in leg[:3]]
    wins = [t for t in own if t.outcome == WIN]
    points = [t.points for t in own]

    trebles = sum(d.multiplier == 3 for d in darts)
    doubles = sum(d.multiplier == 2 and d.number != 25 for d in darts)
    bulls = sum(d.number == 25 for d in darts)
    misses = sum(d.multiplier == 0 for d in darts)
    favourite = Counter(d.number for d in darts if d.multiplier > 0).most_common(1)
    at_double = sum(d.at_double for d in darts)
    distances = [d.finish_distance_mm for d in darts if d.finish_distance_mm is not None]

    centroid = grouping = None
    if len(darts) >= 2:
        xy = np.array([[d.x_mm, d.y_mm] for d in darts])
        c = xy.mean(axis=0)
        centroid = [round(float(c[0]), 1), round(float(c[1]), 1)]
        grouping = round(float(np.linalg.norm(xy - c, axis=1).mean()), 1)

    return dict(
        index=index,
        name=name,
        legs_won=legs_won,
        turns=len(own),
        darts_thrown=sum(t.darts_counted for t in own),
        points=sum(points),
        average=_three_dart_average(sum(points), sum(t.darts_counted for t in own)),
        first9_average=_three_dart_average(sum(t.points for t in first_nine), sum(t.darts_counted for t in first_nine)),
        highest_turn=max(points, default=0),
        scores_180=sum(p == 180 for p in points),
        scores_140=sum(140 <= p < 180 for p in points),
        scores_100=sum(100 <= p < 140 for p in points),
        scores_60=sum(60 <= p < 100 for p in points),
        busts=sum(t.outcome == BUST for t in own),
        highest_checkout=max((t.start for t in wins), default=None),
        best_leg_darts=min((sum(t.darts_counted for t in by_leg[w.leg]) for w in wins), default=None),
        darts_at_double=at_double if double_out else None,
        checkout_pct=_percent(len(wins), at_double) if double_out else None,
        trebles=trebles,
        doubles=doubles,
        bulls=bulls,
        misses=misses,
        singles=len(darts) - trebles - doubles - bulls - misses,
        treble_pct=_percent(trebles, len(darts)),
        double_pct=_percent(doubles, len(darts)),
        miss_pct=_percent(misses, len(darts)),
        favourite=("Bull" if favourite[0][0] == 25 else str(favourite[0][0])) if favourite else None,
        favourite_hits=favourite[0][1] if favourite else 0,
        avg_finish_distance_mm=round(float(np.mean(distances)), 1) if distances else None,
        best_finish_distance_mm=round(float(min(distances)), 1) if distances else None,
        centroid_mm=centroid,
        grouping_mm=grouping,
        darts=[dict(x=round(d.x_mm, 1), y=round(d.y_mm, 1), label=d.label, score=d.score) for d in darts],
    )


def match_summary(players: list[str], turns: list[TurnRecord], legs_won: list[int], start: int, double_out: bool,
                  legs_to_win: int, winner: int) -> dict:
    return dict(
        start=start,
        double_out=double_out,
        legs_to_win=legs_to_win,
        winner=winner,
        legs_played=max((t.leg for t in turns), default=0),
        players=[player_stats(i, name, turns, legs_won[i], double_out) for i, name in enumerate(players)],
    )
