"""x01 games (301, 501, ...): scoring, busts, finishes and checkout suggestions."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..board import geometry as g

OK, BUST, WIN = "ok", "bust", "win"


@dataclass
class TurnResult:
    player: str
    darts: list[str]
    points: int  # points actually subtracted from the remaining score
    remaining: int
    outcome: str  # ok, bust, win


@dataclass
class X01:
    players: list[str]
    start: int = 301
    double_out: bool = False
    scores: list[int] = field(default_factory=list)
    current: int = 0
    legs: list[int] = field(default_factory=list)
    history: list[TurnResult] = field(default_factory=list)
    darts_thrown: list[int] = field(default_factory=list)
    points_scored: list[int] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.players)
        self.scores = self.scores or [self.start] * n
        self.legs = self.legs or [0] * n
        self.darts_thrown = self.darts_thrown or [0] * n
        self.points_scored = self.points_scored or [0] * n

    @property
    def player(self) -> str:
        return self.players[self.current]

    @property
    def remaining(self) -> int:
        return self.scores[self.current]

    def average(self, i: int) -> float:
        """Three-dart average of player `i`."""
        return 3 * self.points_scored[i] / self.darts_thrown[i] if self.darts_thrown[i] else 0.0

    def apply_turn(self, hits: list[g.Hit]) -> TurnResult:
        """Apply the darts of the current player's turn and pass the throw to the next player."""
        p, start = self.current, self.scores[self.current]
        rem, used = start, 0
        outcome = OK
        for hit in hits:
            used += 1
            after = rem - hit.score
            finishing_double = hit.multiplier == 2  # doubles and the bullseye
            if after < 0 or (self.double_out and after == 1) or (after == 0 and self.double_out and not finishing_double):
                outcome = BUST
                break
            rem = after
            if rem == 0:
                outcome = WIN
                break
        labels = [h.label for h in hits[:used]]
        self.darts_thrown[p] += used if outcome == WIN else max(used, 3)
        if outcome == BUST:
            result = TurnResult(self.player, labels, 0, start, outcome)
        elif outcome == WIN:
            self.points_scored[p] += start
            self.legs[p] += 1
            result = TurnResult(self.player, labels, start, 0, outcome)
            self.scores = [self.start] * len(self.players)
        else:
            self.points_scored[p] += start - rem
            self.scores[p] = rem
            result = TurnResult(self.player, labels, start - rem, rem, outcome)
        self.history.append(result)
        self.current = (p + 1) % len(self.players)  # after a won leg the winner does not throw first
        return result

    def checkout(self, remaining: int | None = None, darts: int = 3) -> str | None:
        rem = self.remaining if remaining is None else remaining
        return suggest_checkout(rem, darts, self.double_out)


def finishing_double(remaining: int) -> g.Hit | None:
    """The double (or bullseye) that finishes `remaining` with a single dart, if there is one."""
    if remaining == 50:
        return g.Hit(25, 2)
    if 2 <= remaining <= 40 and remaining % 2 == 0:
        return g.Hit(remaining // 2, 2)
    return None


def _targets() -> list[tuple[str, int]]:
    t = [(f"S{n}", n) for n in range(1, 21)] + [(f"D{n}", 2 * n) for n in range(1, 21)]
    t += [(f"T{n}", 3 * n) for n in range(1, 21)] + [("25", 25), ("BULL", 50)]
    return t


TARGETS = _targets()
FINISHES = [(label, v) for label, v in TARGETS if label.startswith("D") or label == "BULL"]


@lru_cache(maxsize=None)
def suggest_checkout(remaining: int, darts: int = 3, double_out: bool = False) -> str | None:
    """Shortest finish for `remaining` with at most `darts` darts; ties prefer high first darts."""
    if remaining <= 0 or remaining > (170 if double_out else 180):
        return None
    finals = FINISHES if double_out else TARGETS
    order = sorted(TARGETS, key=lambda t: -t[1])
    for n in range(1, darts + 1):
        if n == 1:
            for label, v in finals:
                if v == remaining:
                    return label
        elif n == 2:
            for a, va in order:
                for label, v in finals:
                    if va + v == remaining:
                        return f"{a} {label}"
        else:
            for a, va in order:
                for b, vb in order:
                    for label, v in finals:
                        if va + vb + v == remaining:
                            return f"{a} {b} {label}"
    return None
