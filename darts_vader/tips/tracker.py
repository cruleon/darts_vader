"""Live scoring: darts and turns from the continuous camera stream.

On every frame TipNet finds the tips on the board. A tip becomes a dart once it stays in the same
place in almost all of the recent frames, which filters out isolated false positives, hands and
motion blur. When the board is visible but no tip has been seen for a while, the darts have been
pulled: with ``auto_close`` the turn is closed, otherwise a ``board_empty`` event lets the caller
decide (for example after a review of the turn).

Manual corrections: :meth:`LiveScorer.remove_dart` removes a ghost dart and turns that spot into
an ignored zone (it is not picked up again and does not prevent the turn from closing);
:meth:`LiveScorer.add_dart` adds a dart the model missed; :meth:`LiveScorer.move_dart` moves a
tip. Ignored zones persist until :meth:`LiveScorer.clear_ignored`.
:meth:`LiveScorer.wait_for_empty_board` clears the turn and ignores every tip until the board is
free, so darts that were just confirmed do not leak into the next turn.

Board detection (and orientation) is the caller's job: ``update`` receives the frame and the
board of that frame, or None when the board is not visible.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..board import geometry as g
from ..board.detector import BoardState
from .model import predict_tips

# Dart origins
MODEL, CORRECTED, CLICK, APPROX = "model", "corrected", "click", "approx"


@dataclass
class LiveConfig:
    tip_threshold: float = 0.4  # minimum model confidence of a tip
    max_radius_mm: float = g.R_BOARD  # tips outside the surround are ignored
    window: int = 8  # recent frames considered for stability
    min_hits: int = 6  # frames of the window in which a tip must appear
    cluster_mm: float = 4.0  # same tip across frames
    same_dart_mm: float = 8.0  # tip already registered as a dart
    ignore_mm: float = 8.0  # radius of ignored zones
    max_darts: int = 3
    empty_seconds: float = 2.0  # board visible without tips for this long -> darts pulled
    ready_seconds: float = 1.0  # after a confirmation: free board for this long -> next turn


@dataclass
class Dart:
    tip_mm: np.ndarray
    hit: g.Hit
    confidence: float
    origin: str = MODEL  # model, corrected (moved by hand), click, approx (placed on the mini board)

    @property
    def manual(self) -> bool:
        return self.origin != MODEL


@dataclass
class LiveEvent:
    kind: str  # dart, turn_complete, turn_closed, board_empty, ready, extra_dart
    message: str
    darts: list[Dart] = field(default_factory=list)


class LiveScorer:
    def __init__(self, model, device: str = "cuda", view: str = "camera", config: LiveConfig | None = None,
                 auto_close: bool = True):
        self.cfg = config or LiveConfig()
        self.model, self.device, self.view = model, device, view
        self.auto_close = auto_close
        self.turn: list[Dart] = []
        self.turns: list[list[Dart]] = []
        self.ignored: list[np.ndarray] = []  # zones (mm) where tips are ignored
        self.detections = np.zeros((0, 3))  # tips of the last frame: x mm, y mm, confidence
        self.waiting_empty = False
        self._history: deque[np.ndarray] = deque(maxlen=self.cfg.window)
        self._empty_since: float | None = None
        self._empty_notified = False
        self._extra_warned = False
        self._rings = g.RING_RADII

    # ------------------------------------------------------------------ commands

    @property
    def turn_total(self) -> int:
        return sum(d.hit.score for d in self.turn)

    def remove_dart(self, index: int) -> Dart | None:
        """Remove a dart of the current turn; a model detection also becomes an ignored zone."""
        if not 0 <= index < len(self.turn):
            return None
        dart = self.turn.pop(index)
        if dart.origin == MODEL:
            self.ignored.append(dart.tip_mm.copy())
        self._history.clear()
        self._extra_warned = False
        return dart

    def undo(self) -> Dart | None:
        return self.remove_dart(len(self.turn) - 1)

    def add_dart(self, tip_mm, origin: str = CLICK) -> Dart | None:
        """Add a dart by hand at the given board position (mm)."""
        if len(self.turn) >= self.cfg.max_darts:
            return None
        p = np.asarray(tip_mm, float)
        dart = Dart(p, g.score_model_point(*p, self._rings), 1.0, origin)
        self.turn.append(dart)
        return dart

    def move_dart(self, index: int, tip_mm, origin: str = CORRECTED) -> Dart | None:
        if not 0 <= index < len(self.turn):
            return None
        dart = self.turn[index]
        dart.tip_mm = np.asarray(tip_mm, float)
        dart.hit = g.score_model_point(*dart.tip_mm, self._rings)
        dart.origin = origin
        return dart

    def nearest_dart(self, tip_mm, max_mm: float) -> int | None:
        if not self.turn:
            return None
        d = [np.linalg.norm(x.tip_mm - np.asarray(tip_mm, float)) for x in self.turn]
        k = int(np.argmin(d))
        return k if d[k] <= max_mm else None

    def clear_ignored(self) -> int:
        n = len(self.ignored)
        self.ignored = []
        return n

    def close_turn(self) -> LiveEvent | None:
        darts = self.turn
        if darts:
            self.turns.append(darts)
        self.turn, self._extra_warned, self._empty_notified = [], False, False
        self._history.clear()
        self._empty_since = None
        if not darts:
            return None
        labels = " ".join(d.hit.label for d in darts)
        return LiveEvent("turn_closed", f"turn closed: {labels} = {sum(d.hit.score for d in darts)}", darts)

    def wait_for_empty_board(self) -> None:
        """Close the turn and ignore every tip until the board stays free for ``ready_seconds``."""
        self.close_turn()
        self.waiting_empty = True

    # ------------------------------------------------------------------ frame loop

    def update(self, frame: np.ndarray, board: BoardState | None, now: float) -> list[LiveEvent]:
        cfg = self.cfg
        if board is None:  # board not visible: no decisions
            self.detections = np.zeros((0, 3))
            self._empty_since = None
            return []

        self._rings = board.rings
        tips = predict_tips(self.model, frame, board, self.view, cfg.tip_threshold, max_tips=8, device=self.device)
        if len(tips):
            keep = np.hypot(tips[:, 0], tips[:, 1]) <= cfg.max_radius_mm
            for z in self.ignored:
                keep &= np.hypot(tips[:, 0] - z[0], tips[:, 1] - z[1]) > cfg.ignore_mm
            tips = tips[keep]
        self.detections = tips

        events: list[LiveEvent] = []
        if self.waiting_empty:
            if len(tips):
                self._empty_since = None
            elif self._empty_since is None:
                self._empty_since = now
            elif now - self._empty_since >= cfg.ready_seconds:
                self.waiting_empty, self._empty_since = False, None
                self._history.clear()
                events.append(LiveEvent("ready", "board is free: next turn"))
            return events

        self._history.append(tips)
        if len(tips) == 0:
            if self._empty_since is None:
                self._empty_since = now
            elif self.turn and now - self._empty_since >= cfg.empty_seconds:
                if self.auto_close:
                    events.append(self.close_turn())
                elif not self._empty_notified:
                    self._empty_notified = True
                    events.append(LiveEvent("board_empty", "darts pulled", list(self.turn)))
            return events
        self._empty_since = None

        for x, y, conf in self._stable_tips():
            p = np.array([x, y])
            k = self.nearest_dart(p, cfg.same_dart_mm)
            if k is not None:  # refine the position of a registered dart (never of manual ones)
                d = self.turn[k]
                if not d.manual:
                    d.tip_mm = 0.8 * d.tip_mm + 0.2 * p
                    d.hit = g.score_model_point(*d.tip_mm, self._rings)
                continue
            if len(self.turn) >= cfg.max_darts:
                if not self._extra_warned:
                    self._extra_warned = True
                    events.append(LiveEvent("extra_dart", "a 4th dart is visible: pull the darts or remove the ghost"))
                continue
            hit = g.score_model_point(x, y, self._rings)
            dart = Dart(p, hit, float(conf))
            self.turn.append(dart)
            events.append(LiveEvent("dart", f"dart {len(self.turn)}: {hit.label} ({hit.score})", [dart]))
            if len(self.turn) == cfg.max_darts:
                labels = " ".join(d.hit.label for d in self.turn)
                events.append(LiveEvent("turn_complete", f"turn: {labels} = {self.turn_total}", list(self.turn)))
        return events

    def _stable_tips(self) -> list[tuple[float, float, float]]:
        """Tips of the last frame present (within ``cluster_mm``) in at least ``min_hits`` recent frames."""
        cfg = self.cfg
        if len(self._history) < cfg.min_hits:
            return []
        stable = []
        for x, y, conf in sorted(self._history[-1], key=lambda t: -t[2]):
            points, confs = [], []
            for frame_tips in self._history:
                if len(frame_tips) == 0:
                    continue
                d = np.hypot(frame_tips[:, 0] - x, frame_tips[:, 1] - y)
                k = int(np.argmin(d))
                if d[k] <= cfg.cluster_mm:
                    points.append(frame_tips[k, :2])
                    confs.append(frame_tips[k, 2])
            if len(points) >= cfg.min_hits:
                m = np.mean(points, axis=0)
                if all(np.hypot(m[0] - s[0], m[1] - s[1]) > cfg.cluster_mm for s in stable):
                    stable.append((float(m[0]), float(m[1]), float(np.mean(confs))))
        return stable
