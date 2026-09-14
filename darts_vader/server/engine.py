"""Game engine of the web app: board lock, live dart detection, x01 rules, turn review and
training-data collection.

The frontend reads the engine through :meth:`GameEngine.state` and drives it with
:meth:`GameEngine.command`. "Image" coordinates are always pixels of the original camera frame
(of the still photo while a turn is under review).

Modes: ``searching`` -> ``calibrating`` (the player confirms where the 20 is) -> ``playing`` <-> ``review``.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..board import BoardState, DartboardDetector, geometry as g
from ..game.x01 import BUST, WIN, X01
from ..labels import LABELLED, NO_TIPS, LabelSession, tip_record
from ..tips.tracker import APPROX, CLICK, Dart, LiveConfig, LiveScorer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = PROJECT_ROOT / "webcam_calibration.json"

SEARCHING, CALIBRATING, PLAYING, REVIEW = "searching", "calibrating", "playing", "review"
LOCK_FRAMES = 5  # consecutive confident detections needed to lock the board
LOST_FRAMES = 10  # frames without a board before the player is warned
REVIEW_DELAY = 1.0  # seconds after the third dart before the automatic review
MAX_PLAYERS = 6
START_SCORES = (101, 170, 301, 501, 701, 1001)
EFFECTS = ("180", "bust", "win", "ton")


def _xy(p) -> list[float]:
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


class GameEngine:
    def __init__(self, model, view: str, device: str, players: list[str], start: int = 301,
                 double_out: bool = False, labels_dir: Path = PROJECT_ROOT / "webcam_labels",
                 threshold: float = 0.4, calibration_file: Path | None = CALIBRATION_FILE,
                 save_calibration: bool = True, debug: bool = False):
        self.lock = threading.RLock()
        self.game = X01(players[:MAX_PLAYERS] or ["Player"], start, double_out)
        self.scorer = LiveScorer(model, device, view, LiveConfig(tip_threshold=threshold), auto_close=False)
        self.detector = DartboardDetector()
        self.labels = LabelSession(Path(labels_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{start}",
                                   {"game": start, "view": view})
        self.calibration_file = calibration_file
        self.save_calibration = save_calibration
        self.debug = debug

        self.mode = SEARCHING
        self.board: BoardState | None = None  # confirmed orientation
        self.candidate: BoardState | None = None  # locked board waiting for the 20 to be confirmed
        self.saved_orientation = self._load_orientation()
        self.lock_count = 0
        self.lost_frames = 0
        self.reset_detector = False
        self.show_detections = False

        self.latest_frame: np.ndarray | None = None
        self.frame_size: tuple[int, int] | None = None
        self.snapshot: tuple[np.ndarray, BoardState] | None = None  # last frame with every dart visible
        self.last_snapshot_time = 0.0
        self.last_dart_time: float | None = None
        self.review: dict | None = None
        self.review_id = 0
        self.review_jpeg: bytes | None = None
        self.fps_ai = 0.0

        self.events: deque = deque(maxlen=40)
        self._event_id = 0

    @property
    def session_dir(self) -> Path:
        return self.labels.dir

    # ------------------------------------------------------------------ events for the frontend

    def _emit(self, kind: str, **data) -> None:
        self._event_id += 1
        self.events.append(dict(id=self._event_id, kind=kind, **data))

    def _emit_dart(self, index: int, dart: Dart) -> None:
        self._emit("dart", index=index, label=dart.hit.label, number=int(dart.hit.number),
                   multiplier=int(dart.hit.multiplier), origin=dart.origin)

    def toast(self, text: str, tone: str = "info") -> None:
        self._emit("toast", text=text, tone=tone)
        print(text, flush=True)

    # ------------------------------------------------------------------ calibration file

    def _load_orientation(self) -> BoardState | None:
        if self.calibration_file is None or not self.calibration_file.exists():
            return None
        try:
            H = np.array(json.loads(self.calibration_file.read_text())["board_H"], float)
            return BoardState(H, 1.0, 0.0, 0, False)
        except (OSError, ValueError, KeyError, np.linalg.LinAlgError):
            return None

    def _store_orientation(self, board: BoardState) -> None:
        if self.calibration_file is not None and self.save_calibration:
            self.calibration_file.write_text(json.dumps({"board_H": np.round(board.H, 9).tolist()}))

    # ------------------------------------------------------------------ frame processing

    def process_frame(self, frame: np.ndarray, now: float) -> None:
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.latest_frame = frame
        if self.mode == REVIEW:
            return
        if self.reset_detector:
            self.detector.reset()
            self.reset_detector = False
        detected = self.detector.process(frame)
        with self.lock:
            if self.mode == SEARCHING:
                self._search(detected)
            elif self.mode == CALIBRATING:
                if detected is not None and self.candidate is not None:
                    self.candidate = detected.aligned_to(self.candidate)
            elif self.mode == PLAYING and self.board is not None:
                self._play(frame, detected, now)

    def _search(self, detected: BoardState | None) -> None:
        confident = detected is not None and detected.confidence >= 0.7
        self.lock_count = self.lock_count + 1 if confident else 0
        if self.lock_count >= LOCK_FRAMES:
            self.candidate = detected.aligned_to(self.saved_orientation) if self.saved_orientation else detected
            self.mode = CALIBRATING
            self._emit("board_found")
            self.toast("Board found: check the 20 and press ENTER", "violet")

    def _play(self, frame: np.ndarray, detected: BoardState | None, now: float) -> None:
        if detected is not None:
            self.board, self.lost_frames = detected.aligned_to(self.board), 0
        else:
            self.lost_frames += 1
        scorer = self.scorer
        for event in scorer.update(frame, self.board if detected is not None else None, now):
            if event.kind == "dart":
                self.last_dart_time = now
                self._emit_dart(len(scorer.turn) - 1, event.darts[0])
            elif event.kind == "board_empty" and scorer.turn:
                self.start_review("darts_pulled")
                return
            elif event.kind == "ready":
                self.toast(f"{self.game.player} to throw · {self.game.remaining} left", "success")
            elif event.kind == "extra_dart":
                self.toast("I can see a 4th dart: pull your darts or remove the ghost", "warning")

        if not scorer.turn:
            self.snapshot = None
        elif detected is not None and now - self.last_snapshot_time > 0.25 and self._all_darts_visible():
            # keep a photo of the turn without hands in front of the board
            self.snapshot, self.last_snapshot_time = (frame.copy(), self.board), now
        if (len(scorer.turn) == scorer.cfg.max_darts and self.last_dart_time is not None
                and now - self.last_dart_time >= REVIEW_DELAY and self.snapshot is not None):
            self.last_dart_time = None
            self.start_review("three_darts")

    def _all_darts_visible(self) -> bool:
        det = self.scorer.detections
        return all(d.manual or (len(det) > 0 and np.min(np.hypot(det[:, 0] - d.tip_mm[0], det[:, 1] - d.tip_mm[1])) < 6)
                   for d in self.scorer.turn)

    # ------------------------------------------------------------------ review and confirmation

    def start_review(self, reason: str) -> bool:
        with self.lock:
            if self.mode != PLAYING:
                return False
            snap = self.snapshot
            if snap is None and self.latest_frame is not None and self.board is not None:
                snap = (self.latest_frame.copy(), self.board)
            if snap is None:
                return False
            ok, jpg = cv2.imencode(".jpg", snap[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.review_id += 1
            self.review = dict(frame=snap[0], board=snap[1], reason=reason, id=self.review_id)
            self.review_jpeg = jpg.tobytes() if ok else None
            self.mode = REVIEW
            self._emit("review", reason=reason)
            return True

    def confirm(self, save: bool) -> None:
        """Apply the reviewed turn to the game, optionally saving it as a training sample."""
        with self.lock:
            if self.mode != REVIEW:
                return
            index = self._save_turn() if save else None
            before = self.game.remaining
            result = self.game.apply_turn([d.hit for d in self.scorer.turn])
            self.scorer.wait_for_empty_board()
            self.snapshot, self.review, self.last_dart_time, self.mode = None, None, None, PLAYING
            self._emit("turn", player=result.player, darts=result.darts, points=result.points,
                       remaining=result.remaining, outcome=result.outcome)
            if result.outcome == BUST:
                self._emit("effect", effect="bust", text=result.player)
                self.toast(f"BUST · {result.player} stays on {before}", "danger")
            elif result.outcome == WIN:
                self._emit("effect", effect="win", text=result.player)
                self.toast(f"GAME SHOT! {result.player} checks out with {' '.join(result.darts)}", "gold")
            else:
                if result.points == 180:
                    self._emit("effect", effect="180", text="180")
                elif result.points >= 100:
                    self._emit("effect", effect="ton", text=str(result.points))
                self.toast(f"{result.player}: {result.points} scored · {result.remaining} left", "success")
            if index is not None:
                self.toast(f"Training sample {index} saved", "info")

    def _save_turn(self) -> int:
        frame, board = self.review["frame"], self.review["board"]
        tips = [tip_record(d.tip_mm, board, d.hit.label, d.origin, d.confidence)
                for d in self.scorer.turn if d.origin != APPROX]
        entry = self.labels.add_photo(frame, board, tips, LABELLED if tips else NO_TIPS)
        return entry["index"]

    # ------------------------------------------------------------------ commands from the frontend

    def command(self, msg: dict) -> None:
        handler = getattr(self, f"_cmd_{msg.get('type')}", None)
        if handler is not None:
            with self.lock:
                handler(msg)

    def _image_to_model(self, board: BoardState, msg: dict) -> np.ndarray:
        return board.to_model([(float(msg["x"]), float(msg["y"]))])[0]

    def _cmd_set20(self, msg: dict) -> None:
        if self.mode == CALIBRATING and self.candidate is not None:
            theta = g.polar(*self._image_to_model(self.candidate, msg))[1]
            self.candidate = self.candidate.rotated(g.sector_index(theta))
            self._emit("set20")
            self.toast("20 set: press ENTER to start", "violet")

    def _cmd_confirm_orientation(self, _msg: dict) -> None:
        if self.mode == CALIBRATING and self.candidate is not None:
            self.board, self.mode = self.candidate, PLAYING
            self._store_orientation(self.board)
            self._emit("game_start")
            self.toast(f"Game on: {self.game.start}! {self.game.player} throws first", "success")

    def _cmd_start_review(self, _msg: dict) -> None:
        if not self.start_review("manual"):
            self.toast("Review not available right now", "warning")

    def _cmd_resume(self, _msg: dict) -> None:
        if self.mode == REVIEW:
            self.review, self.mode = None, PLAYING

    def _cmd_confirm(self, msg: dict) -> None:
        self.confirm(bool(msg.get("save", True)))

    def _cmd_undo(self, _msg: dict) -> None:
        if self.mode in (PLAYING, REVIEW):
            dart = self.scorer.undo()
            if dart:
                self._emit("removed", label=dart.hit.label)
            self.toast(f"Undid {dart.hit.label}" if dart else "No dart to undo", "warning")

    def _cmd_remove(self, msg: dict) -> None:
        if self.mode in (PLAYING, REVIEW):
            index = int(msg["index"])
            dart = self.scorer.remove_dart(index)
            if dart:
                self._emit("removed", label=dart.hit.label)
                self.toast(f"Removed dart {index + 1} ({dart.hit.label})", "warning")

    def _cmd_move(self, msg: dict) -> None:
        if self.mode == REVIEW:
            index = int(msg["index"])
            dart = self.scorer.move_dart(index, self._image_to_model(self.review["board"], msg))
            if dart:
                self._emit("moved", index=index, label=dart.hit.label)
                self.toast(f"Dart {index + 1} moved · {dart.hit.label}", "success")

    def _cmd_add(self, msg: dict) -> None:
        board = self.review["board"] if self.mode == REVIEW else self.board if self.mode == PLAYING else None
        if board is not None:
            self._add_dart(self._image_to_model(board, msg), CLICK, "Added")

    def _cmd_add_sim(self, msg: dict) -> None:
        tip = np.array([float(msg["x_mm"]), float(msg["y_mm"])])
        if self.mode in (PLAYING, REVIEW) and np.hypot(*tip) <= g.R_DOUBLE_OUT + 15:
            self._add_dart(tip, APPROX, "Added (approx.)")

    def _add_dart(self, tip_mm: np.ndarray, origin: str, verb: str) -> None:
        dart = self.scorer.add_dart(tip_mm, origin)
        if dart is None:
            self.toast("This turn already has 3 darts", "warning")
            return
        self._emit_dart(len(self.scorer.turn) - 1, dart)
        self.toast(f"{verb} · {dart.hit.label}", "success")

    def _cmd_clear_ignored(self, _msg: dict) -> None:
        self.toast(f"Cleared {self.scorer.clear_ignored()} ignored spots", "warning")

    def _cmd_toggle_detections(self, _msg: dict) -> None:
        self.show_detections = not self.show_detections

    def _cmd_recalibrate(self, _msg: dict) -> None:
        if self.mode != REVIEW:
            self.mode, self.board, self.candidate, self.lock_count = SEARCHING, None, None, 0
            self.reset_detector = True
            self.toast("Recalibrating: looking for the board", "violet")

    def _cmd_debug_effect(self, msg: dict) -> None:
        """Trigger a celebration effect without playing (only with --debug, used by UI tests)."""
        effect = str(msg.get("effect", "180"))
        if self.debug and effect in EFFECTS:
            self._emit("effect", effect=effect, text=str(msg.get("text", self.game.player)))

    def _cmd_new_game(self, msg: dict) -> None:
        players = [str(p).strip()[:18] for p in msg.get("players", []) if str(p).strip()][:MAX_PLAYERS]
        start = int(msg.get("start", self.game.start))
        if not players or start not in START_SCORES:
            self.toast("Invalid game settings", "danger")
            return
        self.game = X01(players, start, bool(msg.get("double_out", False)))
        self.labels.metadata["game"] = start
        if self.mode == REVIEW:
            self.review, self.mode = None, PLAYING
        if self.mode == PLAYING:
            self.scorer.wait_for_empty_board()
        else:
            self.scorer.close_turn()
        self._emit("new_game")
        self.toast(f"New {start} game · {self.game.player} to throw", "info")

    # ------------------------------------------------------------------ state for the frontend

    def state(self) -> dict:
        with self.lock:
            mode, game, scorer = self.mode, self.game, self.scorer
            board = {CALIBRATING: self.candidate, PLAYING: self.board,
                     REVIEW: self.review["board"] if self.review else None}.get(mode)
            turn = [dict(index=k, label=d.hit.label, score=int(d.hit.score), number=int(d.hit.number),
                         multiplier=int(d.hit.multiplier), tip_mm=_xy(d.tip_mm),
                         tip_img=_xy(board.to_image([d.tip_mm])[0]) if board is not None else None,
                         origin=d.origin, confidence=round(float(d.confidence), 3))
                    for k, d in enumerate(scorer.turn)]
            total = sum(t["score"] for t in turn)
            remaining = game.remaining - total
            bust = remaining < 0 or (game.double_out and remaining == 1)
            checkout = (game.checkout(remaining, 3 - len(turn))
                        if mode == PLAYING and remaining > 0 and len(turn) < 3 else None)
            detections = scorer.detections if self.show_detections and mode == PLAYING else np.zeros((0, 3))
            return dict(
                mode=mode,
                frame=dict(w=self.frame_size[0], h=self.frame_size[1]) if self.frame_size else None,
                board=dict(H_inv=np.asarray(board.H_inv, float).tolist(), rings=[float(r) for r in board.rings])
                if board is not None else None,
                game=dict(start=game.start, double_out=game.double_out, current=game.current,
                          players=[dict(name=name, score=int(game.scores[i]), legs=int(game.legs[i]),
                                        average=round(float(game.average(i)), 1))
                                   for i, name in enumerate(game.players)],
                          history=[dict(player=r.player, darts=r.darts, points=r.points, outcome=r.outcome)
                                   for r in game.history[-8:]]),
                turn=turn,
                turn_total=total,
                remaining_after=remaining,
                bust=bust,
                checkout=checkout,
                waiting=bool(scorer.waiting_empty) and mode == PLAYING,
                board_lost=self.lost_frames >= LOST_FRAMES and mode == PLAYING,
                ignored=[_xy(z) for z in scorer.ignored],
                detections=[[round(float(x), 1), round(float(y), 1)] for x, y, _ in detections],
                show_detections=self.show_detections,
                review=dict(id=self.review["id"], reason=self.review["reason"],
                            w=int(self.review["frame"].shape[1]), h=int(self.review["frame"].shape[0]))
                if self.review else None,
                fps_ai=round(float(self.fps_ai), 1),
                saved=self.labels.count(LABELLED),
                events=list(self.events),
            )
