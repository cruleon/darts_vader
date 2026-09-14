"""Labelled turn photos: ``<labels dir>/<session>/labels.json`` plus one JPEG per photo.

Schema::

    {
      "camera": {...},                        # free-form metadata (source, game, model view)
      "photos": [
        {"index": 1, "image": "photo_0001.jpg", "time": "21:04:12",
         "board_H": [[...], [...], [...]],    # image pixels -> board mm
         "rings": [6.35, 15.9, ...],          # calibrated wire radii (mm)
         "tips": [{"tip_mm": [x, y], "tip_image": [u, v], "score": "T20",
                   "origin": "click", "confidence": 1.0}],
         "status": "labelled"}                # labelled | no_tips | pending | discarded
      ]
    }

Sessions written by earlier versions of the tools are converted when loaded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .board import geometry as g
from .board.detector import BoardState

LABELLED, NO_TIPS, PENDING, DISCARDED = "labelled", "no_tips", "pending", "discarded"

_LEGACY_STATUS = {"etichettata": LABELLED, "nessuna punta": NO_TIPS, "da etichettare": PENDING, "scartata": DISCARDED}
_LEGACY_ORIGIN = {"modello": "model", "corretto": "corrected", "proposta": "proposal", "simulato": "approx"}


def tip_record(tip_mm, board: BoardState, score: str, origin: str, confidence: float | None = None) -> dict:
    record = dict(tip_mm=[round(float(v), 2) for v in tip_mm],
                  tip_image=[round(float(v), 1) for v in board.to_image([tip_mm])[0]],
                  score=score, origin=origin)
    if confidence is not None:
        record["confidence"] = round(float(confidence), 3)
    return record


def _normalise(entry: dict) -> dict:
    if "tips" not in entry:
        entry["tips"] = entry.pop("tips_present", []) + entry.pop("new_tips", [])
    for tip in entry["tips"]:
        tip["origin"] = _LEGACY_ORIGIN.get(tip.get("origin"), tip.get("origin", "click"))
    entry["status"] = _LEGACY_STATUS.get(entry.get("status"), entry.get("status", PENDING))
    for key in ("turn", "board_refit"):
        entry.pop(key, None)
    return entry


class LabelSession:
    """A labelling session folder. Nothing is written until the first photo is added."""

    def __init__(self, directory: Path, metadata: dict | None = None):
        self.dir = Path(directory)
        self.path = self.dir / "labels.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data["photos"] = [_normalise(p) for p in self.data.get("photos", [])]
        else:
            self.data = {"camera": dict(metadata or {}), "photos": []}

    @property
    def photos(self) -> list[dict]:
        return self.data["photos"]

    @property
    def metadata(self) -> dict:
        return self.data["camera"]

    def count(self, status: str = LABELLED) -> int:
        return sum(p["status"] == status for p in self.photos)

    def add_photo(self, frame: np.ndarray, board: BoardState, tips: list[dict] | None = None,
                  status: str = PENDING) -> dict:
        """Save the photo and append its entry to the session."""
        self.dir.mkdir(parents=True, exist_ok=True)
        index = max((p["index"] for p in self.photos), default=0) + 1
        name = f"photo_{index:04d}.jpg"
        cv2.imwrite(str(self.dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        entry = dict(index=index, image=name, time=time.strftime("%H:%M:%S"),
                     board_H=np.round(board.H, 9).tolist(), rings=[round(float(r), 3) for r in board.rings],
                     tips=list(tips or []), status=status)
        self.photos.append(entry)
        self.save()
        return entry

    def discard(self, entry: dict) -> None:
        entry["tips"], entry["status"] = [], DISCARDED
        (self.dir / entry["image"]).unlink(missing_ok=True)
        self.save()

    @staticmethod
    def board(entry: dict) -> BoardState:
        """The board fitted on the photo of `entry`."""
        return BoardState(np.array(entry["board_H"], float), 1.0, 0.0, 0, False, tuple(entry.get("rings", g.RING_RADII)))

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
