"""Player photos, stored as square JPEGs in ``<photos dir>/<slug>.jpg``."""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np

PHOTO_SIZE = 256
MAX_UPLOAD_BYTES = 8_000_000
_FILENAME = re.compile(r"[a-z0-9-]+\.jpg")


def player_slug(name: str) -> str:
    """Stable, filesystem-safe identifier of a player name (case insensitive)."""
    key = name.strip().lower()
    base = re.sub(r"[^a-z0-9]+", "-", key).strip("-")[:32] or "player"
    return f"{base}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


def decode_image(data: str | bytes) -> np.ndarray:
    """Decode a data URL, a base64 string or raw bytes into a BGR image."""
    if isinstance(data, str):
        payload = data.split(",", 1)[1] if data.startswith("data:") else data
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64 image") from exc
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("image too large")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not an image")
    return image


class PhotoStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def path(self, name: str) -> Path:
        return self.dir / f"{player_slug(name)}.jpg"

    def url(self, name: str) -> str | None:
        """URL of the player's photo (with a version query that changes on update), or None."""
        path = self.path(name)
        try:
            version = path.stat().st_mtime_ns
        except OSError:
            return None
        return f"/api/player-photo/{path.name}?v={version}"

    def save(self, name: str, data: str | bytes) -> None:
        """Centre-crop the image to a square, resize it and store it for `name`."""
        image = decode_image(data)
        h, w = image.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        square = cv2.resize(image[y0:y0 + side, x0:x0 + side], (PHOTO_SIZE, PHOTO_SIZE), interpolation=cv2.INTER_AREA)
        self.dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.path(name)), square, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"cannot write {self.path(name)}")

    def delete(self, name: str) -> None:
        self.path(name).unlink(missing_ok=True)

    def file(self, filename: str) -> Path | None:
        """Existing photo file for a request path, rejecting anything that is not a stored photo name."""
        if not _FILENAME.fullmatch(filename):
            return None
        path = self.dir / filename
        return path if path.is_file() else None
