"""Torch-free TipNet inference through ONNX Runtime: same math as ``model.py``, but with no
torch/CUDA dependency, for the ``--tip-backend onnx`` server option. Only imported when that
backend is selected, so a plain torch install is enough for everyone else.

Produce the ``.onnx`` checkpoint with ``lite/export_onnx.py`` (exports and validates against the
torch model; also writes a quantized copy, though plain fp32 turned out faster on CPU in testing).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..board.detector import BoardState
from .views import render_view

INPUT_SIZE = 512
STRIDE = 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class TipNetOnnx:
    """Drop-in substitute for a torch ``TipNet``: exposes ``predict_tips`` directly (unlike the
    torch model, which needs the free ``predict_tips`` function), so ``LiveScorer`` can call
    either the same way."""

    def __init__(self, session, view: str):
        self.session = session
        self.view = view

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        s = INPUT_SIZE / image_bgr.shape[1]
        x = cv2.warpAffine(image_bgr, np.float32([[s, 0, 0], [0, s, 0]]), (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def _decode(self, heat_logits: np.ndarray, offset: np.ndarray, threshold: float, max_tips: int):
        heat = 1.0 / (1.0 + np.exp(-heat_logits[0, 0].astype(np.float64)))  # sigmoid
        pooled = _max_pool3x3(heat)
        peaks = np.where(pooled == heat, heat, 0.0)
        height, width = peaks.shape
        order = np.argsort(peaks.reshape(-1))[::-1][:max_tips]
        tips = []
        for i in order:
            score = float(peaks.flat[i])
            if score < threshold:
                break
            y, x = divmod(int(i), width)
            ox, oy = offset[0, 0, y, x], offset[0, 1, y, x]
            tips.append(((x + ox) * STRIDE, (y + oy) * STRIDE, score))
        return tips

    def predict_tips(self, frame: np.ndarray, state: BoardState, view: str, threshold: float = 0.3,
                     max_tips: int = 6) -> np.ndarray:
        image, to_mm = render_view(frame, state, view)
        s = INPUT_SIZE / image.shape[1]
        input_to_mm = to_mm @ np.diag([1 / s, 1 / s, 1.0])
        heat, offset = self.session.run(None, {"image": self._preprocess(image)})
        tips = self._decode(heat, offset, threshold, max_tips)
        if not tips:
            return np.zeros((0, 3))
        q = np.array([(u, v, 1.0) for u, v, _ in tips]) @ input_to_mm.T
        return np.column_stack([q[:, :2] / q[:, 2:3], [c for _, _, c in tips]])


def _max_pool3x3(x: np.ndarray) -> np.ndarray:
    """3x3 max pool, stride 1, same padding (numpy, matches torch's max_pool2d(x, 3, 1, 1))."""
    padded = np.pad(x, 1, mode="constant", constant_values=-np.inf)
    out = np.full_like(x, -np.inf)
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, padded[dy:dy + x.shape[0], dx:dx + x.shape[1]])
    return out


def load_tipnet_onnx(path: str | Path) -> tuple[TipNetOnnx, str]:
    """Load an .onnx checkpoint written by ``lite/export_onnx.py``. Returns ``(model, view)``."""
    import onnxruntime as ort

    path = Path(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    view_file = path.with_name("view.txt")
    view = view_file.read_text().strip() if view_file.exists() else "rect"
    return TipNetOnnx(session, view), view
