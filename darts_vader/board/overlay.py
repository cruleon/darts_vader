"""OpenCV drawing helpers: detected sectors on top of the video and readable text panels."""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as g
from .detector import BoardState

RING_COLOR = (255, 220, 0)
WIRE_COLOR = (0, 220, 255)
TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
NUMBER_RADIUS = g.R_BOARD + 22.0  # mm: sector numbers are drawn just outside the surround


def _poly(pts: np.ndarray) -> np.ndarray:
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def put_text(img, text, org, scale=0.6, color=TEXT_COLOR, thickness=1, center=False) -> None:
    """Outlined text, readable on any background."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    if center:
        x, y = x - tw // 2, y + th // 2
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def label_box(img, text, center, scale=0.5, color=TEXT_COLOR, thickness=1, alpha=0.65) -> None:
    """Text centred on a small translucent dark box."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(center[0]) - tw // 2, int(center[1]) + th // 2
    x0, y0, x1, y1 = x - 3, y - th - 3, x + tw + 3, y + base + 1
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    img[cy0:cy1, cx0:cx1] = (img[cy0:cy1, cx0:cx1] * (1 - alpha)).astype(np.uint8)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img: np.ndarray, lines, x: int, y: int, pad: int = 8, gap: int = 7, alpha: float = 0.6) -> int:
    """Lines of text stacked on a translucent dark panel, each placed below the previous one
    according to its real height (no overlaps). ``lines = [(text, scale, color, thickness)]``.
    Returns the y coordinate of the bottom edge of the panel."""
    sizes = [cv2.getTextSize(t, FONT, s, th) for t, s, _, th in lines]
    w = max(tw for (tw, _), _ in sizes) + 2 * pad
    h = sum(th + base for (_, th), base in sizes) + gap * (len(lines) - 1) + 2 * pad
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)
    cy = y + pad
    for (text, scale, color, thick), ((_, th), base) in zip(lines, sizes):
        cy += th
        cv2.putText(img, text, (x + pad, cy), FONT, scale, color, thick, cv2.LINE_AA)
        cy += base + gap
    return y1


def segment_polygon(r_in, r_out, a0, a1, step=2.0) -> list[np.ndarray]:
    """Outlines (model coordinates) of a board segment; a full ring has two outlines."""
    angles = np.arange(a0, a1 + 1e-6, step)
    if a1 - a0 >= 360.0:
        rings = [g.model_points(r_out, angles)]
        if r_in > 0:
            rings.append(g.model_points(r_in, angles))
        return rings
    return [np.vstack([g.model_points(r_out, angles), g.model_points(r_in, angles[::-1])])]


def draw_board(img: np.ndarray, state: BoardState, hover=None) -> g.Hit | None:
    """Draw rings, sector boundaries and numbers; highlight the segment under `hover`."""
    to_img = state.to_image
    scale = max(1, round(img.shape[1] / 900))

    hit = None
    if hover is not None:
        x, y = state.to_model([hover])[0]
        hit = g.score_model_point(x, y, state.rings)
        region = g.hit_region(x, y, state.rings)
        if region is not None:
            layer = img.copy()
            cv2.fillPoly(layer, [_poly(to_img(c)) for c in segment_polygon(*region)], HIGHLIGHT_COLOR, cv2.LINE_AA)
            cv2.addWeighted(layer, 0.45, img, 0.55, 0, dst=img)

    for r in state.rings:
        cv2.polylines(img, [_poly(to_img(g.model_points(r, np.arange(0, 360, 3))))], True, RING_COLOR, scale, cv2.LINE_AA)
    for k in range(20):
        a, b = to_img(g.model_points([state.rings[1], state.rings[5]], 9.0 + 18.0 * k))
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), WIRE_COLOR, scale, cv2.LINE_AA)
    for k, number in enumerate(g.SECTOR_ORDER):
        # outside the surround, so the labels never cover the numbers printed on the board
        p = to_img(g.model_points(NUMBER_RADIUS, 18.0 * k))[0]
        color = (80, 80, 255) if number == 20 else TEXT_COLOR
        label_box(img, str(number), p, 0.45 * scale, color, scale)

    if hit is not None:
        put_text(img, f"{hit.label} ({hit.score})", (hover[0] + 12, hover[1] - 12), 0.7 * scale, (0, 255, 255), scale + 1)
    return hit


def rectified_view(frame: np.ndarray, state: BoardState, size: int = 520) -> np.ndarray:
    """Fronto-parallel view of the board with the fitted geometry on top."""
    ppm = size / (2 * g.R_BOARD)
    C = np.array([[ppm, 0, size / 2], [0, ppm, size / 2], [0, 0, 1.0]])
    view = cv2.warpPerspective(frame, C @ state.H, (size, size), flags=cv2.INTER_LINEAR)
    center = (size // 2, size // 2)
    for r in state.rings:
        cv2.circle(view, center, int(round(r * ppm)), RING_COLOR, 1, cv2.LINE_AA)
    for k in range(20):
        a, b = g.model_points([state.rings[1], state.rings[5]], 9.0 + 18.0 * k) * ppm + size / 2
        cv2.line(view, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), WIRE_COLOR, 1, cv2.LINE_AA)
    return view
