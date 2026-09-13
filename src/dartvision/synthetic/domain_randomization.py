"""Domain randomization applicata ai ritagli di training.

Un rendering sintetico "pulito" farebbe generalizzare male il modello a
una vera immagine di webcam: queste funzioni componibili simulano la
variabilita' reale (illuminazione, rumore del sensore, leggera sfocatura,
artefatti di compressione). Applicate al crop finale, non al piano
intero: piu' economico, e replica meglio cosa vede davvero il modello a
inferenza (un piccolo ritaglio, non l'intero bersaglio).
"""

from __future__ import annotations

import cv2
import numpy as np


def jitter_brightness_contrast(
    img: np.ndarray,
    rng: np.random.Generator,
    brightness_range: tuple[float, float] = (-30.0, 30.0),
    contrast_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    brightness = rng.uniform(*brightness_range)
    contrast = rng.uniform(*contrast_range)
    out = img.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter_hue(
    img: np.ndarray, rng: np.random.Generator, hue_shift_range: tuple[int, int] = (-8, 8)
) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(*hue_shift_range))) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_gaussian_noise(
    img: np.ndarray, rng: np.random.Generator, std_range: tuple[float, float] = (1.0, 6.0)
) -> np.ndarray:
    std = rng.uniform(*std_range)
    noise = rng.normal(0.0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_blur(
    img: np.ndarray, rng: np.random.Generator, kernel_choices: tuple[int, ...] = (1, 3, 5)
) -> np.ndarray:
    kernel = int(rng.choice(kernel_choices))
    if kernel <= 1:
        return img
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def simulate_jpeg_artifacts(
    img: np.ndarray, rng: np.random.Generator, quality_range: tuple[int, int] = (35, 90)
) -> np.ndarray:
    quality = int(rng.integers(*quality_range))
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def randomize(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applica una sequenza di randomizzazioni, alcune sempre, altre con
    probabilita' variabile, per simulare la variabilita' di una webcam
    reale senza rendere OGNI campione ugualmente degradato."""
    out = jitter_brightness_contrast(img, rng)
    if rng.random() < 0.5:
        out = jitter_hue(out, rng)
    out = add_gaussian_noise(out, rng)
    if rng.random() < 0.6:
        out = apply_blur(out, rng)
    if rng.random() < 0.5:
        out = simulate_jpeg_artifacts(out, rng)
    return out
