"""Dartboard detection and tracking in a video stream.

Pipeline:

1. Initial search: a colour mask of the red/green double and treble rings and an ellipse fitted
   to the outer contour give an approximate affine homography.
2. Iterative refinement (ICP-like): the image is rectified with the current homography, ring
   edges are measured along radial rays and sector boundaries along circular arcs, and a full
   perspective homography is re-estimated by minimising point-to-curve distances
   (Levenberg-Marquardt with a Cauchy loss). Search windows shrink at every iteration.
3. Orientation: the red/green parity fixes the rotation up to 36°; the 20 is the red sector
   closest to a reference direction (image "up" by default, see ``set_sector20``).
4. Tracking: later frames start from the previous homography with fewer iterations and narrower
   windows; when the fit quality collapses, detection restarts from step 1.
5. Calibration: well-locked frames update a running estimate of the actual wire radii of this
   particular board, which are then used for fitting, drawing and scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import geometry as g

# (nominal radius, sign of the saturation gradient going outwards, maximum search window)
RADIAL_EDGES = (
    (g.R_BULL, -1, 5.0),
    (g.R_TREBLE_IN, +1, math.inf),
    (g.R_TREBLE_OUT, -1, math.inf),
    (g.R_DOUBLE_IN, +1, math.inf),
    (g.R_DOUBLE_OUT, -1, math.inf),
)

# Circular bands where sector boundaries are searched:
# "color" = red/green alternation, "bright" = black/cream alternation.
ANGULAR_BANDS = (
    ("color", 163.5, 168.5),
    ("color", 100.5, 105.5),
    ("bright", 118.0, 152.0),
    ("bright", 24.0, 92.0),
)
BAND_RADII = 5

ARC_SAMPLES = 1440
ARC_STEP = 360.0 / ARC_SAMPLES
RAY_STEP = 0.5  # mm
CANVAS_MM = 200.0  # half side of the rectified view


@dataclass
class DetectorConfig:
    process_width: int = 960  # wider frames are downscaled
    init_width: int = 640  # resolution of the initial search
    px_per_mm: float = 1.0  # resolution of the rectified view
    # HSV thresholds of the initial search (OpenCV hue range 0..179)
    red_hue: tuple[int, int] = (10, 165)  # red if H <= a or H >= b
    green_hue: tuple[int, int] = (35, 95)
    min_sat: int = 70
    min_val: int = 40
    min_ellipse_score: float = 0.7
    max_candidates: int = 3
    n_rays: int = 120
    radial_edge_thr: float = 0.10
    color_edge_thr: float = 0.12
    bright_edge_thr: float = 0.06
    min_parity: float = 0.05
    init_radial_windows: tuple[float, ...] = (22, 15, 10, 6, 4, 3)
    init_angular_windows: tuple[float, ...] = (7, 5, 4, 3, 2, 1.5)
    track_radial_windows: tuple[float, ...] = (8, 4)
    track_angular_windows: tuple[float, ...] = (4, 2)
    angular_weight: int = 3  # weight of sector-boundary observations in the fit
    inlier_mm: float = 2.0
    min_inlier_ratio: float = 0.40
    max_rms_mm: float = 1.5
    smoothing: float = 0.4  # 0 disables temporal smoothing
    smoothing_max_px: float = 6.0  # larger motions are not smoothed
    calibrate_rings: bool = True  # estimate the actual wire radii of this board
    calibration_rate: float = 0.05  # weight of each new frame in the running average
    calibration_max_dev_mm: float = 2.5  # maximum deviation from the nominal radii


@dataclass
class BoardState:
    H: np.ndarray  # image pixels -> model millimetres
    confidence: float  # fraction of the expected features found and consistent with the fit
    rms_mm: float
    n_inliers: int
    tracked: bool  # True when obtained by tracking, False from an initial search
    rings: tuple[float, ...] = g.RING_RADII  # wire radii (mm) used for drawing and scoring
    H_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.H_inv = np.linalg.inv(self.H)

    def to_model(self, pts) -> np.ndarray:
        return g.apply_homography(self.H, pts)

    def to_image(self, pts) -> np.ndarray:
        return g.apply_homography(self.H_inv, pts)

    def hit(self, image_pt) -> g.Hit:
        x, y = self.to_model([image_pt])[0]
        return g.score_model_point(x, y, self.rings)

    @property
    def center(self) -> np.ndarray:
        return self.to_image([(0.0, 0.0)])[0]

    def scaled(self, factor: float) -> BoardState:
        """The same board in an image resized by `factor` (e.g. a downscaled preview)."""
        return replace(self, H=self.H @ np.diag([1.0 / factor, 1.0 / factor, 1.0]))

    def rotated(self, sectors: int) -> BoardState:
        """The same board with the model rotated clockwise by `sectors` sectors (18° each)."""
        return replace(self, H=g.rotation(-18.0 * sectors) @ self.H)

    def aligned_to(self, reference: BoardState) -> BoardState:
        """Re-orient a fresh detection so that the 20 stays where it is in `reference`."""
        top = reference.to_image([(0.0, -130.0)])
        theta = g.polar(*self.to_model(top)[0])[1]
        return self.rotated(int(round(theta / 18.0)) % 20)


@dataclass
class _StepResult:
    H: np.ndarray
    errors: np.ndarray  # point-to-curve distance of every observation (mm)
    n_expected: int
    n_angular: int
    edge_obs: dict  # nominal wire radius -> observed radius (mm)


def _circ_smooth(x: np.ndarray) -> np.ndarray:
    return (np.roll(x, 2) + 4 * np.roll(x, 1) + 6 * x + 4 * np.roll(x, -1) + np.roll(x, -2)) / 16.0


def _curve_residuals(H, pts, is_ang, ref) -> np.ndarray:
    """Signed distance (mm) of image points mapped by H from their expected curve: a circle of
    radius `ref`, or a line through the origin at `ref` degrees."""
    m = g.apply_homography(H, pts)
    b = np.deg2rad(ref)
    return np.where(is_ang, m[:, 0] * np.cos(b) + m[:, 1] * np.sin(b), np.hypot(m[:, 0], m[:, 1]) - ref)


def _fit_point_to_curve(H0, pts, is_ang, ref, scale, angular_weight=3, iters=10) -> np.ndarray | None:
    """Homography minimising point-to-curve distances (Levenberg-Marquardt, Cauchy loss)."""
    c = pts.mean(axis=0)
    s = math.sqrt(2) / max(1e-9, np.mean(np.linalg.norm(pts - c, axis=1)))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    x, y = (pts - c).T * s
    Hn = H0 @ np.linalg.inv(T)
    h = (Hn / Hn[2, 2]).ravel()[:8].copy()
    b = np.deg2rad(ref)
    cb, sb = np.cos(b), np.sin(b)
    base_w = np.where(is_ang, float(angular_weight), 1.0)
    one, zero = np.ones_like(x), np.zeros_like(x)

    def evaluate(h):
        u = h[0] * x + h[1] * y + h[2]
        v = h[3] * x + h[4] * y + h[5]
        w = h[6] * x + h[7] * y + 1.0
        X, Y = u / w, v / w
        r = np.maximum(np.hypot(X, Y), 1e-9)
        res = np.where(is_ang, X * cb + Y * sb, r - ref)
        cost = np.sum(base_w * np.log1p((res / scale) ** 2))
        return res, cost, X, Y, w, r

    res, cost, X, Y, w, r = evaluate(h)
    lam = 1e-3
    for _ in range(iters):
        JX = np.column_stack([x, y, one, zero, zero, zero, -X * x, -X * y]) / w[:, None]
        JY = np.column_stack([zero, zero, zero, x, y, one, -Y * x, -Y * y]) / w[:, None]
        J = np.where(is_ang[:, None], cb[:, None] * JX + sb[:, None] * JY, (X[:, None] * JX + Y[:, None] * JY) / r[:, None])
        wr = base_w / (1.0 + (res / scale) ** 2)
        A = J.T @ (wr[:, None] * J)
        grad = J.T @ (wr * res)
        for _ in range(6):
            try:
                dh = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -grad)
            except np.linalg.LinAlgError:
                return None
            trial = evaluate(h + dh)
            if trial[1] < cost:
                h, (res, cost, X, Y, w, r) = h + dh, trial
                lam = max(lam / 10, 1e-7)
                break
            lam *= 10
        else:
            break
    if not np.all(np.isfinite(h)) or np.any(w <= 0) and np.any(w > 0):
        return None
    H = np.append(h, 1.0).reshape(3, 3) @ T
    return H / H[2, 2]


def _parabola(ym1, y0, yp1):
    """Sub-sample offset of the vertex of the parabola through three samples."""
    den = ym1 - 2 * y0 + yp1
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-12, 0.5 * (ym1 - yp1) / den, 0.0)
    return np.clip(off, -1.0, 1.0)


class DartboardDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self.state: BoardState | None = None
        self.reference_dir = np.array([0.0, -1.0])  # direction of the 20 in the image
        self.reset_calibration()
        self.debug: dict[str, np.ndarray] = {}
        self._build_sampling_maps()

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Restart from the initial search (the ring calibration is kept)."""
        self.state = None

    def reset_calibration(self) -> None:
        self.edge_radii = {R: R for R, _, _ in RADIAL_EDGES}
        self._n_calibrations = 0

    @property
    def rings(self) -> tuple[float, ...]:
        """Calibrated radii (mm) of the bullseye, bull, treble and double wires of this board."""
        e = self.edge_radii
        return (g.R_BULLSEYE, e[g.R_BULL], e[g.R_TREBLE_IN], e[g.R_TREBLE_OUT], e[g.R_DOUBLE_IN], e[g.R_DOUBLE_OUT])

    def process(self, frame: np.ndarray) -> BoardState | None:
        """Process a BGR frame and return the board state, or None if no board is visible."""
        cfg = self.cfg
        f = min(1.0, cfg.process_width / frame.shape[1])
        small = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else frame
        to_small = np.diag([f, f, 1.0])

        result, tracked = None, False
        if self.state is not None:
            H0 = self.state.H @ np.linalg.inv(to_small)
            result = self._refine(small, H0, cfg.track_radial_windows, cfg.track_angular_windows)
            tracked = result is not None

        if result is None:
            for H0 in self._initial_guesses(small):
                result = self._refine(small, H0, cfg.init_radial_windows, cfg.init_angular_windows)
                if result is not None:
                    result = (self._orient(result[0]), *result[1:])
                    break

        if result is None:
            self.state = None
            return None

        H, ratio, rms, n_inl, edge_obs = result
        if cfg.calibrate_rings and ratio >= 0.8:
            self._update_calibration(edge_obs)
        H = H @ to_small
        if tracked and cfg.smoothing > 0:
            H = self._smooth(self.state.H, H)
        self.state = BoardState(H / H[2, 2], ratio, rms, n_inl, tracked, self.rings)
        return self.state

    def set_sector20(self, image_pt) -> None:
        """Declare which sector is the 20 (a point clicked in the original image)."""
        if self.state is None:
            return
        x, y = self.state.to_model([image_pt])[0]
        _, theta = g.polar(x, y)
        j = int(round(theta / 36.0)) % 10  # closest red sector
        self.state = replace(self.state, H=g.rotation(-36.0 * j) @ self.state.H)
        d = np.asarray(image_pt, float) - self.state.center
        if np.linalg.norm(d) > 1e-6:
            self.reference_dir = d / np.linalg.norm(d)

    # ------------------------------------------------------------------ initial search

    def _initial_guesses(self, img: np.ndarray) -> list[np.ndarray]:
        cfg = self.cfg
        k = min(1.0, cfg.init_width / img.shape[1])
        work = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        hsv = cv2.cvtColor(cv2.GaussianBlur(work, (5, 5), 0), cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        sv = (sat >= cfg.min_sat) & (val >= cfg.min_val)
        red = sv & ((hue <= cfg.red_hue[0]) | (hue >= cfg.red_hue[1]))
        green = sv & (hue >= cfg.green_hue[0]) & (hue <= cfg.green_hue[1])
        mask = ((red | green) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ks = max(3, round(max(work.shape[:2]) / 90)) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        self.debug["mask"] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: cfg.max_candidates * 3]
        min_area = 1e-3 * work.shape[0] * work.shape[1]
        candidates = []
        for c in contours:
            if len(c) < 20 or cv2.contourArea(c) < min_area:
                continue
            (cx, cy), (ew, eh), ang = cv2.fitEllipse(c)
            if min(ew, eh) < 12 or max(ew, eh) / min(ew, eh) > 5:
                continue
            score = self._ellipse_score(mask, cx, cy, ew / 2, eh / 2, ang)
            if score >= cfg.min_ellipse_score:
                candidates.append((score, cx / k, cy / k, ew / 2 / k, eh / 2 / k, ang))
        candidates.sort(reverse=True)
        return [self._ellipse_to_homography(*c[1:]) for c in candidates[: cfg.max_candidates]]

    @staticmethod
    def _ellipse_score(mask, cx, cy, a, b, ang) -> float:
        """Reward ellipses whose double and treble rings are filled and whose gaps are empty."""
        t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        u = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        v = np.array([-u[1], u[0]])
        h, w = mask.shape

        def fill(scale):
            p = np.array([cx, cy]) + scale * (a * np.cos(t)[:, None] * u + b * np.sin(t)[:, None] * v)
            x, y = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            vals = np.zeros(len(t))
            vals[inside] = mask[y[inside], x[inside]] > 0
            return vals.mean()

        double = max(fill(s) for s in (0.965, 0.98))
        treble = max(fill(s) for s in (0.585, 0.605, 0.625))
        return double + treble - fill(0.80) - fill(1.12)

    @staticmethod
    def _ellipse_to_homography(cx, cy, a, b, ang) -> np.ndarray:
        u0, u1 = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        v0, v1 = -u1, u0
        sx, sy = g.R_DOUBLE_OUT / a, g.R_DOUBLE_OUT / b
        return np.array([
            [sx * u0, sx * u1, -sx * (u0 * cx + u1 * cy)],
            [sy * v0, sy * v1, -sy * (v0 * cx + v1 * cy)],
            [0.0, 0.0, 1.0],
        ])

    # ------------------------------------------------------------------ refinement

    def _build_sampling_maps(self) -> None:
        ppm = self.cfg.px_per_mm
        self._N = int(round(2 * CANVAS_MM * ppm))
        c = self._N / 2
        self._C = np.array([[ppm, 0, c], [0, ppm, c], [0, 0, 1.0]])

        self._ray_theta = (np.arange(self.cfg.n_rays) + 0.5) * 360.0 / self.cfg.n_rays
        self._ray_r = np.arange(0.0, CANVAS_MM - 2, RAY_STEP)
        th = np.deg2rad(self._ray_theta)[:, None]
        self._ray_mx = (c + ppm * self._ray_r[None, :] * np.sin(th)).astype(np.float32)
        self._ray_my = (c - ppm * self._ray_r[None, :] * np.cos(th)).astype(np.float32)

        th = np.deg2rad(np.arange(ARC_SAMPLES) * ARC_STEP)[None, :]
        radii = np.concatenate([np.linspace(r0, r1, BAND_RADII) for _, r0, r1 in ANGULAR_BANDS])[:, None]
        self._arc_mx = (c + ppm * radii * np.sin(th)).astype(np.float32)
        self._arc_my = (c - ppm * radii * np.cos(th)).astype(np.float32)

    @staticmethod
    def _features(warped: np.ndarray) -> np.ndarray:
        """Float channels: reliable saturation, red(+)/green(-) axis, brightness."""
        hsv = cv2.cvtColor(cv2.GaussianBlur(warped, (3, 3), 0), cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = np.deg2rad(hsv[..., 0] * 2.0)
        val = hsv[..., 2] / 255.0
        sat = hsv[..., 1] / 255.0 * np.clip((val - 0.12) / 0.18, 0.0, 1.0)
        color = sat * (np.cos(hue) - np.cos(hue - 2 * np.pi / 3)) / 1.5
        return cv2.merge([sat, color.astype(np.float32), val])

    def _refine(self, img, H, radial_windows, angular_windows):
        step = None
        for rw, aw in zip(radial_windows, angular_windows):
            step = self._step(img, H, rw, aw)
            if step is None:
                return None
            H = step.H
        if step.n_angular < 8:
            return None  # rotation not constrained by sector boundaries
        inl = step.errors < self.cfg.inlier_mm
        ratio = inl.sum() / step.n_expected
        rms = float(np.sqrt(np.mean(step.errors[inl] ** 2))) if inl.any() else math.inf
        if ratio < self.cfg.min_inlier_ratio or rms > self.cfg.max_rms_mm:
            return None
        if not self._plausible(H, img.shape):
            return None
        return H, float(min(1.0, ratio)), rms, int(inl.sum()), step.edge_obs

    def _step(self, img, H, rad_win, ang_win) -> _StepResult | None:
        cfg = self.cfg
        warped = cv2.warpPerspective(img, self._C @ H, (self._N, self._N), flags=cv2.INTER_LINEAR)
        feat = self._features(warped)
        H_inv = np.linalg.inv(H)

        # --- angular signals along the circular bands
        arcs = cv2.remap(feat, self._arc_mx, self._arc_my, cv2.INTER_LINEAR)
        signals, derivs = [], []
        fold = np.zeros(ARC_SAMPLES // 20)
        for b, (kind, _, _) in enumerate(ANGULAR_BANDS):
            ch = 1 if kind == "color" else 2
            s = _circ_smooth(arcs[b * BAND_RADII:(b + 1) * BAND_RADII, :, ch].mean(axis=0).astype(np.float64))
            d = np.roll(s, -2) - np.roll(s, 2)
            signals.append(s)
            derivs.append(np.abs(d))
            if derivs[-1].max() > 1e-6:
                fold += (derivs[-1] / derivs[-1].max()).reshape(20, -1).sum(axis=0)

        # residual rotation: sector boundaries must fall at 9 + 18k degrees
        fold = _circ_smooth(fold)
        p = int(np.argmax(fold))
        phi = (p + float(_parabola(fold[p - 1], fold[p], fold[(p + 1) % len(fold)]))) * ARC_STEP
        delta = (9.0 - phi + 9.0) % 18.0 - 9.0

        # parity: even sectors (20, 18, 13, ...) have red doubles and trebles
        color_sig = (signals[0] + signals[1]) / 2
        idx = np.round((np.arange(20) * 18.0 - delta) / ARC_STEP).astype(int) % ARC_SAMPLES
        parity = color_sig[idx[0::2]].mean() - color_sig[idx[1::2]].mean()
        # With a coarse initial guess the bands can miss the coloured rings:
        # in that case this step only uses the radial edges.
        use_angular = abs(parity) >= cfg.min_parity
        if parity < 0:
            delta += -18.0 if delta > 0 else 18.0

        obs, err_kind, err_ref, edge_nom = [], [], [], []

        # --- sector boundaries
        beta = 9.0 + 18.0 * np.arange(20)
        w = max(1, int(round(ang_win / ARC_STEP)))
        offs = np.arange(-w, w + 1)
        rows = np.arange(20)
        for b, (kind, r0, r1) in enumerate(ANGULAR_BANDS if use_angular else ()):
            i0 = np.round((beta - delta) / ARC_STEP).astype(int)
            seg = derivs[b][(i0[:, None] + offs[None, :]) % ARC_SAMPLES]
            j = seg.argmax(axis=1)
            val = seg[rows, j]
            thr = cfg.color_edge_thr if kind == "color" else cfg.bright_edge_thr
            ok = (val > max(thr, 0.3 * np.median(val))) & (j > 0) & (j < 2 * w)
            jm, jp = np.clip(j - 1, 0, 2 * w), np.clip(j + 1, 0, 2 * w)
            alpha = (i0 - w + j + _parabola(seg[rows, jm], val, seg[rows, jp])) * ARC_STEP
            rm = (r0 + r1) / 2
            obs.append(g.model_points(rm, alpha[ok]))
            err_kind.append(np.ones(ok.sum()))
            edge_nom.append(np.full(ok.sum(), np.nan))
            err_ref.append(beta[ok])

        # --- radial ring edges
        prof = cv2.remap(np.ascontiguousarray(feat[..., 0]), self._ray_mx, self._ray_my, cv2.INTER_LINEAR)
        prof = cv2.GaussianBlur(prof, (5, 1), 0)
        grad = np.zeros_like(prof)
        grad[:, 2:-2] = prof[:, 4:] - prof[:, :-4]
        rows = np.arange(cfg.n_rays)
        edges = {}
        for R, pol, max_win in RADIAL_EDGES:
            win = min(rad_win, max_win)
            lo = max(2, int(round((R - win) / RAY_STEP)))
            hi = min(len(self._ray_r) - 2, int(round((R + win) / RAY_STEP)) + 1)
            seg = pol * grad[:, lo:hi]
            j = seg.argmax(axis=1)
            val = seg[rows, j]
            ok = (val > cfg.radial_edge_thr) & (j > 0) & (j < hi - lo - 1)
            jm, jp = np.clip(j - 1, 0, hi - lo - 1), np.clip(j + 1, 0, hi - lo - 1)
            r_obs = (lo + j + _parabola(seg[rows, jm], val, seg[rows, jp])) * RAY_STEP
            edges[R] = (r_obs, ok)

        # Expected radii: the ones calibrated on this board (see _update_calibration), corrected
        # for the apparent thickness of each ring measured in this frame (blur, coloured sisal
        # showing under the wire).
        cal = self.edge_radii
        targets = {g.R_BULL: cal[g.R_BULL]}
        for r_in, r_out in ((g.R_TREBLE_IN, g.R_TREBLE_OUT), (g.R_DOUBLE_IN, g.R_DOUBLE_OUT)):
            (a, ok_a), (b, ok_b) = edges[r_in], edges[r_out]
            both = ok_a & ok_b
            pad = 0.0
            if both.sum() >= 10:
                pad = float(np.clip((np.median(b[both] - a[both]) - (cal[r_out] - cal[r_in])) / 2, -1.5, 2.5))
            targets[r_in], targets[r_out] = cal[r_in] - pad, cal[r_out] + pad
        for R, (r_obs, ok) in edges.items():
            obs.append(g.model_points(r_obs[ok], self._ray_theta[ok]))
            err_kind.append(np.zeros(ok.sum()))
            err_ref.append(np.full(ok.sum(), targets[R]))
            edge_nom.append(np.full(ok.sum(), R))

        obs = np.concatenate(obs)
        is_ang, ref = np.concatenate(err_kind) == 1, np.concatenate(err_ref)
        edge_nom = np.concatenate(edge_nom)
        if len(obs) < 40:
            return None

        img_pts = g.apply_homography(H_inv, obs)
        H_new = _fit_point_to_curve(H, img_pts, is_ang, ref, scale=max(1.0, 0.25 * rad_win),
                                    angular_weight=cfg.angular_weight)
        if H_new is None:
            return None
        signed = _curve_residuals(H_new, img_pts, is_ang, ref)
        # observed radius of every wire under the new homography, used for calibration
        edge_obs = {R: float(np.median(ref[edge_nom == R] + signed[edge_nom == R]))
                    for R in edges if np.sum(edge_nom == R) >= 20}
        n_expected = cfg.n_rays * len(RADIAL_EDGES) + 20 * len(ANGULAR_BANDS)
        return _StepResult(H_new, np.abs(signed), n_expected, int(is_ang.sum()), edge_obs)

    @staticmethod
    def _plausible(H: np.ndarray, shape) -> bool:
        H_inv = np.linalg.inv(H)
        ring = np.hstack([g.model_points(g.R_BOARD, np.arange(0, 360, 15)), np.ones((24, 1))]) @ H_inv.T
        center = H_inv @ np.array([0.0, 0.0, 1.0])
        if not (np.all(ring[:, 2] * center[2] > 0)):
            return False  # the board would cross the camera horizon
        pts = ring[:, :2] / ring[:, 2:3]
        c = center[:2] / center[2]
        radius = np.linalg.norm(pts - c, axis=1)
        h, w = shape[:2]
        return radius.min() > 10 and -w < c[0] < 2 * w and -h < c[1] < 2 * h

    # ------------------------------------------------------------------ calibration, orientation, smoothing

    def _update_calibration(self, observed: dict) -> None:
        """Running average of the actual wire radii. The centre of the double ring stays at 166 mm
        (scale reference); the other radii may deviate from the nominal values."""
        if g.R_DOUBLE_IN not in observed or g.R_DOUBLE_OUT not in observed:
            return
        k = (g.R_DOUBLE_IN + g.R_DOUBLE_OUT) / (observed[g.R_DOUBLE_IN] + observed[g.R_DOUBLE_OUT])
        self._n_calibrations += 1
        a = max(self.cfg.calibration_rate, 1.0 / self._n_calibrations)
        dev = self.cfg.calibration_max_dev_mm
        for R, r in observed.items():
            new = self.edge_radii[R] + a * (k * r - self.edge_radii[R])
            self.edge_radii[R] = float(np.clip(new, R - dev, R + dev))

    def _orient(self, H: np.ndarray) -> np.ndarray:
        H_inv = np.linalg.inv(H)
        c = g.apply_homography(H_inv, [(0.0, 0.0)])[0]
        tips = g.apply_homography(H_inv, g.model_points(130.0, np.arange(10) * 36.0)) - c
        tips /= np.linalg.norm(tips, axis=1, keepdims=True)
        j = int(np.argmax(tips @ self.reference_dir))
        return g.rotation(-36.0 * j) @ H

    def _smooth(self, H_old: np.ndarray, H_new: np.ndarray) -> np.ndarray:
        ref = g.model_points(140.0, np.arange(8) * 45.0)
        p_old = g.apply_homography(np.linalg.inv(H_old), ref)
        p_new = g.apply_homography(np.linalg.inv(H_new), ref)
        if np.abs(p_new - p_old).max() > self.cfg.smoothing_max_px:
            return H_new
        a = self.cfg.smoothing
        H_s, _ = cv2.findHomography((a * p_old + (1 - a) * p_new).astype(np.float32), ref.astype(np.float32), 0)
        return H_new if H_s is None else H_s
