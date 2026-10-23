"""Ball trajectory cleaning: outlier rejection + short-gap interpolation."""
from __future__ import annotations

import numpy as np

from . import config as C


def clean_ball(ball: np.ndarray, fps: float = 30.0, max_gap: int = C.BALL_MAX_GAP_INTERP) -> np.ndarray:
    """ball (N,3) [x, y, conf] with NaN for missing. Returns cleaned copy with
    column 2 = 1.0 detected, 0.5 interpolated, NaN missing."""
    b = np.array(ball, np.float64, copy=True)
    N = len(b)
    valid = np.isfinite(b[:, 0])
    idx = np.where(valid)[0]
    # 1) isolated detections (no support within +-4 frames at a plausible distance)
    keep = np.zeros(N, bool)
    for k, f in enumerate(idx):
        ok = False
        for j in (k - 2, k - 1, k + 1, k + 2):
            if 0 <= j < len(idx):
                g = abs(idx[j] - f)
                if 0 < g <= 4 and np.hypot(*(b[idx[j], :2] - b[f, :2])) < 70.0 * g:
                    ok = True
                    break
        keep[f] = ok
    # 2) spikes: a point far from the line through its neighbours
    idx = np.where(keep)[0]
    for k in range(1, len(idx) - 1):
        a, f, c = idx[k - 1], idx[k], idx[k + 1]
        if c - a <= 6:
            t = (f - a) / (c - a)
            pred = b[a, :2] + t * (b[c, :2] - b[a, :2])
            span = np.hypot(*(b[c, :2] - b[a, :2]))
            if np.hypot(*(b[f, :2] - pred)) > max(60.0, 1.5 * span):
                keep[f] = False
    b[~keep] = np.nan
    b[keep, 2] = 1.0
    # 3) interpolate short gaps
    idx = np.where(keep)[0]
    for a, c in zip(idx[:-1], idx[1:]):
        g = c - a
        if 1 < g <= max_gap and np.hypot(*(b[c, :2] - b[a, :2])) < 60.0 * g:
            for f in range(a + 1, c):
                t = (f - a) / g
                b[f, :2] = b[a, :2] + t * (b[c, :2] - b[a, :2])
                b[f, 2] = 0.5
    return b.astype(np.float32)


def ball_valid(b: np.ndarray) -> np.ndarray:
    return np.isfinite(b[:, 0])
