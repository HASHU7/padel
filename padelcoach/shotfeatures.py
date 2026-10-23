"""Shot sequence features shared by the Colab-trained shot model and inference.

Per frame in a window centred on the contact frame:
  17 keypoints (x, y, conf) of the hitter, centred on the hip midpoint and
  scaled by torso length; far-side players are mirrored horizontally (so the
  player's right is always +x) and left-handed players mirrored again (so the
  dominant arm is always the right arm). Plus ball offset (x, y, valid) and a
  far-side flag. Shape: (T, 55). Pure numpy -> usable on Colab and locally.
"""
from __future__ import annotations

import numpy as np

WINDOW = 31          # frames (~1 s at 30 fps)
N_FEAT = 17 * 3 + 3 + 1

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
# keypoint index permutation that swaps left/right (used when mirroring)
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _norm_frame(kp: np.ndarray, ball: np.ndarray, mirror: bool, permute: bool, far: bool) -> np.ndarray:
    out = np.zeros(N_FEAT, np.float32)
    if not np.isfinite(kp[:, 0]).any():
        return out
    k = kp.copy()
    vis = (k[:, 2] > 0.2) & np.isfinite(k[:, 0])
    hips = [i for i in (L_HIP, R_HIP) if vis[i]]
    shs = [i for i in (L_SH, R_SH) if vis[i]]
    if not hips or not shs:
        return out
    hip = k[hips, :2].mean(0)
    sh = k[shs, :2].mean(0)
    torso = max(np.linalg.norm(sh - hip), 8.0)
    xy = (k[:, :2] - hip) / torso
    bxy = np.zeros(2)
    bvalid = 0.0
    if ball is not None and np.isfinite(ball[0]):
        bxy = (ball[:2] - hip) / torso
        bvalid = 1.0
    if mirror:
        xy[:, 0] = -xy[:, 0]
        bxy[0] = -bxy[0]
    if permute:  # left-handed: relabel so the dominant arm is always "right"
        xy = xy[FLIP_PERM]
        vis = vis[FLIP_PERM]
    xy[~vis] = 0.0
    out[:51] = np.column_stack([np.clip(xy, -6, 6), vis.astype(np.float32)]).reshape(-1)
    out[51:53] = np.clip(bxy, -8, 8)
    out[53] = bvalid
    out[54] = 1.0 if far else 0.0
    return out


def sequence_features(kp_seq: np.ndarray, ball_seq: np.ndarray, far: bool, left_handed: bool) -> np.ndarray:
    """kp_seq (T,17,3) image px (NaN missing), ball_seq (T,3) -> (T, N_FEAT)."""
    # Far players face the camera: mirror x so the player's right is +x (keypoint
    # labels are already from the player's own perspective). Left-handers are
    # mirrored + relabelled so the dominant arm is canonical "right".
    mirror = bool(far) ^ bool(left_handed)
    return np.stack([_norm_frame(kp_seq[t], ball_seq[t], mirror, bool(left_handed), far)
                     for t in range(len(kp_seq))])


def window_indices(center: int, n_frames: int, window: int = WINDOW) -> np.ndarray:
    half = window // 2
    return np.clip(np.arange(center - half, center + half + 1), 0, n_frames - 1)
