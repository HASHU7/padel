"""Accuracy of raw-video detections (Kaggle/Colab models) vs PadelTracker100 labels.

Ball: precision / recall within `tol_px` on frames inside the annotated range.
Players: recall of annotated players (IoU >= 0.5), precision of on-court
detections, and PCK@0.1*bbox-height of the 17 keypoints for matched players.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .tracks import RawDetections


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-6)


def evaluate_detections(det: RawDetections, gt: RawDetections, tol_px: float = 10.0, step: int = 5) -> Dict:
    n = min(det.n_frames, gt.n_frames)
    gb = gt.ball[:n]
    last_gt_ball = int(np.where(np.isfinite(gb[:, 0]))[0].max()) if np.isfinite(gb[:, 0]).any() else n
    tp = fp = fn = 0
    for f in range(0, last_gt_ball + 1):
        g, d = gb[f], det.ball[f]
        hg, hd = np.isfinite(g[0]), np.isfinite(d[0])
        if hg and hd and np.hypot(d[0] - g[0], d[1] - g[1]) <= tol_px:
            tp += 1
        else:
            fp += int(hd)
            fn += int(hg)
    ball = {"frames_evaluated": last_gt_ball + 1, "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3), "tolerance_px": tol_px}

    p_tp = p_gt = p_det = 0
    pck_hits = pck_n = 0
    for f in range(0, n, step):
        gk, gbx = gt.persons_kp.get(f), gt.persons_bbox.get(f)
        dk, dbx = det.persons_kp.get(f), det.persons_bbox.get(f)
        if gbx is None:
            continue
        p_gt += len(gbx)
        if dbx is None or len(dbx) == 0:
            continue
        used = set()
        for gi, g in enumerate(gbx):
            best, bj = 0.0, None
            for j, d in enumerate(dbx):
                if j in used:
                    continue
                v = _iou(g, d)
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= 0.5:
                used.add(bj)
                p_tp += 1
                vis = gk[gi][:, 2] > 0
                err = np.hypot(*(dk[bj][:, :2] - gk[gi][:, :2]).T)
                thr = 0.1 * g[3]
                pck_hits += int((err[vis] <= thr).sum())
                pck_n += int(vis.sum())
        # detections overlapping no annotated player count against precision only
        # when they are player-sized and inside the annotated players' area
        p_det += len(used) + sum(1 for j, d in enumerate(dbx) if j not in used and d[3] > 0.6 * np.median(gbx[:, 3]))
    players = {"recall": round(p_tp / max(p_gt, 1), 3), "precision": round(p_tp / max(p_det, 1), 3),
               "keypoint_pck@0.1": round(pck_hits / max(pck_n, 1), 3)}
    return {"ball": ball, "players": players}
