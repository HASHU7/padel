"""Raw-video perception: players + 17-keypoint poses, ball, main-camera check.

Models (trained/exported on Colab, see colab/):
  models/padel_pose.onnx  YOLO-pose export, output (1, 56, N): box(4) conf(1) kps(51)
  models/padel_ball.onnx  YOLO detect export (1 class), output (1, 5, N)
Without padel_ball.onnx a classical detector (yellow + motion inside the
court) is used. Without a pose model raw-video analysis is not possible
locally: run colab/padel_gpu_inference.ipynb and import the result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .court import CourtCalibration
from .tracks import RawDetections
from .video import probe


def _letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.5) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, int)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou)
    return np.array(idx).reshape(-1).astype(int)


class _OnnxYolo:
    def __init__(self, path: Path, imgsz: Optional[int] = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=ort.get_available_providers())
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        shp = inp.shape
        self.imgsz = imgsz or (int(shp[2]) if isinstance(shp[2], int) else 1280)

    def raw(self, img: np.ndarray):
        lb, r, (dx, dy) = _letterbox(img, self.imgsz)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.name: x})[0][0]   # (C, N)
        return out.T, r, dx, dy


class PoseDetector(_OnnxYolo):
    def __call__(self, img: np.ndarray, conf: float = 0.3):
        p, r, dx, dy = self.raw(img)
        p = p[p[:, 4] > conf]
        if len(p) == 0:
            return np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([(cx - w / 2 - dx) / r, (cy - h / 2 - dy) / r, w / r, h / r], 1)
        keep = _nms(boxes, p[:, 4], 0.5)
        kp = p[keep, 5:5 + 51].reshape(-1, 17, 3).copy()
        kp[..., 0] = (kp[..., 0] - dx) / r
        kp[..., 1] = (kp[..., 1] - dy) / r
        return kp.astype(np.float32), boxes[keep].astype(np.float32), p[keep, 4].astype(np.float32)


class OnnxBallDetector(_OnnxYolo):
    def __call__(self, img: np.ndarray, prev: Optional[np.ndarray] = None, conf: float = 0.15):
        p, r, dx, dy = self.raw(img)
        p = p[p[:, 4] > conf]
        if len(p) == 0:
            return None
        if prev is not None and np.isfinite(prev[0]):
            # prefer the candidate consistent with the previous position
            cand = np.stack([(p[:, 0] - dx) / r, (p[:, 1] - dy) / r], 1)
            d = np.hypot(*(cand - prev[:2]).T)
            score = p[:, 4] - 0.002 * np.minimum(d, 300)
        else:
            score = p[:, 4]
        i = int(np.argmax(score))
        return np.array([(p[i, 0] - dx) / r, (p[i, 1] - dy) / r, p[i, 4]], np.float32)


def _onnx_ball_candidates(det: "OnnxBallDetector", img: np.ndarray, k: int = 5, conf: float = 0.05) -> np.ndarray:
    p, r, dx, dy = det.raw(img)
    p = p[p[:, 4] > conf]
    if len(p) == 0:
        return np.zeros((0, 3), np.float32)
    p = p[np.argsort(-p[:, 4])[:k]]
    return np.column_stack([(p[:, 0] - dx) / r, (p[:, 1] - dy) / r, p[:, 4]]).astype(np.float32)


class ClassicBallDetector:
    """Small bright moving blob inside the court enclosure (no model needed).
    On broadcast footage the motion-blurred ball appears as a 3-6 px bright
    streak (often pale, not saturated yellow, over the blue court). Candidate =
    local brightness peak (top-hat) AND frame-difference motion; scored by
    contrast, yellowness, size and continuity with the previous position."""

    def __init__(self, calib: CourtCalibration, width: int, height: int, tophat_thr: int = 30,
                 motion_thr: int = 14):
        poly = calib.court_polygon_px(margin_m=0.5).astype(np.int32)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        top = poly[:, 1].min()  # ball flies up to ~6 m: extend upwards
        mask[max(0, top - int(0.25 * height)):top, max(0, poly[:, 0].min()):poly[:, 0].max()] = 255
        self.mask = cv2.resize(mask, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.tophat_thr, self.motion_thr = tophat_thr, motion_thr
        self.prev_gray = None

    def candidates(self, img: np.ndarray, people_boxes=None, k: int = 30) -> np.ndarray:
        """Top-k ball candidates (x, y, score) in full-resolution pixels."""
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((0, 3), np.float32)
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, self.kernel)
        cand = ((tophat > self.tophat_thr) & (diff > self.motion_thr) & (self.mask > 0)).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
        out = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if a < 2 or a > 60 or max(w, h) > 12:
                continue
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            roi = hsv[y0:y0 + h, x0:x0 + w].reshape(-1, 3)
            th = float(tophat[y0:y0 + h, x0:x0 + w].max())
            hue, sat = np.median(roi[:, 0]), np.median(roi[:, 1])
            yellow = 1.0 if (18 <= hue <= 48 and sat > 70) else (0.4 if sat < 110 else 0.0)
            cx, cy = cents[i] * 2.0
            sc = th / 60.0 + yellow + min(a, 12) / 12.0
            if people_boxes is not None:
                for (x, y, bw, bh) in people_boxes:
                    if x <= cx <= x + bw and y + 0.55 * bh <= cy <= y + bh:
                        sc -= 1.0
                    elif x + 0.15 * bw <= cx <= x + 0.85 * bw and y + 0.15 * bh <= cy <= y + 0.55 * bh:
                        sc -= 0.4
            out.append((cx, cy, sc))
        if not out:
            return np.zeros((0, 3), np.float32)
        out = np.array(out, np.float32)
        return out[np.argsort(-out[:, 2])[:k]]


def link_detector_candidates(cands: dict, n_frames: int, max_gap: int = 4, single_conf: float = 0.55,
                             min_conf: float = 0.0, static_px: float = 0.0, min_mean_conf: float = 0.0) -> np.ndarray:
    """Temporal linking for a trained ball detector. cands: frame -> (k,3)
    [x, y, conf] top-k detections (low threshold). Detections are chained into
    tracklets with a constant-velocity gate; a tracklet's score is its summed
    confidence boosted by length, so a run of consistent medium-confidence
    detections beats an isolated confident false positive (shoe, head, logo).
    The best non-overlapping tracklets form the ball track (N,3)."""
    active, done = [], []
    for f in range(n_frames):
        c = cands.get(f)
        if c is not None and min_conf > 0:
            c = c[c[:, 2] >= min_conf]
        used = set()
        if c is not None and len(c):
            for t in sorted(active, key=lambda t: -sum(t["s"])):
                gap = f - t["f"][-1]
                if gap > max_gap:
                    continue
                pred = t["p"][-1] + t["v"] * gap
                d = np.hypot(*(c[:, :2] - pred).T)
                gate = 30 + 0.5 * np.hypot(*t["v"]) * gap + 10 * gap
                for j in np.argsort(d):
                    if j in used or d[j] > gate:
                        continue
                    used.add(j)
                    nv = (c[j, :2] - t["p"][-1]) / gap
                    t["v"] = 0.5 * t["v"] + 0.5 * nv if len(t["f"]) > 1 else nv
                    t["f"].append(f); t["p"].append(c[j, :2].copy()); t["s"].append(float(c[j, 2]))
                    break
            for j in range(len(c)):
                if j not in used:
                    active.append({"f": [f], "p": [c[j, :2].copy()], "s": [float(c[j, 2])], "v": np.zeros(2)})
        still = []
        for t in active:
            (still if f - t["f"][-1] <= max_gap else done).append(t)
        active = still
    done += active
    scored = []
    for t in done:
        n = len(t["f"])
        if n < 3 and max(t["s"]) < single_conf:
            continue
        if min_mean_conf and float(np.mean(t["s"])) < min_mean_conf:
            continue
        if static_px and n >= 8:   # logos / lights / a ball lying still: not the ball in play
            P = np.array(t["p"])
            if np.hypot(*(P.max(0) - P.min(0))) < static_px:
                continue
        scored.append((float(np.sum(t["s"])) * (1.0 + 0.15 * min(n, 10)), t))
    scored.sort(key=lambda st: -st[0])
    ball = np.full((n_frames, 3), np.nan, np.float32)
    for _, t in scored:
        fr = np.array(t["f"])
        if np.isnan(ball[fr, 0]).mean() < 0.6:
            continue
        for f, p, s in zip(t["f"], t["p"], t["s"]):
            if np.isnan(ball[f, 0]):
                ball[f] = (p[0], p[1], s)
    return ball


def best_per_frame(cands: dict, n_frames: int, min_conf: float = 0.1) -> np.ndarray:
    """Highest-confidence detection per frame, mildly favouring continuity."""
    ball = np.full((n_frames, 3), np.nan, np.float32)
    prev, prev_f = None, -99
    for f in range(n_frames):
        c = cands.get(f)
        if c is None or not len(c):
            continue
        c = c[c[:, 2] >= min_conf]
        if not len(c):
            continue
        sc = c[:, 2].copy()
        if prev is not None and f - prev_f <= 5:
            sc -= 0.002 * np.minimum(np.hypot(*(c[:, :2] - prev).T), 300)
        j = int(np.argmax(sc))
        ball[f] = c[j]
        prev, prev_f = c[j, :2], f
    return ball


BALL_METHODS = {
    "best": lambda c, n: best_per_frame(c, n, 0.1),
    "best25": lambda c, n: best_per_frame(c, n, 0.25),
    "link_raw": lambda c, n: link_detector_candidates(c, n),
    "link_filtered": lambda c, n: link_detector_candidates(c, n, min_conf=0.1, static_px=25, min_mean_conf=0.15),
    "link_strict": lambda c, n: link_detector_candidates(c, n, min_conf=0.2, static_px=25, min_mean_conf=0.3),
}
DEFAULT_BALL_METHOD = "link_strict"   # chosen by measured F1 on unseen matches (0.878 vs 0.780)


def select_ball(cands: dict, n_frames: int, method: str = DEFAULT_BALL_METHOD) -> np.ndarray:
    """Turn per-frame ball candidates into one ball track (method chosen by
    measurement on labelled matches, see kaggle/make_cv_kernels.py)."""
    return BALL_METHODS[method](cands, n_frames)


def link_ball_candidates(cands: dict, n_frames: int, max_gap: int = 3) -> np.ndarray:
    """Offline track-before-detect: link per-frame candidates into tracklets
    with a constant-velocity gate, score them (length, contrast, real motion),
    then keep the best non-overlapping tracklets -> (N,3) ball track."""
    active, done = [], []
    for f in range(n_frames):
        c = cands.get(f)
        used = set()
        if c is not None and len(c):
            for t in sorted(active, key=lambda t: -len(t["f"])):
                gap = f - t["f"][-1]
                if gap > max_gap:
                    continue
                pred = t["p"][-1] + t["v"] * gap
                d = np.hypot(*(c[:, :2] - pred).T)
                gate = 22 + 0.35 * np.hypot(*t["v"]) * gap + 8 * gap
                for j in np.argsort(d):
                    if j in used or d[j] > gate:
                        continue
                    used.add(j)
                    newv = (c[j, :2] - t["p"][-1]) / gap
                    t["v"] = 0.5 * t["v"] + 0.5 * newv if len(t["f"]) > 1 else newv
                    t["f"].append(f); t["p"].append(c[j, :2].copy()); t["s"].append(float(c[j, 2]))
                    break
            for j in range(len(c)):
                if j not in used:
                    active.append({"f": [f], "p": [c[j, :2].copy()], "s": [float(c[j, 2])], "v": np.zeros(2)})
        still = []
        for t in active:
            (still if f - t["f"][-1] <= max_gap else done).append(t)
        active = still
    done += active
    scored = []
    for t in done:
        if len(t["f"]) < 4:
            continue
        P = np.array(t["p"])
        steps = np.hypot(*np.diff(P, axis=0).T)
        disp = float(np.hypot(*(P[-1] - P[0])))
        path = float(steps.sum())
        straight = disp / max(path, 1e-6)
        acc = np.hypot(*np.diff(P, 2, axis=0).T)
        jerk = float(np.median(acc)) if len(acc) else 0.0
        speed = float(np.median(steps))
        # a flying ball moves fast along a smooth, nearly straight 2D path; clutter
        # (crowd, rackets, LED boards) jitters
        if straight < 0.8 or speed < 3.0 or jerk > 12.0:
            continue
        score = float(np.sum(t["s"])) + 0.6 * len(t["f"]) + 3.0 * straight - 0.15 * jerk
        scored.append((score, t))
    scored.sort(key=lambda st: -st[0])
    ball = np.full((n_frames, 3), np.nan, np.float32)
    for score, t in scored:
        fr = np.array(t["f"])
        free = np.isnan(ball[fr, 0])
        if free.mean() < 0.6:
            continue
        for f, p, sc in zip(t["f"], t["p"], t["s"]):
            if np.isnan(ball[f, 0]):
                ball[f] = (p[0], p[1], min(1.0, sc / 3.0))
    return ball


class MainViewChecker:
    """True when the frame shows the calibrated main camera (court lines where
    the calibration predicts them). Replays / close-ups are skipped."""

    def __init__(self, calib: CourtCalibration, width: int, height: int):
        self.lines = np.zeros((height // 4, width // 4), np.uint8)
        for a, b in calib.court_lines_px():
            if np.isfinite(a).all() and np.isfinite(b).all():
                cv2.line(self.lines, tuple((a / 4).astype(int)), tuple((b / 4).astype(int)), 255, 1)
        self.n = max(int((self.lines > 0).sum()), 1)

    def __call__(self, img: np.ndarray) -> float:
        small = cv2.resize(img, (self.lines.shape[1], self.lines.shape[0]))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 2] > 170) & (hsv[..., 1] < 80)).astype(np.uint8) * 255
        white = cv2.dilate(white, np.ones((3, 3), np.uint8))
        return float(((white > 0) & (self.lines > 0)).sum()) / self.n


def models_status() -> dict:
    return {"pose": C.POSE_MODEL.exists(), "ball": C.BALL_MODEL.exists(), "shot": C.SHOT_MODEL.exists()}


def run_perception(video: Path, calib: CourtCalibration, progress: Optional[Callable[[float, str], None]] = None,
                   pose_stride: int = 1, start: int = 0, end: Optional[int] = None,
                   view_threshold: float = 0.07) -> RawDetections:
    info = probe(video)
    if not C.POSE_MODEL.exists():
        raise FileNotFoundError(
            "No pose model at models/padel_pose.onnx. Train/export it with colab/padel_train_models.ipynb, "
            "or run colab/padel_gpu_inference.ipynb on the video and import the detections file.")
    pose = PoseDetector(C.POSE_MODEL)
    ball_det = OnnxBallDetector(C.BALL_MODEL) if C.BALL_MODEL.exists() else None
    classic = ClassicBallDetector(calib, info.width, info.height)
    view = MainViewChecker(calib, info.width, info.height)
    end = info.n_frames if end is None else min(end, info.n_frames)
    raw = RawDetections(info.fps, info.n_frames, info.width, info.height,
                        source=f"video:{video.name}:pose{'+ballonnx' if ball_det else '+ballclassic'}",
                        frame_range=(start, end))
    cap = cv2.VideoCapture(str(video))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev_ball = None
    last_boxes = None
    cands = {}
    view_ok = True
    for f in range(start, end):
        ok, img = cap.read()
        if not ok:
            break
        if f % 15 == 0:
            view_ok = view(img) >= view_threshold
        if not view_ok:
            classic.prev_gray = None
            continue
        if (f - start) % pose_stride == 0:
            kp, boxes, sc = pose(img)
            raw.persons_kp[f], raw.persons_bbox[f], raw.persons_score[f] = kp, boxes, sc
            last_boxes = boxes
        if ball_det:
            cands[f] = _onnx_ball_candidates(ball_det, img)
        else:
            cands[f] = classic.candidates(img, last_boxes)
        if progress and (f - start) % 50 == 0:
            progress((f - start) / max(end - start, 1), "detecting players and ball")
    cap.release()
    if progress:
        progress(1.0, "linking ball trajectory")
    raw.ball = select_ball(cands, info.n_frames) if ball_det else link_ball_candidates(cands, info.n_frames)
    return raw


def evaluate_ball_detector(video: Path, calib: CourtCalibration, gt_ball: np.ndarray, n: int = 900, start: int = 0,
                           tol_px: float = 12.0, raw_people: Optional[RawDetections] = None) -> dict:
    """Precision/recall of the classical detector + linking vs annotated balls."""
    info = probe(video)
    det = ClassicBallDetector(calib, info.width, info.height)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cands = {}
    for f in range(start, start + n):
        ok, img = cap.read()
        if not ok:
            break
        boxes = raw_people.persons_bbox.get(f) if raw_people is not None else None
        cands[f - start] = det.candidates(img, boxes)
    cap.release()
    ball = link_ball_candidates(cands, n)
    tp = fp = fn = 0
    for i in range(n):
        g = gt_ball[start + i]
        b = ball[i]
        has_g, has_b = np.isfinite(g[0]), np.isfinite(b[0])
        if has_b and has_g and np.hypot(b[0] - g[0], b[1] - g[1]) <= tol_px:
            tp += 1
        else:
            fp += int(has_b)
            fn += int(has_g)
    return {"frames": n, "gt_frames": int(np.isfinite(gt_ball[start:start + n, 0]).sum()), "tp": tp, "fp": fp,
            "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
