"""Core data containers shared by every pipeline stage.

RawDetections: per-frame unordered person detections + ball candidates
               (output of perception or of a ground-truth loader).
Tracks:        identity-resolved arrays (4 player slots) used by analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config as C

N_SLOTS = 4


@dataclass
class RawDetections:
    fps: float
    n_frames: int
    width: int
    height: int
    # frame -> (k,17,3) keypoints [x, y, conf] and (k,4) bbox [x, y, w, h], (k,) score
    persons_kp: Dict[int, np.ndarray] = field(default_factory=dict)
    persons_bbox: Dict[int, np.ndarray] = field(default_factory=dict)
    persons_score: Dict[int, np.ndarray] = field(default_factory=dict)
    ball: np.ndarray = None   # (N,3) x, y, conf ; NaN where missing
    source: str = "unknown"
    frame_range: tuple = None  # (start, end) frames covered

    def __post_init__(self):
        if self.ball is None:
            self.ball = np.full((self.n_frames, 3), np.nan, np.float32)
        if self.frame_range is None:
            self.frame_range = (0, self.n_frames)

    def crop(self, n: int) -> "RawDetections":
        r = RawDetections(self.fps, n, self.width, self.height, ball=self.ball[:n].copy(), source=self.source,
                          frame_range=(0, n))
        for f in self.persons_kp:
            if f < n:
                r.persons_kp[f], r.persons_bbox[f], r.persons_score[f] =                     self.persons_kp[f], self.persons_bbox[f], self.persons_score[f]
        return r

    def save(self, path: Path) -> None:
        frames = np.array(sorted(self.persons_kp), dtype=np.int32)
        counts = np.array([len(self.persons_kp[f]) for f in frames], dtype=np.int16)
        kp = np.concatenate([self.persons_kp[f] for f in frames]) if len(frames) else np.zeros((0, 17, 3), np.float32)
        bb = np.concatenate([self.persons_bbox[f] for f in frames]) if len(frames) else np.zeros((0, 4), np.float32)
        sc = np.concatenate([self.persons_score[f] for f in frames]) if len(frames) else np.zeros((0,), np.float32)
        np.savez_compressed(path, fps=self.fps, n_frames=self.n_frames, width=self.width, height=self.height,
                            frames=frames, counts=counts, kp=kp.astype(np.float32), bbox=bb.astype(np.float32),
                            score=sc.astype(np.float32), ball=self.ball.astype(np.float32),
                            source=self.source, frame_range=np.array(self.frame_range))

    @classmethod
    def load(cls, path: Path) -> "RawDetections":
        z = np.load(path, allow_pickle=False)
        r = cls(float(z["fps"]), int(z["n_frames"]), int(z["width"]), int(z["height"]),
                ball=z["ball"], source=str(z["source"]), frame_range=tuple(int(v) for v in z["frame_range"]))
        kp, bb, sc = z["kp"], z["bbox"], z["score"]  # decompress once
        off = 0
        for f, c in zip(z["frames"], z["counts"]):
            c = int(c)
            r.persons_kp[int(f)] = kp[off:off + c]
            r.persons_bbox[int(f)] = bb[off:off + c]
            r.persons_score[int(f)] = sc[off:off + c]
            off += c
        return r


@dataclass
class Tracks:
    fps: float
    n_frames: int
    width: int
    height: int
    kp: np.ndarray        # (N,4,17,3)
    bbox: np.ndarray      # (N,4,4)
    feet: np.ndarray      # (N,4,2) court metres (floor contact point)
    ball: np.ndarray      # (N,3) image px + conf (cleaned/interpolated)
    team_of_slot: List[str] = field(default_factory=lambda: ["A", "A", "B", "B"])
    slot_meta: List[dict] = field(default_factory=list)
    source: str = "unknown"
    frame_range: tuple = (0, 0)
    info: dict = field(default_factory=dict)

    @property
    def present(self) -> np.ndarray:
        return np.isfinite(self.feet[..., 0])

    def side_of_slot(self, f: int, s: int) -> Optional[str]:
        y = self.feet[f, s, 1]
        if not np.isfinite(y):
            return None
        return "near" if y >= C.NET_Y else "far"

    def team_side(self, f: int, team: str) -> Optional[str]:
        """Which half ('near'/'far') a team occupies at frame f (median of partners)."""
        ys = [self.feet[f, s, 1] for s in range(N_SLOTS) if self.team_of_slot[s] == team and np.isfinite(self.feet[f, s, 1])]
        if not ys:
            return None
        return "near" if np.median(ys) >= C.NET_Y else "far"

    def save(self, path: Path) -> None:
        import json
        np.savez_compressed(path, fps=self.fps, n_frames=self.n_frames, width=self.width, height=self.height,
                            kp=self.kp.astype(np.float16), bbox=self.bbox.astype(np.float32),
                            feet=self.feet.astype(np.float32), ball=self.ball.astype(np.float32),
                            team_of_slot=np.array(self.team_of_slot), slot_meta=json.dumps(self.slot_meta),
                            source=self.source, frame_range=np.array(self.frame_range),
                            info=json.dumps(self.info))

    @classmethod
    def load(cls, path: Path) -> "Tracks":
        import json
        z = np.load(path, allow_pickle=False)
        return cls(float(z["fps"]), int(z["n_frames"]), int(z["width"]), int(z["height"]),
                   z["kp"].astype(np.float32), z["bbox"], z["feet"], z["ball"],
                   [str(t) for t in z["team_of_slot"]], json.loads(str(z["slot_meta"])),
                   str(z["source"]), tuple(int(v) for v in z["frame_range"]),
                   json.loads(str(z["info"])) if "info" in z else {})


def feet_point_px(kp: np.ndarray, bbox: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """Floor contact point in pixels for (...,17,3) keypoints: midpoint of
    visible ankles (lowest ankle if only one), else bbox bottom-centre."""
    kp = np.asarray(kp)
    la, ra = kp[..., C.KP["left_ankle"], :], kp[..., C.KP["right_ankle"], :]
    lv, rv = la[..., 2] > min_conf, ra[..., 2] > min_conf
    both = lv & rv
    out = np.empty(kp.shape[:-2] + (2,), np.float64)
    bx = bbox[..., 0] + bbox[..., 2] / 2.0
    by = bbox[..., 1] + bbox[..., 3]
    out[..., 0] = np.where(both, (la[..., 0] + ra[..., 0]) / 2, np.where(lv, la[..., 0], np.where(rv, ra[..., 0], bx)))
    # small offset: ankle joint sits ~8 cm above the floor contact
    ankle_y = np.where(both, np.maximum(la[..., 1], ra[..., 1]), np.where(lv, la[..., 1], np.where(rv, ra[..., 1], np.nan)))
    out[..., 1] = np.where(np.isfinite(ankle_y), ankle_y + 0.02 * bbox[..., 3], by)
    return out
