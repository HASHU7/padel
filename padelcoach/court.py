"""Court model and camera calibration (floor-plane homography).

A homography is only valid for points ON THE FLOOR. Use player ankles (not
hips) for positions and ball positions only at floor bounces. For airborne
points use `pixel_scale_at` to reason about heights instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C

# Calibration verified on the WPT Barcelona 2022 Master Final main camera
# (identical for women's and men's finals, residual < 2.5 cm). Pixel coords
# for 1920x1080.
PADELTRACKER_BROADCAST_POINTS = {
    "far_left_corner": (553.0, 318.0),
    "far_right_corner": (1368.0, 318.0),
    "far_service_center": (960.0, 367.5),
    "near_service_center": (960.0, 786.0),
    "near_service_left": (195.0, 786.0),
    "near_service_right": (1730.0, 786.0),
}


@dataclass
class CourtCalibration:
    image_points: Dict[str, Tuple[float, float]]
    width: int = 1920
    height: int = 1080
    scoreboard_roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h (broadcast overlay)
    H: np.ndarray = field(init=False)
    H_inv: np.ndarray = field(init=False)
    reprojection_error_m: float = field(init=False, default=0.0)

    def __post_init__(self):
        names = [n for n in self.image_points if n in C.COURT_LANDMARKS]
        if len(names) < 4:
            raise ValueError("Calibration needs at least 4 known court landmarks")
        src = np.float32([self.image_points[n] for n in names])
        dst = np.float32([C.COURT_LANDMARKS[n] for n in names])
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise ValueError("Degenerate calibration points")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        self.reprojection_error_m = float(np.mean(np.linalg.norm(proj - dst, axis=1)))

    # ---------------------------------------------------------------- mapping
    def to_court(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,2) metres on the floor plane. NaN-safe."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        out = np.full_like(pts, np.nan)
        ok = np.isfinite(pts).all(1)
        if ok.any():
            out[ok] = cv2.perspectiveTransform(pts[ok].reshape(-1, 1, 2), self.H).reshape(-1, 2)
        return out

    def to_image(self, pts_m: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        out = np.full_like(pts, np.nan)
        ok = np.isfinite(pts).all(1)
        if ok.any():
            out[ok] = cv2.perspectiveTransform(pts[ok].reshape(-1, 1, 2), self.H_inv).reshape(-1, 2)
        return out

    def pixel_scale_at(self, pts_px: np.ndarray) -> np.ndarray:
        """Approximate pixels-per-metre (horizontal) at floor points. Used to
        convert image heights of standing players into metres."""
        m = self.to_court(pts_px)
        a = self.to_image(m - np.array([0.5, 0.0]))
        b = self.to_image(m + np.array([0.5, 0.0]))
        return np.linalg.norm(b - a, axis=1)

    def court_polygon_px(self, margin_m: float = 0.0) -> np.ndarray:
        g = margin_m
        corners = np.array([[-g, -g], [C.COURT_W + g, -g], [C.COURT_W + g, C.COURT_L + g], [-g, C.COURT_L + g]])
        return self.to_image(corners).astype(np.float32)

    def court_lines_px(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Line segments (image px) for drawing the court model."""
        segs_m = [
            ((0, 0), (10, 0)), ((10, 0), (10, 20)), ((10, 20), (0, 20)), ((0, 20), (0, 0)),
            ((0, C.FAR_SERVICE_Y), (10, C.FAR_SERVICE_Y)), ((0, C.NEAR_SERVICE_Y), (10, C.NEAR_SERVICE_Y)),
            ((5, C.FAR_SERVICE_Y), (5, C.NEAR_SERVICE_Y)), ((0, 10), (10, 10)),
        ]
        out = []
        for a, b in segs_m:
            pa, pb = self.to_image(np.array([a, b], float))
            out.append((pa, pb))
        return out

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "image_points": {k: list(v) for k, v in self.image_points.items()},
            "width": self.width, "height": self.height,
            "scoreboard_roi": list(self.scoreboard_roi) if self.scoreboard_roi else None,
            "reprojection_error_m": round(self.reprojection_error_m, 4),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "CourtCalibration":
        roi = d.get("scoreboard_roi")
        return cls({k: tuple(v) for k, v in d["image_points"].items()},
                   int(d.get("width", 1920)), int(d.get("height", 1080)),
                   tuple(roi) if roi else None)

    @classmethod
    def load(cls, path: Path) -> "CourtCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def padeltracker_broadcast(cls, width: int = 1920, height: int = 1080) -> "CourtCalibration":
        sx, sy = width / 1920.0, height / 1080.0
        pts = {k: (x * sx, y * sy) for k, (x, y) in PADELTRACKER_BROADCAST_POINTS.items()}
        return cls(pts, width, height, scoreboard_roi=(int(30 * sx), int(50 * sy), int(470 * sx), int(150 * sy)))


def calibration_path_for(video_path: Path) -> Path:
    return C.CALIB_DIR / f"{Path(video_path).stem}.json"


def load_calibration_for(video_path: Path, width: int = 1920, height: int = 1080) -> Tuple[CourtCalibration, str]:
    """Saved per-video calibration, else the PadelTracker100 broadcast preset."""
    p = calibration_path_for(video_path)
    if p.exists():
        return CourtCalibration.load(p), "saved"
    return CourtCalibration.padeltracker_broadcast(width, height), "preset"


# ---------------------------------------------------------------------- zones
def side_of(y_m: float) -> str:
    return "near" if y_m >= C.NET_Y else "far"


def dist_from_net(y_m) -> np.ndarray:
    return np.abs(np.asarray(y_m, float) - C.NET_Y)


def zone_of(y_m: float) -> str:
    d = abs(y_m - C.NET_Y)
    if d < C.ZONE_NET_MAX:
        return "net"
    if d < C.ZONE_TRANSITION_MAX:
        return "transition"
    return "defense"


def lateral_of(x_m: float, side: str) -> str:
    """'left'/'right' half of court from the perspective of a player on `side`
    facing the net. Near players face away from the camera (their left = image
    left); far players face the camera (their left = image right)."""
    left_img = x_m < C.CENTER_X
    if side == "near":
        return "left" if left_img else "right"
    return "right" if left_img else "left"


def service_box(side: str, box: str) -> Tuple[float, float, float, float]:
    """Return (x0, x1, y0, y1) of a service box on `side` ('near'/'far'),
    `box` = 'left'/'right' from the perspective of the player on that side."""
    if side == "near":
        y0, y1 = C.NET_Y, C.NEAR_SERVICE_Y
        x0, x1 = (0.0, C.CENTER_X) if box == "left" else (C.CENTER_X, C.COURT_W)
    else:
        y0, y1 = C.FAR_SERVICE_Y, C.NET_Y
        x0, x1 = (C.CENTER_X, C.COURT_W) if box == "left" else (0.0, C.CENTER_X)
    return x0, x1, y0, y1


def in_box(xy: Sequence[float], box: Tuple[float, float, float, float], tol: float = 0.15) -> bool:
    x0, x1, y0, y1 = box
    x, y = xy
    return (x0 - tol) <= x <= (x1 + tol) and (y0 - tol) <= y <= (y1 + tol)


def wall_region(xy_m: Sequence[float], margin: float = 0.6) -> Optional[str]:
    """Classify the nearest enclosure element for a floor-projected point.
    Returns 'far_back', 'near_back', 'side_glass', 'side_mesh' or None."""
    x, y = xy_m
    if y < margin:
        return "far_back"
    if y > C.COURT_L - margin:
        return "near_back"
    if x < margin or x > C.COURT_W - margin:
        if y < C.SIDE_GLASS_LEN or y > C.COURT_L - C.SIDE_GLASS_LEN:
            return "side_glass"
        return "side_mesh"
    return None
