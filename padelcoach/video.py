"""Video helpers: metadata, sequential sparse frame sampling, single-frame grab."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import cv2
import numpy as np

from . import config as C


@dataclass
class VideoInfo:
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def to_dict(self) -> dict:
        return {"path": str(self.path), "name": self.path.name, "fps": self.fps, "n_frames": self.n_frames,
                "width": self.width, "height": self.height, "duration_s": round(self.duration_s, 2)}


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    info = VideoInfo(Path(path), float(cap.get(cv2.CAP_PROP_FPS) or 30.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frame(path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def iter_frames(path: Path, wanted: Iterable[int], end: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (idx, frame) for the sorted set of wanted frame indices, decoding
    sequentially (grab() for skipped frames) which is much faster than seeking."""
    wanted = sorted(set(int(w) for w in wanted))
    if not wanted:
        return
    span = (wanted[-1] - wanted[0] + 1)
    sparse = len(wanted) / max(span, 1) < 0.1
    cap = None
    if sparse:  # hardware decode makes grab() ~3x faster; worth it when most frames are skipped
        try:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            if not cap.isOpened():
                cap = None
        except Exception:
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(str(path))
    last = wanted[-1] if end is None else min(wanted[-1], end)
    i = 0
    wi = 0
    if wanted[0] > 300:  # jump close to the first wanted frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        i = wanted[0]
    while i <= last and wi < len(wanted):
        if i == wanted[wi]:
            ok, img = cap.read()
            if not ok:
                break
            yield i, img
            wi += 1
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()


def list_videos() -> list:
    seen, out = set(), []
    for d in C.VIDEO_SEARCH_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in C.VIDEO_EXTENSIONS and p.name not in seen and "padel_" not in p.stem:
                seen.add(p.name)
                out.append(p)
    return out


def find_video(name: str) -> Path:
    for d in C.VIDEO_SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(name)
