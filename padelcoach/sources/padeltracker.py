"""PadelTracker100 ground-truth loader (CC-BY-4.0, zenodo.org/records/14653706).

Produces RawDetections from the annotated pose/ball JSON so the full pipeline
can run on annotated matches (perfect perception), and exposes shot labels
for evaluation. Annotation `image_id` is 1-based; frames are 0-based.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..tracks import RawDetections

DATASETS = {
    "2022_BCN_FinalF_1": {"pose": "2022_BCN_FinalF_1_pose.json", "ball": "2022_BCN_FinalF_1_ball.json",
                          "shots": "2022_BCN_FinalF_1_shots.csv", "legacy_cache": "womens_final"},
    "2022_BCN_FinalM_1": {"pose": "2022_BCN_FinalM_1_pose.json", "ball": "2022_BCN_FinalM_1_ball.json",
                          "shots": "2022_BCN_FinalM_1_shots.csv", "legacy_cache": "mens_final",
                          # The released pose/ball JSON (53,888 images) starts 65 frames after
                          # the video (53,953 frames); shot labels are aligned with the video.
                          # Verified by matching annotated ball positions to video pixels.
                          "frame_offset": 65},
}


def dataset_for_video(video_name: str) -> Optional[str]:
    stem = Path(video_name).stem
    for k in DATASETS:
        if stem == k or stem.startswith(k + "_"):
            return k
    return None


def has_labels(video_name: str) -> bool:
    k = dataset_for_video(video_name)
    return bool(k) and (C.LABELS_DIR / DATASETS[k]["pose"]).exists()


def _cache_dir() -> Path:
    d = C.LABELS_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_raw(key: str, fps: float = 30.0, width: int = 1920, height: int = 1080,
             n_frames: Optional[int] = None) -> RawDetections:
    meta = DATASETS[key]
    off = int(meta.get("frame_offset", 0))
    cache = _cache_dir() / f"{key}_raw_v2.npz"
    if cache.exists():
        return RawDetections.load(cache)

    legacy = _cache_dir() / f"{meta['legacy_cache']}_pose_cache.pkl"
    if legacy.exists():
        with open(legacy, "rb") as f:
            poses = pickle.load(f)
    else:
        with open(C.LABELS_DIR / meta["pose"], "r", encoding="utf-8") as f:
            pj = json.load(f)
        poses = {}
        for a in pj["annotations"]:
            poses.setdefault(a["image_id"] - 1, []).append({"bbox": a["bbox"], "keypoints": a["keypoints"]})

    with open(C.LABELS_DIR / meta["ball"], "r", encoding="utf-8") as f:
        bj = json.load(f)
    ball_cat = {c["id"] for c in bj["categories"] if c["name"].lower() == "ball"}
    n_img = len(bj["images"])
    n = n_frames or (max(n_img, (max(poses) + 1) if poses else 0) + off)

    raw = RawDetections(fps, n, width, height, source=f"padeltracker100:{key}")
    for fid, plist in poses.items():
        fid = int(fid) + off
        if fid >= n:
            continue
        kps = np.array([np.array(p["keypoints"], np.float32).reshape(17, 3) for p in plist], np.float32)
        # GT visibility flags (0/2) -> confidences (0/1)
        kps[..., 2] = (kps[..., 2] > 0).astype(np.float32)
        raw.persons_kp[int(fid)] = kps
        raw.persons_bbox[int(fid)] = np.array([p["bbox"] for p in plist], np.float32)
        raw.persons_score[int(fid)] = np.ones(len(plist), np.float32)
    for a in bj["annotations"]:
        if a["category_id"] not in ball_cat or not a.get("bbox"):
            continue
        fid = a["image_id"] - 1 + off
        if 0 <= fid < n:
            x, y, w, h = a["bbox"]
            raw.ball[fid] = (x + w / 2.0, y + h / 2.0, 1.0)
    raw.save(cache)
    return raw


def load_shot_labels(key: str) -> pd.DataFrame:
    """Per-frame shot labels: columns frame, has_shot, category ('0' = none)."""
    df = pd.read_csv(C.LABELS_DIR / DATASETS[key]["shots"], sep=";")
    df["frame"] = df["file_name"].str.extract(r"(\d+)").astype(int)
    df["category"] = df["category"].astype(str)
    return df[["frame", "has_shot", "category"]]


def shot_label_events(key: str, merge_gap: int = 6) -> List[dict]:
    """Contiguous labelled shot windows -> list of {start, end, category}."""
    df = load_shot_labels(key)
    ev: List[dict] = []
    cur = None
    for f, cat in zip(df["frame"].values, df["category"].values):
        if cat != "0":
            if cur and cur["category"] == cat and f - cur["end"] <= merge_gap:
                cur["end"] = f
            else:
                if cur:
                    ev.append(cur)
                cur = {"start": int(f), "end": int(f), "category": cat}
    if cur:
        ev.append(cur)
    return ev


def load_ball_gt(key: str, n_frames: Optional[int] = None) -> np.ndarray:
    """Annotated ball centres only (N,3) in video frames - reads just the small
    ball JSON (the pose JSON is 110-130 MB and not needed for ball metrics)."""
    meta = DATASETS[key]
    off = int(meta.get("frame_offset", 0))
    with open(C.LABELS_DIR / meta["ball"], "r", encoding="utf-8") as f:
        bj = json.load(f)
    ball_cat = {c["id"] for c in bj["categories"] if c["name"].lower() == "ball"}
    n = n_frames or (len(bj["images"]) + off)
    ball = np.full((n, 3), np.nan, np.float32)
    for a in bj["annotations"]:
        if a["category_id"] in ball_cat and a.get("bbox"):
            fid = a["image_id"] - 1 + off
            if 0 <= fid < n:
                x, y, w, h = a["bbox"]
                ball[fid] = (x + w / 2.0, y + h / 2.0, 1.0)
    return ball
