"""Build the Colab training bundle (feature extraction only - no training here).

Contents of colab_bundle/padel_bundle.zip:
  padelcoach/                 package source (Colab imports it)
  colab_scripts/              training / inference scripts used by the notebooks
  labels/*.json, *.csv        PadelTracker100 annotations (ball, pose, shots)
  calibrations/*.json         court calibrations (scoreboard ROI etc.)
  shot_sequences.npz          X (n, 31, 55) pose/ball sequences, y labels, match ids
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from . import config as C
from .court import CourtCalibration
from .shotfeatures import sequence_features, window_indices
from .sources import padeltracker as pt
from .tracks import Tracks

BUNDLE_DIR = C.PROJECT_ROOT / "colab_bundle"


def shot_sequences() -> dict:
    X, y, match, frames = [], [], [], []
    classes = list(C.BASE_STROKES)
    for key in pt.DATASETS:
        rd = C.RESULTS_DIR / key
        if not (rd / "tracks.npz").exists():
            print(f"[bundle] skip {key}: analyse it first (python -m padelcoach analyze {key}.mp4 --mode labels)")
            continue
        tr = Tracks.load(rd / "tracks.npz")
        shots = json.loads((rd / "shots.json").read_text())
        hand = {}
        for s in shots:
            hand.setdefault(s["player"], []).append(s["contact"].get("hand") == "left")
        lefty = {p: (np.mean(v) > 0.5) for p, v in hand.items()}
        for g in pt.shot_label_events(key):
            if g["category"] not in classes:
                continue
            c = (g["start"] + g["end"]) // 2
            # matched detected hit inside the window, else nearest wrist-ball player at the centre
            cand = [s for s in shots if g["start"] - 6 <= s["frame"] <= g["end"] + 6]
            if cand:
                s = min(cand, key=lambda s: abs(s["frame"] - c))
                f, slot = s["frame"], s["player"]
            else:
                f, slot = c, None
                b = tr.ball[c]
                best = 1e9
                if np.isfinite(b[0]):
                    for q in range(tr.kp.shape[1]):
                        for w in (9, 10):
                            k = tr.kp[c, q, w]
                            if np.isfinite(k[0]) and k[2] > 0.2:
                                d = np.hypot(k[0] - b[0], k[1] - b[1])
                                if d < best:
                                    best, slot = d, q
                if slot is None:
                    continue
            if not np.isfinite(tr.feet[f, slot, 1]):
                continue
            far = tr.feet[f, slot, 1] < C.NET_Y
            win = window_indices(f, tr.n_frames)
            X.append(sequence_features(tr.kp[win, slot], tr.ball[win], far, lefty.get(slot, False)))
            y.append(classes.index(g["category"]))
            match.append(key)
            frames.append(f)
    return {"X": np.array(X, np.float32), "y": np.array(y, np.int64), "match": np.array(match), "frame": np.array(frames),
            "classes": np.array(classes)}


def build_bundle() -> Path:
    BUNDLE_DIR.mkdir(exist_ok=True)
    seq = shot_sequences()
    np.savez_compressed(BUNDLE_DIR / "shot_sequences.npz", **seq)
    print(f"[bundle] shot sequences: {len(seq['y'])} examples, classes {list(seq['classes'])}, "
          f"per class {np.bincount(seq['y'], minlength=len(seq['classes'])).tolist()}")
    zpath = BUNDLE_DIR / "padel_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (C.PROJECT_ROOT / "padelcoach").rglob("*.py"):
            z.write(p, p.relative_to(C.PROJECT_ROOT))
        for p in (C.PROJECT_ROOT / "padelcoach").rglob("*.json"):
            z.write(p, p.relative_to(C.PROJECT_ROOT))
        for p in (C.PROJECT_ROOT / "colab" / "scripts").glob("*.py"):
            z.write(p, Path("colab_scripts") / p.name)
        for key, m in pt.DATASETS.items():
            for k in ("pose", "ball", "shots"):
                p = C.LABELS_DIR / m[k]
                if p.exists():
                    z.write(p, Path("labels") / p.name)
        for p in C.CALIB_DIR.glob("*.json"):
            z.write(p, Path("calibrations") / p.name)
        z.write(BUNDLE_DIR / "shot_sequences.npz", "shot_sequences.npz")
    print(f"[bundle] wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB). Upload it to Google Drive: MyDrive/padel/")
    return zpath
