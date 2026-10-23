"""Evaluation against PadelTracker100 ground truth (annotated part of a match).

Hit detection: a detected hit matches a labelled shot window if it falls in
[start - tol, end + tol]; each window matches at most one hit.
Stroke classification: confusion of our base label vs the GT category for
matched hits.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from .shots import Shot
from .sources import padeltracker as pt


def evaluate_against_labels(key: str, shots: List[Shot], tol: int = 6) -> Dict:
    gt = pt.shot_label_events(key)
    if not gt:
        return {}
    last_labelled = max(e["end"] for e in gt)
    cand = [s for s in shots if s.frame <= last_labelled + tol]
    used = set()
    pairs = []
    for s in cand:
        best, bd = None, 1e9
        for j, g in enumerate(gt):
            if j in used:
                continue
            if g["start"] - tol <= s.frame <= g["end"] + tol:
                c = (g["start"] + g["end"]) / 2
                if abs(s.frame - c) < bd:
                    best, bd = j, abs(s.frame - c)
        if best is not None:
            used.add(best)
            pairs.append((s, gt[best]))
    tp = len(pairs)
    prec = tp / max(len(cand), 1)
    rec = tp / max(len(gt), 1)
    labels = ["Serve", "Forehand", "Backhand", "Smash", "Dropshot", "Other"]
    conf = {a: Counter() for a in labels}
    for s, g in pairs:
        conf[g["category"]][s.base] += 1
    correct = sum(conf[c][c] for c in labels)
    non_other = sum(sum(conf[c].values()) for c in labels if c != "Other")
    correct_no = sum(conf[c][c] for c in labels if c != "Other")
    per_class = {}
    for c in labels:
        support = sum(conf[c].values())
        predicted = sum(conf[a][c] for a in labels)
        tpc = conf[c][c]
        per_class[c] = {"support": support, "precision": round(tpc / predicted, 3) if predicted else None,
                        "recall": round(tpc / support, 3) if support else None}
    return {
        "labelled_frames": last_labelled + 1,
        "gt_shots": len(gt), "detected_hits": len(cand), "matched": tp,
        "hit_precision": round(prec, 3), "hit_recall": round(rec, 3),
        "hit_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        "timing_error_frames_median": float(np.median([abs(s.frame - (g["start"] + g["end"]) / 2) for s, g in pairs])) if pairs else None,
        "stroke_accuracy": round(correct / max(tp, 1), 3),
        "stroke_accuracy_excl_other": round(correct_no / max(non_other, 1), 3),
        "per_class": per_class,
        "confusion": {g: dict(c) for g, c in conf.items()},
    }
