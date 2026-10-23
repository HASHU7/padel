"""Broadcast scoreboard -> authoritative point winners.

Timeline JSON (written by colab/padel_gpu_inference.ipynb, by a local OCR
engine if one is installed, or by hand):
  {"rows": ["JOS / SAN", "SAL / TRI"],
   "samples": [{"frame": 520, "rows": [{"games": [0], "points": "15"}, {"games": [0], "points": "0"}]}, ...]}
Rows are mapped to teams A/B by agreement with the detected serving team and
the inferred point winners.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

ORDER = {"0": 0, "00": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "ADV": 4}
SCOREBOARD_DIR = C.WORKSPACE / "scoreboard"
SCOREBOARD_DIR.mkdir(parents=True, exist_ok=True)


def timeline_path(video: Path) -> Path:
    return SCOREBOARD_DIR / f"{Path(video).stem}.json"


def load_timeline(video: Path) -> Optional[dict]:
    p = timeline_path(video)
    return json.loads(p.read_text()) if p.exists() else None


def _pts_value(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().upper()
    if s in ORDER:
        return ORDER[s]
    if s.isdigit():
        return int(s) + 100  # tie-break counts
    return None


def _stable(samples: List[dict], k: int = 3) -> List[dict]:
    """Keep readings that repeat in k consecutive samples (OCR noise filter)."""
    out = []
    for i in range(len(samples) - k + 1):
        win = samples[i:i + k]
        key = json.dumps(win[0]["rows"], sort_keys=True)
        if all(json.dumps(w["rows"], sort_keys=True) == key for w in win[1:]):
            if not out or json.dumps(out[-1]["rows"], sort_keys=True) != key:
                out.append(win[0])
    return out


POINT_TOKENS = {"15", "30", "40", "A"}


def split_row(row: dict) -> Tuple[List[int], Optional[str]]:
    """(games per set, points or None) from an OCR row. Broadcast layouts show
    either the points box only, set games only, or set games + points box. The
    points cell is the rightmost token and is one of 0/15/30/40/A; games are
    small integers to its left. Works for old timelines too (games, points)."""
    nums = [str(n).upper() for n in (row.get("nums") or (list(row.get("games") or []) + [row.get("points")]))
            if n is not None and str(n) != ""]
    nums = ["A" if n in ("AD", "ADV") else n for n in nums]
    if not nums:
        return [], None
    last = nums[-1]
    if len(nums) == 1 and last in ("0", "00"):
        return [], None          # lone "0": points 0 or 0 games - ambiguous
    if last in POINT_TOKENS or (last in ("0", "00") and len(nums) >= 2):
        pts, rest = ("0" if last == "00" else last), nums[:-1]
    else:
        pts, rest = None, nums
    games = [int(n) for n in rest if n.isdigit() and int(n) <= 7]
    return games, pts


def point_changes(tl: dict) -> List[dict]:
    """Chronological scoreboard events by row index (0/1):
    kind 'game'  - a row's games count went up by one (that row won the game,
                   i.e. the last point before it);
    kind 'point' - the points cell changed in that row's favour."""
    st = _stable(sorted(tl.get("samples", []), key=lambda s: s["frame"]), k=3)
    out = []
    last_games, last_pts = None, None
    for smp in st:
        rows = smp["rows"]
        if len(rows) != 2:
            continue
        parsed = [split_row(r) for r in rows]
        games = [sum(g) if g else None for g, _ in parsed]
        pts = [_pts_value(p) if p is not None else None for _, p in parsed]
        f = int(smp["frame"])
        if None not in games:
            if last_games is not None:
                d = [games[0] - last_games[0], games[1] - last_games[1]]
                sets = [parsed[0][0], parsed[1][0]] if len(parsed[0][0]) == len(parsed[1][0]) else None
                if d == [1, 0]:
                    out.append({"frame": f, "row": 0, "kind": "game", "sets_rows": sets})
                elif d == [0, 1]:
                    out.append({"frame": f, "row": 1, "kind": "game", "sets_rows": sets})
            last_games = games
        if None not in pts:
            if last_pts is not None and pts != [0, 0]:
                d = [pts[0] - last_pts[0], pts[1] - last_pts[1]]
                w = None
                if d[0] > 0 and d[1] == 0:
                    w = 0
                elif d[1] > 0 and d[0] == 0:
                    w = 1
                elif d[0] < 0 and d[1] == 0 and last_pts[0] == 4:   # advantage lost
                    w = 1
                elif d[1] < 0 and d[0] == 0 and last_pts[1] == 4:
                    w = 0
                if w is not None:
                    out.append({"frame": f, "row": w, "kind": "point"})
            last_pts = pts
    return out


def map_rows_to_teams(changes: List[dict], points: List, fps: float) -> Tuple[Dict[int, str], float]:
    """Choose row->team mapping maximising agreement with inferred winners."""
    def agree(mapping):
        n = ok = 0
        for ch in changes:
            cands = [p for p in points if p.end_frame <= ch["frame"] and ch["frame"] - p.end_frame < 25 * fps]
            if cands and cands[-1].winner:
                n += 1
                ok += cands[-1].winner == mapping[ch["row"]]
        return ok / max(n, 1)
    m1, m2 = {0: "A", 1: "B"}, {0: "B", 1: "A"}
    a1, a2 = agree(m1), agree(m2)
    return (m1, a1) if a1 >= a2 else (m2, a2)


def scoreboard_points(video: Path, points: List, fps: float) -> Tuple[Optional[List[dict]], dict]:
    tl = load_timeline(video)
    if not tl:
        return None, {"available": False}
    ch = point_changes(tl)
    mapping, agreement = map_rows_to_teams(ch, points, fps)
    names = tl.get("rows") or []
    team_names = {mapping[i]: names[i] for i in range(min(2, len(names)))}
    def sets_ab(c):  # per-set games as [[A, B], ...] in team order
        sr = c.get("sets_rows")
        if not sr:
            return None
        a_row = 0 if mapping[0] == "A" else 1
        return [[sr[a_row][i], sr[1 - a_row][i]] for i in range(len(sr[0]))]
    return ([{"frame": c["frame"], "winner": mapping[c["row"]], "kind": c.get("kind", "point"), "sets": sets_ab(c)}
             for c in ch],
            {"available": True, "changes": len(ch), "games_read": sum(c.get("kind") == "game" for c in ch),
             "points_read": sum(c.get("kind") == "point" for c in ch), "row_mapping": mapping, "agreement_with_inferred": round(agreement, 2),
             "team_names": team_names})


# ------------------------------------------------------------------ local OCR
def _ocr_engine():
    """Use an OCR library if one is already installed (nothing is downloaded)."""
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return lambda img: [(t, float(np.mean([p[1] for p in b])), float(np.mean([p[0] for p in b])))
                            for b, t, c in reader.readtext(img) if c > 0.3]
    except Exception:
        pass
    try:
        import pytesseract  # type: ignore
        def run(img):
            d = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
            return [(t, d["top"][i] + d["height"][i] / 2, d["left"][i] + d["width"][i] / 2)
                    for i, t in enumerate(d["text"]) if t.strip()]
        return run
    except Exception:
        return None


def parse_ocr_tokens(tokens: List[Tuple[str, float, float]]) -> Optional[dict]:
    """tokens (text, y, x) inside the scoreboard ROI -> {'names', 'rows'}."""
    if not tokens:
        return None
    ys = np.array([t[1] for t in tokens])
    mid = (ys.min() + ys.max()) / 2
    rows = [sorted([t for t in tokens if t[1] < mid], key=lambda t: t[2]),
            sorted([t for t in tokens if t[1] >= mid], key=lambda t: t[2])]
    out, names = [], []
    for r in rows:
        txt = [t[0].strip().upper() for t in r]
        name = " ".join(w for w in txt if re.search(r"[A-Z]{2,}", w))
        nums = [w for w in txt if re.fullmatch(r"\d{1,2}|A|AD|ADV", w)]
        if not nums:
            return None
        pts = nums[-1]
        games = [int(n) for n in nums[:-1] if n.isdigit() and int(n) <= 7]
        out.append({"games": games, "points": pts, "nums": nums})
        names.append(name)
    return {"names": names, "rows": out}


def read_timeline_local(video: Path, roi: Tuple[int, int, int, int], step: int = 15,
                        progress=None) -> Optional[dict]:
    engine = _ocr_engine()
    if engine is None:
        return None
    from .video import iter_frames, probe
    info = probe(video)
    x, y, w, h = roi
    samples, names = [], None
    frames = range(0, info.n_frames, step)
    for i, (f, img) in enumerate(iter_frames(video, frames)):
        crop = img[y:y + h, x:x + w]
        r = parse_ocr_tokens(engine(crop))
        if r:
            names = names or r["names"]
            samples.append({"frame": f, "rows": r["rows"]})
        if progress and i % 100 == 0:
            progress(i / len(frames), "reading scoreboard")
    tl = {"rows": names or ["Row 1", "Row 2"], "samples": samples, "source": "local-ocr"}
    timeline_path(video).write_text(json.dumps(tl))
    return tl
