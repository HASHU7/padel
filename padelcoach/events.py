"""Ball event detection from the 2D trajectory + player poses.

Events are impulses in the ball trajectory (sharp change of direction/speed):
  hit     - impulse within racket reach of a player (hitter = that player)
  bounce  - floor bounce: on-screen vertical motion flips downward->upward and the
            floor projection lies inside the court (only here is the floor
            homography valid for the ball)
  wall    - impulse whose floor projection lies at/outside the enclosure
            (back glass, side glass, side metal mesh)
  deflect - other impulse (net cord, frame, uncertain)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import config as C
from .court import CourtCalibration, wall_region
from .tracks import N_SLOTS, Tracks


@dataclass
class BallEvent:
    frame: int
    kind: str
    px: Tuple[float, float]
    court: Tuple[float, float]
    side: Optional[str] = None
    player: Optional[int] = None
    wall: Optional[str] = None
    confidence: float = 0.5
    reach_ratio: float = float("nan")
    v_in: Tuple[float, float] = (0.0, 0.0)
    v_out: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["px"] = [round(float(v), 1) for v in self.px]
        d["court"] = [round(float(v), 2) if np.isfinite(v) else None for v in self.court]
        d["v_in"] = [round(float(v), 2) for v in self.v_in]
        d["v_out"] = [round(float(v), 2) for v in self.v_out]
        d["reach_ratio"] = None if not np.isfinite(self.reach_ratio) else round(float(self.reach_ratio), 2)
        d["confidence"] = round(float(self.confidence), 2)
        return d


def _nearest_valid(valid: np.ndarray, f: int, lo: int, hi: int, target: int) -> Optional[int]:
    best = None
    for g in range(lo, hi + 1):
        j = f + g
        if 0 <= j < len(valid) and valid[j]:
            if best is None or abs(g - target) < abs(best - f - target):
                best = j
    return best


def _impulse_scores(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(b)
    valid = np.isfinite(b[:, 0])
    score = np.zeros(N)
    vin = np.zeros((N, 2))
    vout = np.zeros((N, 2))
    for f in np.where(valid)[0]:
        a = _nearest_valid(valid, f, -6, -1, -3)
        c = _nearest_valid(valid, f, 1, 6, 3)
        if a is None or c is None:
            continue
        vi = (b[f, :2] - b[a, :2]) / (f - a)
        vo = (b[c, :2] - b[f, :2]) / (c - f)
        ni, no = np.linalg.norm(vi), np.linalg.norm(vo)
        vin[f], vout[f] = vi, vo
        if max(ni, no) < 1.5:
            continue
        if min(ni, no) > 0.8:
            ang = np.degrees(np.arccos(np.clip(vi @ vo / (ni * no), -1, 1)))
        else:
            ang = 0.0
        ratio = max(ni, no) / max(min(ni, no), 0.3)
        score[f] = ang / 40.0 + max(0.0, np.log2(ratio) - 0.6)
    return score, vin, vout


def _gap_candidates(b: np.ndarray, min_gap: int = 3, max_gap: int = 14) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Direction changes hidden inside detection gaps (ball occluded at contact)."""
    valid = np.isfinite(b[:, 0])
    idx = np.where(valid)[0]
    out = []
    for a, c in zip(idx[:-1], idx[1:]):
        g = c - a
        if min_gap <= g <= max_gap:
            a0 = _nearest_valid(valid, a, -5, -2, -3)
            c1 = _nearest_valid(valid, c, 2, 5, 3)
            if a0 is None or c1 is None:
                continue
            vi = (b[a, :2] - b[a0, :2]) / (a - a0)
            vo = (b[c1, :2] - b[c, :2]) / (c1 - c)
            ni, no = np.linalg.norm(vi), np.linalg.norm(vo)
            if ni < 1 or no < 1:
                continue
            ang = np.degrees(np.arccos(np.clip(vi @ vo / (ni * no), -1, 1)))
            if ang > 45:
                out.append(((a + c) // 2, b[a, :2].copy(), vi, vo))
    return out


def _player_reach(tracks: Tracks, calib: CourtCalibration, f: int, p: np.ndarray) -> Tuple[Optional[int], float]:
    """Best player (slot) and distance/reach ratio for ball pixel p at frame f."""
    best, best_r = None, np.inf
    for df in (0, -1, 1, -2, 2):
        g = f + df
        if not (0 <= g < tracks.n_frames):
            continue
        for s in range(N_SLOTS):
            if not np.isfinite(tracks.feet[g, s, 0]):
                continue
            kp = tracks.kp[g, s]
            bb = tracks.bbox[g, s]
            ppm = calib.pixel_scale_at(calib.to_image(tracks.feet[g, s][None]))[0]
            reach = max(0.85 * ppm, 0.45 * bb[3], 18.0)  # arm + racket ~0.85 m
            ds = []
            for w in (C.KP["left_wrist"], C.KP["right_wrist"]):
                if kp[w, 2] > 0.25:
                    ds.append(np.hypot(*(kp[w, :2] - p)))
            # overheads / occluded wrists: inside bbox extended upward by the racket
            x, y, bw, bh = bb
            if x - 0.25 * bw <= p[0] <= x + 1.25 * bw and y - 0.45 * bh <= p[1] <= y + 0.75 * bh:
                ds.append(0.6 * reach)
            if ds:
                r = min(ds) / reach
                if r < best_r:
                    best, best_r = s, r
        if best is not None and best_r < 1.0:
            break
    return best, best_r


def detect_events(tracks: Tracks, calib: CourtCalibration, thresh: float = 1.0) -> List[BallEvent]:
    b = tracks.ball
    fps = tracks.fps
    score, vin, vout = _impulse_scores(b)
    cands = []  # (frame, score, p, vi, vo)
    for f in range(2, len(score) - 2):
        if score[f] >= thresh and score[f] == score[max(0, f - 3):f + 4].max():
            cands.append((f, score[f], b[f, :2].copy(), vin[f].copy(), vout[f].copy()))
    for f, p, vi, vo in _gap_candidates(b):
        cands.append((f, 1.5, p, vi, vo))
    # NMS
    min_sep = max(4, int(round(5 * fps / 30)))
    cands.sort(key=lambda c: -c[1])
    taken: List[tuple] = []
    for c in cands:
        if all(abs(c[0] - t[0]) >= min_sep for t in taken):
            taken.append(c)
    taken.sort(key=lambda c: c[0])

    events: List[BallEvent] = []
    for f, sc, p, vi, vo in taken:
        P = calib.to_court(p[None])[0]
        player, r = _player_reach(tracks, calib, f, p)
        conf_imp = float(min(1.0, sc / 2.5))
        if player is not None and r < 1.0:
            # refine hit frame: closest ball-wrist approach within +-4 frames
            best_f, best_d = f, np.inf
            for g in range(max(0, f - 4), min(tracks.n_frames, f + 5)):
                if not np.isfinite(b[g, 0]) or not np.isfinite(tracks.feet[g, player, 0]):
                    continue
                kp = tracks.kp[g, player]
                for w in (C.KP["left_wrist"], C.KP["right_wrist"]):
                    if kp[w, 2] > 0.25:
                        d = np.hypot(*(kp[w, :2] - b[g, :2]))
                        if d < best_d:
                            best_d, best_f = d, g
            side = tracks.side_of_slot(best_f, player) or tracks.side_of_slot(f, player)
            events.append(BallEvent(int(best_f), "hit", tuple(p), tuple(P), side, int(player), None,
                                    float(np.clip(0.4 + 0.4 * conf_imp + 0.3 * (1 - r), 0, 1)), float(r),
                                    tuple(vi), tuple(vo)))
            continue
        x, y = P
        inside = -0.15 <= x <= C.COURT_W + 0.15 and -0.15 <= y <= C.COURT_L + 0.15
        down_up = vi[1] > 0.4 and vo[1] < -0.2
        up_down = vi[1] < -0.4 and vo[1] > 0.2
        x_flip = np.sign(vi[0]) != np.sign(vo[0]) and abs(vi[0]) > 0.8 and abs(vo[0]) > 0.5
        wr = wall_region((x, y), margin=0.7) if np.isfinite(x) else None
        if down_up and inside and wr not in ("side_glass", "side_mesh", "far_back"):
            kind, wall, conf = "bounce", None, 0.55 + 0.35 * conf_imp
            if wr == "near_back":
                conf -= 0.2  # near-glass impacts look like bounces from this camera
        elif not inside or wr is not None:
            kind = "wall"
            if not inside:
                wall = "far_back" if y < 0 else ("near_back" if y > C.COURT_L else
                       ("side_glass" if (y < C.SIDE_GLASS_LEN or y > C.COURT_L - C.SIDE_GLASS_LEN) else "side_mesh"))
            else:
                wall = wr
            if wall in ("side_glass", "side_mesh") and not x_flip and not (up_down or down_up):
                kind, wall = "deflect", None
            conf = 0.45 + 0.35 * conf_imp
        elif up_down:
            kind, wall, conf = "deflect", None, 0.3  # likely lob apex noise / net cord
        else:
            kind, wall, conf = "deflect", None, 0.3 + 0.2 * conf_imp
        side = None
        if np.isfinite(y):
            side = "near" if y >= C.NET_Y else "far"
        events.append(BallEvent(int(f), kind, tuple(p), (float(x), float(y)), side, None, wall, float(conf),
                                float("nan"), tuple(vi), tuple(vo)))
    return _enforce_alternation(_dedupe_hits(events, fps), tracks)


def _enforce_alternation(events: List[BallEvent], tracks: Tracks, split_gap_s: float = 2.5) -> List[BallEvent]:
    """In padel the ball must alternate between the teams. A run of hits by the
    same team within one rally means all but the last are not real shots
    (pre-serve ball bouncing, ball passing next to the partner, picking the
    ball up). Those are relabelled 'touch'. Runs separated by a long gap
    belong to different rallies and are judged separately."""
    fps = tracks.fps
    hits = [e for e in events if e.kind == "hit"]
    i = 0
    while i < len(hits):
        team = tracks.team_of_slot[hits[i].player]
        j = i
        while j + 1 < len(hits) and tracks.team_of_slot[hits[j + 1].player] == team:
            j += 1
        if j > i:
            run = hits[i:j + 1]
            # split the run at long pauses (end of one rally / start of the next)
            groups, cur = [], [run[0]]
            for a, b in zip(run[:-1], run[1:]):
                if (b.frame - a.frame) / fps > split_gap_s:
                    groups.append(cur)
                    cur = []
                cur.append(b)
            groups.append(cur)
            for gi, g in enumerate(groups):
                keep = g[0] if (gi == 0 and len(groups) > 1) else g[-1]
                # the first group of a split run ends the previous rally: its first hit is
                # the real shot; later groups precede the next rally: keep their last hit
                if gi == 0 and len(groups) > 1:
                    keep = g[0]
                for e in g:
                    if e is not keep:
                        e.kind = "touch"
        i = j + 1
    return events


def _dedupe_hits(events: List[BallEvent], fps: float) -> List[BallEvent]:
    """Keep one hit per swing: hits by the same player closer than the minimum
    swing interval are merged (higher confidence wins)."""
    min_gap = int(C.MIN_FRAMES_BETWEEN_HITS * fps / 30)
    out: List[BallEvent] = []
    for e in events:
        if e.kind == "hit" and out:
            prev = next((o for o in reversed(out) if o.kind == "hit"), None)
            if prev is not None and prev.player == e.player and e.frame - prev.frame < min_gap:
                if e.confidence > prev.confidence:
                    out.remove(prev)
                    out.append(e)
                continue
        out.append(e)
    out.sort(key=lambda e: e.frame)
    return out
