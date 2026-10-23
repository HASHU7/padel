"""Stable player identities across a full match.

1. Court-space tracking of on-court persons into short, unambiguous tracklets
   (broken whenever two players come close, so tracklets never mix people).
2. Appearance (kit colour histograms) + height per tracklet from video frames.
3. Teams: 2-means on kit colour, then temporal consistency (partners share a
   half; teams only swap ends at changeovers).
4. Partners: split each team into its left ("reves") and right ("drive")
   player using lateral preference, height and appearance with cannot-link
   constraints for simultaneously visible tracklets.
Output: Tracks with slots 0,1 = team A (left, right), 2,3 = team B.
Team A is the team on the near side at the start of the video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config as C
from .court import CourtCalibration
from .tracks import N_SLOTS, RawDetections, Tracks, feet_point_px
from .video import iter_frames


@dataclass
class Tracklet:
    tid: int
    frames: List[int] = field(default_factory=list)
    det_idx: List[int] = field(default_factory=list)
    pos: List[np.ndarray] = field(default_factory=list)  # court metres
    hist: Optional[np.ndarray] = None
    legs_hist: Optional[np.ndarray] = None
    color_bgr: Optional[np.ndarray] = None
    height_m: float = float("nan")
    team: int = -1
    role: int = -1   # 0 = left (reves), 1 = right (drive)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def __len__(self):
        return len(self.frames)


def _on_court(xy: np.ndarray) -> np.ndarray:
    """Strict test for on-court padel positions.
    Excludes spectators behind the glass (Y < -0.35 or Y > 20.35) and
    outside side walls (X < -0.35 or X > 10.35).
    Allows out-of-court play up to 1.5 m only in the door zone near the net (Y 8.0-12.0m)."""
    x, y = xy[:, 0], xy[:, 1]
    in_bounds = (x >= -0.35) & (x <= C.COURT_W + 0.35) & (y >= -0.35) & (y <= C.COURT_L + 0.35)
    door_zone = (y >= 8.0) & (y <= 12.0) & (x >= -1.5) & (x <= C.COURT_W + 1.5)
    return in_bounds | door_zone


def build_tracklets(raw: RawDetections, calib: CourtCalibration, fps: float) -> List[Tracklet]:
    active: List[Tracklet] = []
    done: List[Tracklet] = []
    next_id = 0
    max_gap = int(0.5 * fps)
    for f in range(raw.n_frames):
        kp = raw.persons_kp.get(f)
        if kp is None or len(kp) == 0:
            pos = np.zeros((0, 2))
            idx = np.zeros(0, int)
        else:
            feet = feet_point_px(kp, raw.persons_bbox[f])
            pos = calib.to_court(feet)
            ok = np.isfinite(pos).all(1) & _on_court(pos)
            idx = np.where(ok)[0]
            pos = pos[ok]
            if len(idx) > N_SLOTS:
                # Prefer detections well inside the court; penalize hovering near outer walls
                inside_dist = np.minimum.reduce([
                    pos[:, 0] - (-0.1),
                    (C.COURT_W + 0.1) - pos[:, 0],
                    pos[:, 1] - (-0.1),
                    (C.COURT_L + 0.1) - pos[:, 1],
                ])
                sc = raw.persons_score[f][idx] + 0.25 * np.clip(inside_dist, -1.0, 1.5) - 0.04 * np.abs(pos[:, 0] - C.CENTER_X)
                # Split candidates by court half: near (y >= NET_Y) vs far (y < NET_Y)
                # In padel, exactly 2 players per side during play!
                near_sub = np.where(pos[:, 1] >= C.NET_Y)[0]
                far_sub = np.where(pos[:, 1] < C.NET_Y)[0]
                k_near = near_sub[np.argsort(-sc[near_sub])[:2]] if len(near_sub) else np.zeros(0, int)
                k_far = far_sub[np.argsort(-sc[far_sub])[:2]] if len(far_sub) else np.zeros(0, int)
                keep = np.concatenate([k_near, k_far])
                if len(keep) < N_SLOTS:
                    rem = [i for i in np.argsort(-sc) if i not in keep]
                    keep = np.concatenate([keep, rem[:N_SLOTS - len(keep)]])
                idx, pos = idx[keep], pos[keep]
        # retire stale tracklets
        still = []
        for t in active:
            (still if f - t.end <= max_gap else done).append(t)
        active = still
        assigned = set()
        if active and len(pos):
            cost = np.full((len(active), len(pos)), 1e6)
            for i, t in enumerate(active):
                gap = f - t.end
                gate = 0.9 + 0.3 * gap
                d = np.linalg.norm(pos - t.pos[-1], axis=1)
                cost[i] = np.where(d < gate, d, 1e6)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] < 1e5:
                    t = active[i]
                    t.frames.append(f); t.det_idx.append(int(idx[j])); t.pos.append(pos[j])
                    assigned.add(j)
        for j in range(len(pos)):
            if j not in assigned:
                t = Tracklet(next_id); next_id += 1
                t.frames.append(f); t.det_idx.append(int(idx[j])); t.pos.append(pos[j])
                active.append(t)
        # break tracklets of players that come too close (identity ambiguity)
        if len(active) > 1:
            cur = [t for t in active if t.end == f]
            for a in range(len(cur)):
                for b in range(a + 1, len(cur)):
                    if np.linalg.norm(cur[a].pos[-1] - cur[b].pos[-1]) < 0.7:
                        for t in (cur[a], cur[b]):
                            if t in active:
                                active.remove(t); done.append(t)
    done.extend(active)
    return [t for t in done if len(t) >= 3]


def _hsv_hist(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or img.size == 0 or img.shape[0] < 3 or img.shape[1] < 3:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ignore the blue court floor and very dark pixels
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (~((h > 95) & (h < 130) & (s > 120))) & (v > 35)
    mask = mask.astype(np.uint8)
    if mask.sum() < 6:
        return None
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / (hist.sum() + 1e-9)


def _crop(img: np.ndarray, x0, y0, x1, y1) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    x0, x1 = int(max(0, min(x0, x1))), int(min(W, max(x0, x1)))
    y0, y1 = int(max(0, min(y0, y1))), int(min(H, max(y0, y1)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return img[y0:y1, x0:x1]


def compute_appearance(tracklets: List[Tracklet], raw: RawDetections, calib: CourtCalibration,
                       video: Optional[Path], samples_per_tracklet: int = 8,
                       progress: Optional[Callable[[float, str], None]] = None) -> bool:
    if video is None or not Path(video).exists():
        return False
    need: Dict[int, List[tuple]] = {}
    for t in tracklets:
        n = len(t)
        k = min(samples_per_tracklet, max(1, n // 4))
        for i in np.linspace(0, n - 1, k).astype(int):
            need.setdefault(t.frames[i], []).append((t, t.det_idx[i]))
    torso_h: Dict[int, list] = {}
    legs_h: Dict[int, list] = {}
    colors: Dict[int, list] = {}
    total = len(need)
    for n_done, (f, img) in enumerate(iter_frames(video, need.keys())):
        if progress and n_done % 200 == 0:
            progress(n_done / max(total, 1), "appearance")
        for t, di in need[f]:
            kp = raw.persons_kp[f][di]
            bb = raw.persons_bbox[f][di]
            sh = kp[[C.KP["left_shoulder"], C.KP["right_shoulder"]]]
            hp = kp[[C.KP["left_hip"], C.KP["right_hip"]]]
            kn = kp[[C.KP["left_knee"], C.KP["right_knee"]]]
            if (sh[:, 2] > 0.3).all() and (hp[:, 2] > 0.3).all():
                xs = np.r_[sh[:, 0], hp[:, 0]]
                pad = 0.15 * (xs.max() - xs.min() + 4)
                torso = _crop(img, xs.min() - pad, sh[:, 1].min(), xs.max() + pad, hp[:, 1].max())
                legs = _crop(img, xs.min() - pad, hp[:, 1].max(), xs.max() + pad,
                             kn[:, 1].max() if (kn[:, 2] > 0.3).all() else hp[:, 1].max() + bb[3] * 0.2)
            else:
                x, y, w, h = bb
                torso = _crop(img, x + 0.2 * w, y + 0.2 * h, x + 0.8 * w, y + 0.5 * h)
                legs = _crop(img, x + 0.2 * w, y + 0.5 * h, x + 0.8 * w, y + 0.7 * h)
            th, lh = _hsv_hist(torso), _hsv_hist(legs)
            if th is not None:
                torso_h.setdefault(t.tid, []).append(th)
                colors.setdefault(t.tid, []).append(torso.reshape(-1, 3).mean(0))
            if lh is not None:
                legs_h.setdefault(t.tid, []).append(lh)
    for t in tracklets:
        if t.tid in torso_h:
            t.hist = np.mean(torso_h[t.tid], 0)
            t.color_bgr = np.median(colors[t.tid], 0)
        if t.tid in legs_h:
            t.legs_hist = np.mean(legs_h[t.tid], 0)
    return True


def _estimate_heights(tracklets: List[Tracklet], raw: RawDetections, calib: CourtCalibration) -> None:
    for t in tracklets:
        hs = []
        for f, di in list(zip(t.frames, t.det_idx))[:: max(1, len(t) // 15)]:
            kp = raw.persons_kp[f][di]
            nose, la, ra = kp[0], kp[C.KP["left_ankle"]], kp[C.KP["right_ankle"]]
            if nose[2] > 0.3 and la[2] > 0.3 and ra[2] > 0.3:
                foot = np.array([[(la[0] + ra[0]) / 2, max(la[1], ra[1])]])
                ppm = calib.pixel_scale_at(foot)[0]
                if ppm > 1:
                    hs.append((foot[0, 1] - nose[1]) / ppm * 1.07)  # nose->top of head
        if hs:
            t.height_m = float(np.median(hs))


def _hist_dist(a, b) -> float:
    if a is None or b is None:
        return 0.5
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def assign_teams(tracklets: List[Tracklet], n_frames: int, fps: float) -> Optional[np.ndarray]:
    """Team 0 / 1 per tracklet (majority) using kit colour + spatial consistency.
    Returns the per-frame indicator near_team0 (+1: team 0 on the near half,
    -1: on the far half, 0: unknown) which decides the team of every
    detection frame by frame (partners always share a half)."""
    sides = {t.tid: np.array([1 if p[1] >= C.NET_Y else -1 for p in t.pos]) for t in tracklets}
    with_hist = [t for t in tracklets if t.hist is not None]
    # Per-frame near-side vote.
    vote = np.zeros(n_frames)
    weight = np.zeros(n_frames)
    if len(with_hist) >= 4:
        from sklearn.cluster import KMeans
        X = np.sqrt(np.stack([t.hist for t in with_hist]))  # Hellinger embedding
        w = np.array([len(t) for t in with_hist], float)
        km = KMeans(2, n_init=10, random_state=0).fit(X, sample_weight=w)
        d = km.transform(X)
        for t, lab, dd in zip(with_hist, km.labels_, d):
            conf = abs(dd[0] - dd[1]) / (dd.sum() + 1e-9)
            s = sides[t.tid]
            sign = 1.0 if lab == 0 else -1.0
            fr = np.array(t.frames)
            # +1 means "team 0 on near side"
            np.add.at(vote, fr, sign * s * conf * 1.0)
            np.add.at(weight, fr, conf)
    has = weight > 0
    if has.sum() == 0:
        # No appearance: team purely by half at first sight (no changeover handling).
        for t in tracklets:
            t.team = 0 if np.median(sides[t.tid]) > 0 else 1
        return None
    # Smooth the near-team indicator: piecewise constant, changes only at changeovers.
    idx = np.where(has)[0]
    sig = np.sign(vote[idx])
    win = int(20 * fps)
    smooth = np.empty_like(sig)
    cs = np.cumsum(np.r_[0, sig])
    for i in range(len(idx)):
        lo = np.searchsorted(idx, idx[i] - win)
        hi = np.searchsorted(idx, idx[i] + win, side="right")
        smooth[i] = np.sign(cs[hi] - cs[lo]) or sig[i]
    smooth = _min_segment(smooth, idx, int(60 * fps))
    smooth = _refine_changeovers(smooth, idx, vote[idx], int(90 * fps))
    near_team0 = np.zeros(n_frames)
    near_team0[idx] = smooth
    # fill gaps by nearest known value
    known = idx
    for t in tracklets:
        fr = np.array(t.frames)
        nearest = known[np.clip(np.searchsorted(known, fr), 0, len(known) - 1)]
        v = near_team0[nearest] * sides[t.tid]  # +1 -> team 0
        t.team = 0 if v.sum() >= 0 else 1
    # fill unknown frames with the nearest known label
    full = near_team0.copy()
    unk = full == 0
    if unk.any():
        full[unk] = near_team0[known[np.clip(np.searchsorted(known, np.where(unk)[0]), 0, len(known) - 1)]]
    return full


def _refine_changeovers(labels: np.ndarray, frames: np.ndarray, votes: np.ndarray, radius: int) -> np.ndarray:
    """Move every side-change boundary to the frame that maximises agreement
    with the per-frame appearance votes inside +-radius (change-point fit)."""
    lab = labels.copy()
    cuts = np.where(np.diff(lab) != 0)[0] + 1
    for c in cuts:
        before, after = lab[c - 1], lab[c]
        lo = np.searchsorted(frames, frames[c] - radius)
        hi = np.searchsorted(frames, frames[c] + radius, side="right")
        seg = votes[lo:hi]
        # score(b) = sum_{i<b} seg*before + sum_{i>=b} seg*after
        cs = np.r_[0, np.cumsum(seg)]
        score = cs * before + (cs[-1] - cs) * after
        b = lo + int(np.argmax(score))
        lab[lo:b] = before
        lab[b:hi] = after
    return lab


def _min_segment(labels: np.ndarray, frames: np.ndarray, min_len: int) -> np.ndarray:
    """Teams change ends only at changeovers (>= several minutes apart): absorb
    side segments shorter than min_len frames into the surrounding segment."""
    lab = labels.copy()
    for _ in range(10):
        cuts = np.where(np.diff(lab) != 0)[0] + 1
        bounds = np.r_[0, cuts, len(lab)]
        segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        short = [(a, b) for a, b in segs if frames[b - 1] - frames[a] < min_len]
        if not short or len(segs) == 1:
            break
        a, b = min(short, key=lambda ab: frames[ab[1] - 1] - frames[ab[0]])
        lab[a:b] = lab[a - 1] if a > 0 else lab[b]
    return lab


def assign_roles(tracklets: List[Tracklet]) -> None:
    """Split each team into left (0) and right (1) player."""
    for team in (0, 1):
        ts = sorted([t for t in tracklets if t.team == team], key=len, reverse=True)
        if not ts:
            continue

        def lat(t):  # + = player's left
            p = np.array(t.pos)
            near = p[:, 1] >= C.NET_Y
            return float(np.mean(np.where(near, C.CENTER_X - p[:, 0], p[:, 0] - C.CENTER_X)))

        def overlap(a, b):
            return max(0, min(a.end, b.end) - max(a.start, b.start))

        lats = {t.tid: lat(t) for t in ts}
        protos = {0: [], 1: []}
        labeled: List[Tracklet] = []
        # seed with the most overlapping long pair
        seed = ts[0]
        partners = [t for t in ts[1:] if overlap(seed, t) > 0]
        if partners:
            p = max(partners, key=lambda t: overlap(seed, t))
            a, b = (seed, p) if lats[seed.tid] >= lats[p.tid] else (p, seed)
            a.role, b.role = 0, 1
            labeled += [a, b]
        else:
            seed.role = 0 if lats[seed.tid] >= 0 else 1
            labeled.append(seed)
        for t in labeled:
            protos[t.role].append(t)

        def dist(t, role):
            ps = protos[role]
            if not ps:
                return 1.0 if (lats[t.tid] >= 0) == (role == 0) else 2.0
            w = np.array([len(p) for p in ps], float)
            plat = np.average([lats[p.tid] for p in ps], weights=w)
            hs = [(p.height_m, len(p)) for p in ps if np.isfinite(p.height_m)]
            d = abs(lats[t.tid] - plat) / 1.5
            if hs and np.isfinite(t.height_m):
                ph = np.average([h for h, _ in hs], weights=[n for _, n in hs])
                d += abs(t.height_m - ph) / 0.08
            d += 2.0 * np.mean([_hist_dist(t.legs_hist, p.legs_hist) for p in ps[:5]])
            return d

        for t in ts:
            if t.role >= 0:
                continue
            forced = None
            for o in labeled:
                if overlap(t, o) > 5:
                    forced = 1 - o.role
                    break
            t.role = forced if forced is not None else (0 if dist(t, 0) <= dist(t, 1) else 1)
            labeled.append(t)
            protos[t.role].append(t)


def resolve_identities(raw: RawDetections, calib: CourtCalibration, video: Optional[Path] = None,
                       progress: Optional[Callable[[float, str], None]] = None) -> Tracks:
    fps = raw.fps
    if progress:
        progress(0.0, "tracking players")
    tracklets = build_tracklets(raw, calib, fps)
    if progress:
        progress(0.2, "reading kit colours")
    have_app = compute_appearance(tracklets, raw, calib, video,
                                  progress=(lambda p, m: progress(0.2 + 0.6 * p, m)) if progress else None)
    _estimate_heights(tracklets, raw, calib)
    near_team0 = assign_teams(tracklets, raw.n_frames, fps)
    # Team A = team on near side at the start.
    first = sorted(tracklets, key=lambda t: t.start)[:4]
    near_first = [t.team for t in first if t.pos[0][1] >= C.NET_Y]
    team_a = near_first[0] if near_first else 0
    for t in tracklets:
        t.team = 0 if t.team == team_a else 1
    if near_team0 is not None and team_a == 1:
        near_team0 = -near_team0
    assign_roles(tracklets)

    N = raw.n_frames
    kp = np.full((N, N_SLOTS, 17, 3), np.nan, np.float32)
    bbox = np.full((N, N_SLOTS, 4), np.nan, np.float32)
    feet = np.full((N, N_SLOTS, 2), np.nan, np.float32)
    owner_len = np.zeros((N, N_SLOTS), np.int32)
    for t in tracklets:
        for f, di, p in zip(t.frames, t.det_idx, t.pos):
            team = t.team
            if near_team0 is not None and near_team0[f] != 0:
                near = p[1] >= C.NET_Y
                team = 0 if (near_team0[f] > 0) == near else 1   # team follows the half
            s = team * 2 + t.role
            if owner_len[f, s] >= len(t):
                continue
            kp[f, s] = raw.persons_kp[f][di]
            bbox[f, s] = raw.persons_bbox[f][di]
            feet[f, s] = p
            owner_len[f, s] = len(t)

    meta = []
    for s in range(N_SLOTS):
        team, role = divmod(s, 2)
        ts = [t for t in tracklets if t.team == team and t.role == role]
        n = sum(len(t) for t in ts)
        cols = [t.color_bgr for t in ts if t.color_bgr is not None]
        hs = [t.height_m for t in ts if np.isfinite(t.height_m)]
        n = int(np.isfinite(feet[:, s, 0]).sum())
        meta.append({
            "slot": s, "team": "AB"[team], "role": "left" if role == 0 else "right",
            "name": f"{'AB'[team]}{role + 1} ({'left' if role == 0 else 'right'})",
            "frames_tracked": int(n), "tracklets": len(ts),
            "kit_color_bgr": [int(v) for v in np.median(cols, 0)] if cols else None,
            "height_rel": float(np.median(hs)) if hs else None,
        })
    # The camera looks down on the court, so vertical image distances are
    # compressed relative to the floor scale. Calibrate with an average adult
    # player height of 1.78 m (used to convert contact heights to metres).
    rel = [m["height_rel"] for m in meta if m["height_rel"]]
    vfac = 1.78 / float(np.median(rel)) if rel else 1.0
    for m in meta:
        m["height_m"] = round(m.pop("height_rel") * vfac, 2) if m.get("height_rel") else None
    tr = Tracks(fps, N, raw.width, raw.height, kp, bbox, feet, raw.ball.copy(),
                ["A", "A", "B", "B"], meta, raw.source, raw.frame_range,
                {"appearance_used": bool(have_app), "n_tracklets": len(tracklets), "vertical_factor": round(vfac, 3)})
    return tr
