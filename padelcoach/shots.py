"""Shot construction & classification.

Every detected hit becomes a Shot with:
  stroke   forehand / backhand / overhead / serve
  context  serve / volley / after_bounce / off_wall
  shot_type refined padel taxonomy (drive, volley, wall, lob, smash, bandeja,
           vibora, dropshot, chiquita, serve, return)
  base     PadelTracker100 label (Serve/Forehand/Backhand/Smash/Dropshot/Other)
plus contact geometry, landing point, flight time, speed estimate and
biomechanics. An optional Colab-trained ONNX sequence model refines the base
stroke; rules decide the padel-specific subtype.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .court import CourtCalibration, lateral_of, zone_of
from .events import BallEvent
from .shotfeatures import WINDOW, sequence_features, window_indices
from .tracks import N_SLOTS, Tracks


@dataclass
class Shot:
    idx: int
    frame: int
    time_s: float
    player: int
    team: str
    side: str
    stroke: str = "forehand"
    context: str = "after_bounce"
    shot_type: str = "other"
    base: str = "Other"
    confidence: float = 0.5
    court_pos: List[float] = field(default_factory=list)
    zone: str = "defense"
    court_half: str = "left"
    dist_from_net_m: float = 0.0
    contact: Dict[str, float] = field(default_factory=dict)
    landing: Optional[List[float]] = None
    landing_zone: Optional[str] = None
    flight_time_s: Optional[float] = None
    speed_kmh: Optional[float] = None
    biomech: Dict[str, float] = field(default_factory=dict)
    model_probs: Optional[Dict[str, float]] = None
    hit_confidence: float = 0.5
    # filled by the rally stage
    point: Optional[int] = None
    num_in_point: Optional[int] = None
    outcome: str = "in_play"   # in_play | winner | error | forced_error_induced | fault
    pressure_in: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = None if not np.isfinite(v) else round(v, 3)
        return d


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1))))


def _pose_at(tracks: Tracks, f: int, s: int):
    for d in (0, -1, 1, -2, 2, -3, 3):
        g = f + d
        if 0 <= g < tracks.n_frames and np.isfinite(tracks.kp[g, s, 0, 0]):
            return g, tracks.kp[g, s], tracks.bbox[g, s]
    return None, None, None


def estimate_handedness(tracks: Tracks, hits: List[BallEvent]) -> Dict[int, str]:
    votes = {s: [0, 0] for s in range(N_SLOTS)}
    for e in hits:
        g, kp, _ = _pose_at(tracks, e.frame, e.player)
        if kp is None:
            continue
        p = np.array(e.px)
        lw, rw = kp[C.KP["left_wrist"]], kp[C.KP["right_wrist"]]
        if lw[2] > 0.3 and rw[2] > 0.3:
            dl, dr = np.hypot(*(lw[:2] - p)), np.hypot(*(rw[:2] - p))
            if abs(dl - dr) > 6:
                votes[e.player][0 if dr < dl else 1] += 1
    return {s: ("left" if v[1] > 1.25 * v[0] and v[1] >= 5 else "right") for s, v in votes.items()}


class ShotModel:
    """Optional ONNX temporal pose classifier trained on Colab."""

    def __init__(self):
        self.sess = None
        self.classes = list(C.BASE_STROKES)
        if C.SHOT_MODEL.exists():
            try:
                import onnxruntime as ort
                self.sess = ort.InferenceSession(str(C.SHOT_MODEL), providers=["CPUExecutionProvider"])
                if C.SHOT_MODEL_META.exists():
                    self.classes = json.loads(C.SHOT_MODEL_META.read_text()).get("classes", self.classes)
            except Exception as ex:  # pragma: no cover
                print("[shots] ONNX shot model unavailable:", ex)
                self.sess = None

    @property
    def available(self) -> bool:
        return self.sess is not None

    def predict(self, X: np.ndarray) -> np.ndarray:
        name = self.sess.get_inputs()[0].name
        logits = self.sess.run(None, {name: X.astype(np.float32)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def serve_score(tracks: Tracks, f: int, s: int, side: str, dist_net: float, overhead: bool,
                touches: List[BallEvent], time_since_prev: float) -> float:
    """Likelihood (0..1) that the hit at frame f by slot s is a serve, from the
    serve formation (FIP Rule 6): server behind the service line and still, the
    ball bounced by the server just before, receiver deep on the diagonal,
    server's partner usually at the net, and a pause in play beforehand."""
    if overhead or dist_net < 5.6:
        return 0.0
    fps = tracks.fps
    sc = 0.25 if dist_net >= 6.5 else 0.1
    team = tracks.team_of_slot[s]
    me = tracks.feet[f, s]
    partner = [q for q in range(N_SLOTS) if q != s and tracks.team_of_slot[q] == team]
    opps = [q for q in range(N_SLOTS) if tracks.team_of_slot[q] != team]
    for q in partner:
        p = tracks.feet[f, q]
        if np.isfinite(p[1]) and abs(p[1] - C.NET_Y) < 4.5:
            sc += 0.1
    deep_diag = False
    for q in opps:
        p = tracks.feet[f, q]
        if np.isfinite(p[1]) and abs(p[1] - C.NET_Y) > 5.5 and (p[0] - C.CENTER_X) * (me[0] - C.CENTER_X) < 0:
            deep_diag = True
    sc += 0.15 if deep_diag else -0.1
    a = max(0, f - int(1.0 * fps))
    moved = tracks.feet[a:f + 1, s]
    moved = moved[np.isfinite(moved[:, 0])]
    if len(moved) > 3 and np.hypot(*(moved[-1] - moved[0])) < 0.9:
        sc += 0.15
    if any(t.player == s and f - int(3.5 * fps) <= t.frame < f for t in touches):
        sc += 0.2   # bounced the ball before serving
    if time_since_prev > 3.0:
        sc += 0.2
    elif time_since_prev > 1.4:
        sc += 0.05
    else:
        sc -= 0.3   # in the middle of a fast exchange
    return float(np.clip(sc, 0, 1))


def build_shots(tracks: Tracks, calib: CourtCalibration, events: List[BallEvent],
                model: Optional[ShotModel] = None) -> List[Shot]:
    fps = tracks.fps
    vf = float(tracks.info.get("vertical_factor", 1.0)) if tracks.info else 1.0
    hits = [e for e in events if e.kind == "hit" and e.player is not None]
    hand = estimate_handedness(tracks, hits)
    others = [e for e in events if e.kind not in ("hit", "touch")]
    touches = [e for e in events if e.kind == "touch"]
    shots: List[Shot] = []
    model_inputs, model_rows = [], []

    for i, e in enumerate(hits):
        s = e.player
        f = e.frame
        team = tracks.team_of_slot[s]
        g, kp, bb = _pose_at(tracks, f, s)
        feet = tracks.feet[g if g is not None else f, s]
        if not np.isfinite(feet[0]):
            continue
        side = "near" if feet[1] >= C.NET_Y else "far"
        prev = hits[i - 1] if i > 0 else None
        nxt = hits[i + 1] if i + 1 < len(hits) else None
        sh = Shot(len(shots), int(f), round(f / fps, 3), int(s), team, side,
                  court_pos=[round(float(feet[0]), 2), round(float(feet[1]), 2)],
                  zone=zone_of(float(feet[1])), court_half=lateral_of(float(feet[0]), side),
                  dist_from_net_m=round(float(abs(feet[1] - C.NET_Y)), 2), hit_confidence=e.confidence)

        # ---------------- contact geometry
        foot_px = calib.to_image(feet[None])[0]
        ppm = float(calib.pixel_scale_at(foot_px[None])[0])
        ball = np.array(e.px, float)
        overhead = False
        lat_m = 0.0
        if kp is not None:
            vis = kp[:, 2] > 0.25
            hips = [j for j in (11, 12) if vis[j]]
            shs = [j for j in (5, 6) if vis[j]]
            hip = kp[hips, :2].mean(0) if hips else np.array([bb[0] + bb[2] / 2, bb[1] + 0.55 * bb[3]])
            shp = kp[shs, :2].mean(0) if shs else hip - np.array([0, 0.3 * bb[3]])
            torso = max(np.linalg.norm(shp - hip), 6.0)
            head_y = kp[0, 1] if vis[0] else shp[1] - 0.45 * torso
            dom_w = C.KP["right_wrist"] if hand[s] == "right" else C.KP["left_wrist"]
            wrist = kp[dom_w, :2] if vis[dom_w] else ball
            contact_y = min(ball[1], wrist[1]) if np.isfinite(ball[1]) else wrist[1]
            overhead = bool(contact_y < head_y - 0.15 * torso)
            sgn = 1.0 if side == "near" else -1.0
            lat_px = ((ball[0] if np.isfinite(ball[0]) else wrist[0]) - hip[0]) * sgn
            lat_m = lat_px / max(ppm, 1.0)
            dom_sign = 1.0 if hand[s] == "right" else -1.0
            on_dom = lat_px * dom_sign > 0 if abs(lat_px) > 0.1 * torso else (wrist[0] - hip[0]) * sgn * dom_sign > 0
            height_m = vf * (foot_px[1] - (ball[1] if np.isfinite(ball[1]) else wrist[1])) / max(ppm, 1.0)
            # biomechanics at contact (2D image angles)
            knees = [_angle(kp[a, :2], kp[b, :2], kp[c, :2]) for a, b, c in ((11, 13, 15), (12, 14, 16))
                     if vis[a] and vis[b] and vis[c]]
            arm = (6, 8, 10) if hand[s] == "right" else (5, 7, 9)
            elbow = _angle(kp[arm[0], :2], kp[arm[1], :2], kp[arm[2], :2]) if vis[list(arm)].all() else float("nan")
            stance = abs(kp[15, 0] - kp[16, 0]) / max(ppm, 1.0) if vis[15] and vis[16] else float("nan")
            sh.contact = {"height_m": round(float(height_m), 2), "lateral_m": round(float(lat_m), 2),
                          "on_dominant_side": bool(on_dom), "above_head": overhead,
                          "hand": hand[s]}
            sh.biomech = {"knee_flexion_deg": round(float(np.nanmin(knees)), 1) if knees else None,
                          "elbow_deg": None if not np.isfinite(elbow) else round(elbow, 1),
                          "stance_width_m": None if not np.isfinite(stance) else round(float(stance), 2)}
        else:
            on_dom = True
            sh.contact = {"hand": hand[s], "above_head": False}

        # ---------------- context: what happened on the hitter's side before contact
        lo = prev.frame if prev is not None else f - int(2.5 * fps)
        before = [o for o in others if lo < o.frame < f]
        bounced = [o for o in before if o.kind == "bounce" and o.side == side]
        walled = [o for o in before if o.kind == "wall" and (o.side == side or (o.wall or "").startswith(side))]
        time_since_prev = (f - prev.frame) / fps if prev is not None else 99.0
        sscore = serve_score(tracks, f, s, side, sh.dist_from_net_m, overhead, touches, time_since_prev)
        sh.contact["serve_score"] = round(sscore, 2)
        if sscore >= 0.6:
            context = "serve"
        elif bounced and walled and walled[-1].frame > bounced[0].frame:
            context = "off_wall"
        elif bounced:
            context = "after_bounce"
        elif prev is not None and time_since_prev < 3.0:
            context = "volley" if sh.zone != "defense" else "after_bounce"  # deep volleys are rare
        else:
            context = "after_bounce"
        sh.context = context

        # ---------------- outgoing ball: landing & speed
        hi = nxt.frame if nxt is not None else f + int(3 * fps)
        after = [o for o in others if f < o.frame <= hi]
        opp = "far" if side == "near" else "near"
        land = next((o for o in after if o.kind == "bounce" and o.side == opp), None)
        if land is not None:
            sh.landing = [round(land.court[0], 2), round(land.court[1], 2)]
            sh.landing_zone = zone_of(land.court[1])
            sh.flight_time_s = round((land.frame - f) / fps, 3)
            dist = float(np.hypot(land.court[0] - feet[0], land.court[1] - feet[1]))
            sh.speed_kmh = round(dist / max(sh.flight_time_s, 1e-3) * 3.6, 1)
        elif nxt is not None and tracks.team_of_slot[nxt.player] != team:
            g2 = nxt.frame
            f2 = tracks.feet[g2, nxt.player]
            if np.isfinite(f2[0]):
                dt = (g2 - f) / fps
                sh.flight_time_s = round(dt, 3)
                sh.speed_kmh = round(float(np.hypot(*(f2 - feet))) / max(dt, 1e-3) * 3.6, 1)
        flight_to_next = ((min(land.frame, nxt.frame) if (land and nxt) else (land.frame if land else (nxt.frame if nxt else None))))
        flight_to_next = (flight_to_next - f) / fps if flight_to_next is not None else None
        # height gained after contact (image rise in metres, rough)
        b = tracks.ball
        seg = b[f:min(len(b), f + int(0.7 * fps)), 1]
        rise_m = vf * float((ball[1] - np.nanmin(seg)) / max(ppm, 1.0)) if np.isfinite(seg).any() and np.isfinite(ball[1]) else 0.0

        # ---------------- classification rules
        stroke = "overhead" if overhead else ("forehand" if on_dom else "backhand")
        if context == "serve":
            stroke, stype = "serve", "serve"
        elif overhead:
            fast = (sh.speed_kmh or 0) > 70 or (sh.flight_time_s is not None and sh.flight_time_s < 0.5)
            if fast:
                stype = "smash"
            elif abs(lat_m) > 0.45 and on_dom:
                stype = "vibora"
            else:
                stype = "bandeja"
        else:
            fh = stroke == "forehand"
            land_d = abs(sh.landing[1] - C.NET_Y) if sh.landing else None
            slow = (sh.speed_kmh or 99) < 38
            if flight_to_next is not None and flight_to_next > 1.15 and rise_m > 1.2:
                stype = "lob"
            elif land_d is not None and land_d < 2.5 and slow and sh.zone in ("net", "transition"):
                stype = "dropshot"
            elif land_d is not None and land_d < 4.0 and slow and sh.dist_from_net_m > 5.5:
                stype = "chiquita"
            elif context == "volley":
                stype = "forehand_volley" if fh else "backhand_volley"
            elif context == "off_wall":
                stype = "forehand_wall" if fh else "backhand_wall"
            else:
                stype = "forehand_drive" if fh else "backhand_drive"
        sh.stroke, sh.shot_type = stroke, stype
        sh.base = ("Serve" if stype == "serve" else "Smash" if stroke == "overhead" else
                   "Dropshot" if stype == "dropshot" else "Forehand" if stroke == "forehand" else "Backhand")
        sh.confidence = round(float(0.35 + 0.5 * e.confidence), 2)

        if model is not None and model.available and g is not None:
            win = window_indices(f, tracks.n_frames)
            X = sequence_features(tracks.kp[win, s], tracks.ball[win], side == "far", hand[s] == "left")
            model_inputs.append(X)
            model_rows.append(len(shots))
        shots.append(sh)

    if model_inputs:
        probs = model.predict(np.stack(model_inputs))
        for row, p in zip(model_rows, probs):
            sh = shots[row]
            sh.model_probs = {c: round(float(v), 3) for c, v in zip(model.classes, p)}
            top = model.classes[int(np.argmax(p))]
            if p.max() > 0.55 and sh.context != "serve" and top != "Other":
                sh.base = top
                if top == "Smash" and sh.stroke != "overhead":
                    sh.stroke, sh.shot_type = "overhead", "bandeja"
                elif top in ("Forehand", "Backhand") and sh.stroke in ("forehand", "backhand"):
                    want = top.lower()
                    if want != sh.stroke:
                        sh.stroke = want
                        sh.shot_type = sh.shot_type.replace("backhand" if want == "forehand" else "forehand", want)
            sh.confidence = round(float(0.5 * sh.confidence + 0.5 * p.max()), 2)
    return shots
