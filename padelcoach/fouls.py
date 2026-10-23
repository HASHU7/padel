"""FIP rule-infraction checks (FIP Rules of Padel, review applicable 01.01.2026).

Every check reports evidence and a confidence. Measurement error (pose
keypoints, 2D ball, floor homography) is taken into account with explicit
margins so that only clear infractions reach "likely". Status:
  likely  (conf >= 0.70)  - shown as infraction
  possible(conf >= 0.55)  - shown, flagged for video review
  review  (below)         - kept in data, hidden by default
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .court import CourtCalibration
from .events import BallEvent
from .rally import Point
from .shots import Shot
from .tracks import N_SLOTS, Tracks

RULES = {
    "SERVE_FOOT_FAULT": {
        "name": "Service foot fault", "rule": "FIP Rule 6.1 / 6.3", "severity": "service_fault",
        "text": "At the moment of service the server must keep at least one foot behind the service line and must not touch the service line or the central line.",
        "correction": "Set up 20-30 cm behind the line and keep your front foot planted through contact; do not step in before the strike."},
    "SERVE_ABOVE_WAIST": {
        "name": "Serve struck above the waist", "rule": "FIP Rule 6.4", "severity": "service_fault",
        "text": "The ball must be struck at or below waist height at the moment of impact.",
        "correction": "Let the ball drop lower after the bounce and contact it beside the front hip with a compact, slightly brushed swing."},
    "SERVE_NO_BOUNCE": {
        "name": "No bounce before service", "rule": "FIP Rule 6.2", "severity": "service_fault",
        "text": "The server must bounce the ball on the ground behind the service line, within their service area, before striking it.",
        "correction": "Always bounce the ball once inside your half, behind the line, then strike."},
    "SERVE_OUT": {
        "name": "Serve out of the service box", "rule": "FIP Rule 7", "severity": "service_fault",
        "text": "The served ball must bounce in the receiver's service box diagonally opposite (lines are good).",
        "correction": "Aim 50-80 cm inside the box (T or glass corner) and serve with more slice for control rather than pace."},
    "SERVE_MESH": {
        "name": "Serve hit the metal mesh after the bounce", "rule": "FIP Rule 7", "severity": "service_fault",
        "text": "A serve that bounces in the box and then touches the metal fence before its second bounce is a fault.",
        "correction": "Target the T or the back glass; avoid wide angles that run into the side fence."},
    "DOUBLE_FAULT": {
        "name": "Double fault", "rule": "FIP Rule 13 (q)", "severity": "point_lost",
        "text": "Two consecutive service faults lose the point.",
        "correction": "On second serve reduce pace and add margin: a safe, deep slice serve to the body or the T."},
    "DIRECT_WALL": {
        "name": "Ball hit the opponents' wall before bouncing", "rule": "FIP Rule 13 (g)", "severity": "point_lost",
        "text": "The point is lost if the ball, after being returned, hits the opponents' walls or fence before bouncing in their court.",
        "correction": "Take pace off and add topspin/slice so the ball bounces before the glass; lob higher and earlier."},
    "DOUBLE_BOUNCE": {
        "name": "Double bounce", "rule": "FIP Rule 13 (c)", "severity": "point_lost",
        "text": "The ball must be returned before it bounces twice on the player's court.",
        "correction": "Split-step as the opponent strikes and move earlier to low balls; use the glass only when the ball will rebound."},
    "NET_INVASION": {
        "name": "Net touch / invasion of the opponents' court", "rule": "FIP Rule 13 (a)", "severity": "point_lost",
        "text": "The point is lost if a player touches the net/posts or enters the opponents' court while the ball is in play.",
        "correction": "Stop 70-100 cm from the net after a put-away; brake with the outside foot instead of running through the shot."},
}


def _status(conf: float) -> str:
    return "likely" if conf >= 0.70 else ("possible" if conf >= C.FOUL_MIN_CONFIDENCE_REPORT else "review")


def _foul(kind: str, shot: Optional[Shot], frame: int, player: Optional[int], team: Optional[str], conf: float,
          evidence: Dict, explanation: str, fps: float, point: Optional[int] = None) -> dict:
    r = RULES[kind]
    return {"type": kind, "name": r["name"], "rule": r["rule"], "rule_text": r["text"], "severity": r["severity"],
            "frame": int(frame), "time_s": round(frame / fps, 2), "player": player, "team": team,
            "point": shot.point if shot is not None else point, "shot": shot.idx if shot is not None else None,
            "confidence": round(float(conf), 2), "status": _status(conf), "evidence": evidence,
            "explanation": explanation, "correction": r["correction"]}


def _serve_checks(sh: Shot, tracks: Tracks, calib: CourtCalibration, events: List[BallEvent],
                  next_hit: Optional[Shot]) -> List[dict]:
    out = []
    fps = tracks.fps
    f, s = sh.frame, sh.player
    # ---- feet position (ankles projected to the floor)
    margins = []
    for g in range(max(0, f - 3), min(tracks.n_frames, f + 1)):
        kp = tracks.kp[g, s]
        if not np.isfinite(kp[0, 0]):
            continue
        feet = []
        for a in (C.KP["left_ankle"], C.KP["right_ankle"]):
            if kp[a, 2] > 0.4:
                feet.append(calib.to_court(kp[a, :2][None] + np.array([[0, 0.02 * tracks.bbox[g, s, 3]]]))[0])
        if feet:
            d = [abs(p[1] - C.NET_Y) for p in feet]
            margins.append(max(d) - C.SERVICE_LINE_FROM_NET)  # >0: rear foot behind the line
    if len(margins) >= 2:
        worst = float(np.median(margins))
        if worst < -0.30:   # both feet clearly inside the court (> 30 cm past the line)
            conf = min(0.9, 0.55 + (-worst - 0.30) * 0.8)
            out.append(_foul("SERVE_FOOT_FAULT", sh, f, s, sh.team, conf,
                             {"rear_foot_margin_m": round(worst, 2)},
                             f"At contact the server's rear foot was ~{-worst:.2f} m inside the service line.", fps))
    # ---- contact height vs waist
    kp = tracks.kp[f, s]
    # The ball rises quickly after contact: use the LOWEST detected ball position
    # in the 3 frames up to contact (lower bound on the contact height).
    win = tracks.ball[max(0, f - 2):f + 1]
    win = win[np.isfinite(win[:, 0]) & (win[:, 2] >= 1.0)]
    ball = win[np.argmax(win[:, 1])] if len(win) else np.full(3, np.nan)
    if np.isfinite(kp[0, 0]) and np.isfinite(ball[0]):
        hips = [kp[i] for i in (11, 12) if kp[i, 2] > 0.4]
        shs = [kp[i] for i in (5, 6) if kp[i, 2] > 0.4]
        if hips and shs:
            hip_y = np.mean([h[1] for h in hips])
            torso = abs(hip_y - np.mean([q[1] for q in shs]))
            if torso > 12:
                # Perspective: the serve is struck ~0.45 m in front of the body (towards the
                # net). For a near-side server that point is further from the camera and
                # appears higher in the image; for a far-side server lower. Shift the ball
                # back to body depth using the floor scale at the server's feet.
                feet = tracks.feet[f, s]
                ball_y = ball[1]
                if np.isfinite(feet[0]):
                    toward_net = -0.45 if feet[1] >= C.NET_Y else 0.45
                    a_px = calib.to_image(feet[None])[0]
                    b_px = calib.to_image((feet + np.array([0.0, toward_net]))[None])[0]
                    ball_y = ball[1] + (a_px[1] - b_px[1])
                above = (hip_y - ball_y) / torso  # fraction of torso above the hip joint
                # the waist sits ~0.25 torso above the hip joint; allow 0.25 more for
                # ball size, keypoint jitter and contact-frame uncertainty
                if above > 0.50:
                    conf = min(0.85, 0.5 + (above - 0.50) * 1.0)
                    out.append(_foul("SERVE_ABOVE_WAIST", sh, f, s, sh.team, conf,
                                     {"contact_above_hip_torso_frac": round(float(above), 2)},
                                     f"Ball contact was {above:.0%} of the torso length above the hip joint (waist ~25%).", fps))
    # ---- bounce before serving
    pre = [e for e in events if f - int(2.5 * fps) <= e.frame < f and e.kind in ("touch", "bounce", "hit")
           and (e.player == s or e.kind == "bounce")]
    if not pre and np.isfinite(tracks.ball[max(0, f - int(1.5 * fps)):f, 0]).sum() > 0.6 * 1.5 * fps:
        out.append(_foul("SERVE_NO_BOUNCE", sh, f, s, sh.team, 0.45, {},
                         "No ball bounce by the server was seen in the 2.5 s before the serve (ball was visible).", fps))
    # ---- landing in the diagonal box
    if sh.landing is not None:
        recv_side = "far" if sh.side == "near" else "near"
        server_left_img = sh.court_pos[0] < C.CENTER_X
        # diagonal: target box is on the opposite image half
        tx0, tx1 = (C.CENTER_X, C.COURT_W) if server_left_img else (0.0, C.CENTER_X)
        ty0, ty1 = ((C.FAR_SERVICE_Y, C.NET_Y) if recv_side == "far" else (C.NET_Y, C.NEAR_SERVICE_Y))
        lx, ly = sh.landing
        dx = max(tx0 - lx, 0, lx - tx1)
        dy = max(ty0 - ly, 0, ly - ty1)
        miss = float(np.hypot(dx, dy))
        if miss > 0.30:
            conf = min(0.85, 0.5 + (miss - 0.30) * 0.6)
            out.append(_foul("SERVE_OUT", sh, f, s, sh.team, conf,
                             {"landing": [round(lx, 2), round(ly, 2)], "distance_outside_box_m": round(miss, 2)},
                             f"Serve bounced {miss:.2f} m outside the diagonal service box.", fps))
        else:
            # mesh after bounce
            hi = next_hit.frame if next_hit is not None else f + int(3 * fps)
            land_f = f + int(round((sh.flight_time_s or 0) * fps))
            wall = next((e for e in events if land_f < e.frame < hi and e.kind == "wall"), None)
            if wall is not None and wall.wall == "side_mesh" and wall.side == recv_side:
                out.append(_foul("SERVE_MESH", sh, wall.frame, s, sh.team, 0.6,
                                 {"wall_point": [round(v, 2) for v in wall.court]},
                                 "After bouncing in the box the serve struck the side metal mesh.", fps))
    return out


def detect_fouls(tracks: Tracks, calib: CourtCalibration, shots: List[Shot], points: List[Point],
                 events: List[BallEvent]) -> List[dict]:
    fps = tracks.fps
    fouls: List[dict] = []
    by_idx = {s.idx: s for s in shots}
    for pt in points:
        pshots = [by_idx[i] for i in pt.shot_ids if i in by_idx]
        if not pshots:
            continue
        sv = pshots[0]
        if sv.context == "serve":
            fouls += _serve_checks(sv, tracks, calib, events, pshots[1] if len(pshots) > 1 else None)
        if pt.how == "double_fault":
            fouls.append(_foul("DOUBLE_FAULT", sv, sv.frame, sv.player, sv.team, 0.6, {"serve_faults": pt.serve_faults + 1},
                               "Second consecutive service fault.", fps))
        # ---- in-play checks on every shot of the rally
        for k, sh in enumerate(pshots):
            nxt = pshots[k + 1] if k + 1 < len(pshots) else None
            hi = nxt.frame if nxt else sh.frame + int(3 * fps)
            after = [e for e in events if sh.frame < e.frame < hi and e.kind in ("bounce", "wall")]
            opp_side = "far" if sh.side == "near" else "near"
            first = after[0] if after else None
            seg = tracks.ball[sh.frame:first.frame + 1, 0] if first is not None else np.zeros(0)
            visible = float(np.isfinite(seg).mean()) if len(seg) else 0.0
            # only when the ball was tracked all the way to the glass (otherwise a
            # missed floor bounce is the likelier explanation)
            if first is not None and first.kind == "wall" and first.side == opp_side and first.wall in ("far_back", "near_back", "side_mesh") \
                    and sh.context != "serve" and first.confidence >= 0.55 and nxt is None and visible >= 0.9:
                fouls.append(_foul("DIRECT_WALL", sh, first.frame, sh.player, sh.team, 0.58 + 0.04 * (first.wall != "near_back"),
                                   {"wall": first.wall, "wall_point": [round(v, 2) for v in first.court]},
                                   f"Ball struck the {first.wall.replace('_', ' ')} without bouncing in the opponents' court.", fps))
            if nxt is not None:
                own = [e for e in after if e.kind == "bounce" and e.side == nxt.side]

                def _interior(e):  # "bounces" next to the glass are often wall impacts
                    return 1.5 < e.court[1] < C.COURT_L - 1.5 and 0.8 < e.court[0] < C.COURT_W - 0.8
                if len(own) >= 2 and (own[1].frame - own[0].frame) > 0.25 * fps and own[0].confidence > 0.6                         and own[1].confidence > 0.6 and _interior(own[0]) and _interior(own[1]):
                    # second floor bounce on the receiver's side before their hit
                    d = abs(own[1].court[1] - own[0].court[1]) + abs(own[1].court[0] - own[0].court[0])
                    if d > 0.8:
                        fouls.append(_foul("DOUBLE_BOUNCE", nxt, own[1].frame, nxt.player, nxt.team, 0.58,
                                           {"bounces": [[round(v, 2) for v in own[0].court], [round(v, 2) for v in own[1].court]]},
                                           "Two floor bounces were seen on the receiver's side before the return.", fps))
        # ---- net invasion while the ball is in play (serve .. last shot + 0.3 s)
        a, b = pt.start_frame, pt.end_frame + int(0.3 * fps)
        for s in range(N_SLOTS):
            xy = tracks.feet[a:b, s]
            y = xy[:, 1]
            ok = np.isfinite(y)
            if ok.sum() < 10:
                continue
            side0 = np.nanmedian(y) >= C.NET_Y
            beyond = ((y < C.NET_Y - 0.5) if side0 else (y > C.NET_Y + 0.5)) & ok
            # continuous movement only (no identity jump), sustained >= 8 frames
            steps = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)]
            beyond &= ~(steps > 0.6)
            run = np.convolve(beyond.astype(int), np.ones(8, int), "valid")
            if (run >= 8).any():
                r0 = int(np.argmax(run >= 8))
                # a real invasion is reached by walking/running across the net line:
                # the frames just before must be close to the net on the player's side
                prev = y[max(0, r0 - 6):r0]
                prev = prev[np.isfinite(prev)]
                if len(prev) == 0 or np.min(np.abs(prev - C.NET_Y)) > 1.0:
                    continue
                fr = a + r0
                depth = float(np.nanmax(np.abs(y[beyond] - C.NET_Y)))
                if depth > 2.5:   # deeper than any lunge over the net: tracking artefact
                    continue
                fouls.append(_foul("NET_INVASION", None, fr, s, tracks.team_of_slot[s], min(0.8, 0.45 + depth * 0.2),
                                   {"max_depth_past_net_m": round(depth, 2)},
                                   f"Player's feet were {depth:.2f} m past the net line while the ball was in play.", fps,
                                   point=pt.idx))
    fouls.sort(key=lambda d: d["frame"])
    for pt in points:
        pt.fouls = [fl for fl in fouls if fl["point"] == pt.idx and fl["status"] != "review"]
    return fouls
