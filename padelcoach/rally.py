"""Rally / point segmentation, point-winner inference and score progression.

Winner inference follows the FIP point-loss rules using what the ball did
after the last shot of the rally:
  * ball's next floor contact on the hitter's own side  -> hitter's error (net)
  * ball hits the opponents' glass/mesh before bouncing -> hitter's error (Rule 13g)
  * ball bounced in the opponents' court and was not returned -> hitter wins
    (winner if no opponent could reach it, forced error otherwise)
When the broadcast scoreboard is readable its point changes override this.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .events import BallEvent
from .scoring import PadelScorer, ScoreState
from .shots import Shot
from .tracks import N_SLOTS, Tracks


@dataclass
class Point:
    idx: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    shot_ids: List[int]
    server: Optional[int] = None
    server_team: Optional[str] = None
    serve_faults: int = 0
    winner: Optional[str] = None      # 'A' / 'B'
    how: str = "unknown"             # winner | forced_error | unforced_error | foul | double_fault | ace | unknown
    ending_player: Optional[int] = None
    ending_shot: Optional[int] = None
    reason: str = ""
    confidence: float = 0.3
    source: str = "inferred"          # inferred | scoreboard | manual
    rally_length: int = 0
    duration_s: float = 0.0
    score_before: dict = field(default_factory=dict)
    score_after: dict = field(default_factory=dict)
    game_won: Optional[str] = None
    set_won: Optional[str] = None
    fouls: List[dict] = field(default_factory=list)
    sync_sets: Optional[List[List[int]]] = None   # broadcast set score after this point

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 2)
        return d


def segment_points(shots: List[Shot], fps: float) -> List[List[Shot]]:
    """A rally starts at a serve and runs while consecutive shots are less than
    MAX_RALLY_SHOT_GAP apart. Hits outside rallies (balls fed to the server
    between points) are marked 'dead_ball' and excluded. If a serve was missed,
    a deep hit after a long pause also starts a rally."""
    rallies: List[List[Shot]] = []
    cur: List[Shot] = []
    for sh in shots:
        gap = (sh.frame - cur[-1].frame) / fps if cur else 99.0
        starts = sh.context == "serve" or (gap > 6.0 and sh.dist_from_net_m > 6.0 and sh.stroke != "overhead")
        if starts:
            if cur:
                rallies.append(cur)
            cur = [sh]
        elif cur and gap <= C.MAX_RALLY_SHOT_GAP_SECONDS:
            cur.append(sh)
        else:
            if cur:
                rallies.append(cur)
                cur = []
            sh.outcome = "dead_ball"
    if cur:
        rallies.append(cur)
    return rallies


def _pressure_on(shot: Shot, prev: Optional[Shot], tracks: Tracks) -> float:
    """0..1 how difficult the incoming ball was for the player hitting `shot`."""
    if prev is None:
        return 0.2
    p = 0.0
    if prev.shot_type in ("smash", "vibora"):
        p += 0.45
    elif prev.shot_type in ("bandeja", "lob"):
        p += 0.15
    if (prev.speed_kmh or 0) > 75:
        p += 0.25
    if shot.context == "off_wall":
        p += 0.15
    if shot.context == "volley" and (shot.contact.get("height_m") or 1.0) < 0.6:
        p += 0.2  # low volley at the feet
    dt = (shot.frame - prev.frame) / tracks.fps
    a, b = prev.frame, shot.frame
    fa, fb = tracks.feet[a, shot.player], tracks.feet[b, shot.player]
    if np.isfinite(fa[0]) and np.isfinite(fb[0]) and dt > 0:
        move = float(np.hypot(*(fb - fa)))
        if move / dt > 3.5:
            p += 0.25  # had to sprint
    return float(min(1.0, p))


def _nearest_opponent_dist(tracks: Tracks, team: str, frame: int, xy) -> float:
    best = np.inf
    for s in range(N_SLOTS):
        if tracks.team_of_slot[s] == team:
            continue
        p = tracks.feet[min(frame, tracks.n_frames - 1), s]
        if np.isfinite(p[0]):
            best = min(best, float(np.hypot(p[0] - xy[0], p[1] - xy[1])))
    return best


def infer_outcome(rally: List[Shot], events: List[BallEvent], tracks: Tracks, next_start: Optional[int]) -> dict:
    fps = tracks.fps
    last = rally[-1]
    team = last.team
    opp = "B" if team == "A" else "A"
    horizon = last.frame + int(4 * fps)
    if next_start is not None:
        horizon = min(horizon, next_start - 1)
    after = [e for e in events if last.frame < e.frame <= horizon and e.kind != "hit"]
    opp_side = "far" if last.side == "near" else "near"
    prev = rally[-2] if len(rally) > 1 else None
    pressure = _pressure_on(last, prev, tracks)

    def err(reason, conf, foul=None):
        how = "forced_error" if pressure >= 0.5 else "unforced_error"
        return {"winner": opp, "how": how if not foul else "foul", "ending_player": last.player,
                "reason": reason, "confidence": conf, "foul": foul, "pressure": pressure}

    if last.context == "serve" and len(rally) == 1:
        # serve not returned
        if last.landing is not None:
            return {"winner": team, "how": "ace", "ending_player": last.player, "reason": "Serve landed and was not returned.",
                    "confidence": 0.45, "pressure": 0.0}
        return {"winner": opp, "how": "fault", "ending_player": last.player, "reason": "Serve not returned and no valid bounce seen (likely fault).",
                "confidence": 0.35, "pressure": 0.0}

    first = after[0] if after else None
    bounces_opp = [e for e in after if e.kind == "bounce" and e.side == opp_side]
    if first is not None and first.kind == "bounce" and first.side == last.side:
        return err("Ball dropped on the hitter's own side (did not clear the net).", 0.6)
    if first is not None and first.kind == "wall" and first.side == opp_side and first.wall and "back" in first.wall:
        return err(f"Ball hit the opponents' {first.wall.replace('_', ' ')} wall before bouncing (FIP Rule 13g).", 0.5,
                   foul="DIRECT_WALL")
    if bounces_opp:
        land = bounces_opp[0]
        d = _nearest_opponent_dist(tracks, team, land.frame, land.court)
        double = len(bounces_opp) >= 2
        if d > 2.8:
            how, conf = "winner", 0.6
            reason = f"Ball landed {d:.1f} m from the nearest opponent and was not returned."
        else:
            how, conf = "forced_error", 0.5
            reason = "Opponents reached the ball but could not return it."
        if double:
            reason += " Ball bounced twice on the opponents' side."
            conf += 0.1
        return {"winner": team, "how": how, "ending_player": last.player, "reason": reason, "confidence": min(conf, 0.8),
                "pressure": pressure, "forced_player": None}
    # No informative ball events: use shot quality priors.
    if last.shot_type in ("smash", "vibora") or (last.zone == "net" and (last.speed_kmh or 0) > 60):
        return {"winner": team, "how": "winner", "ending_player": last.player,
                "reason": "Rally ended on an attacking shot (ball tracking lost after the shot).", "confidence": 0.4,
                "pressure": pressure}
    return err("Rally ended on this shot without a visible bounce in the opponents' court (likely net or out).", 0.35)


def build_points(shots: List[Shot], events: List[BallEvent], tracks: Tracks, golden_point: bool = False,
                 scoreboard_points: Optional[List[dict]] = None, overrides: Optional[Dict[int, str]] = None,
                 start_score: Optional[ScoreState] = None) -> List[Point]:
    fps = tracks.fps
    rallies = segment_points(shots, fps)
    points: List[Point] = []
    pending_faults = 0
    i = 0
    raw_outcomes = []
    for ri, r in enumerate(rallies):
        nxt = rallies[ri + 1][0].frame if ri + 1 < len(rallies) else None
        raw_outcomes.append(infer_outcome(r, events, tracks, nxt))

    first_server = None
    for r in rallies:
        if r[0].context == "serve":
            first_server = 0 if r[0].team == "A" else 1
            break
    scorer = PadelScorer(golden_point=golden_point, first_server=first_server or 0, start=start_score)

    for ri, (r, oc) in enumerate(zip(rallies, raw_outcomes)):
        is_serve = r[0].context == "serve"
        # First-serve fault: a lone serve followed by another serve from the same team soon after.
        if is_serve and oc["how"] == "fault" and pending_faults == 0 and ri + 1 < len(rallies):
            nr = rallies[ri + 1]
            if nr[0].context == "serve" and nr[0].team == r[0].team and (nr[0].frame - r[0].frame) / fps < 40:
                pending_faults = 1
                for sh in r:
                    sh.outcome = "fault"
                continue
        pt = Point(len(points), r[0].frame, r[-1].frame, round(r[0].frame / fps, 2), round(r[-1].frame / fps, 2),
                   [s.idx for s in r], rally_length=len(r), duration_s=round((r[-1].frame - r[0].frame) / fps, 2))
        if is_serve:
            pt.server, pt.server_team = r[0].player, r[0].team
        pt.serve_faults = pending_faults
        if pending_faults and oc["how"] == "fault":
            oc = dict(oc, how="double_fault", reason="Second consecutive service fault (double fault).", confidence=0.4)
        pending_faults = 0
        pt.winner, pt.how, pt.reason, pt.confidence = oc["winner"], oc["how"], oc["reason"], oc["confidence"]
        pt.ending_player = oc.get("ending_player")
        pt.ending_shot = r[-1].idx
        if oc.get("foul"):
            pt.fouls.append({"type": oc["foul"], "shot": r[-1].idx})
        points.append(pt)
        for k, sh in enumerate(r):
            sh.point, sh.num_in_point = pt.idx, k + 1
            if k > 0 and sh.shot_type not in ("serve",) and k == 1:
                sh.shot_type = "return" if r[0].context == "serve" else sh.shot_type
            sh.pressure_in = round(_pressure_on(sh, r[k - 1] if k > 0 else None, tracks), 2)
        last = r[-1]
        if pt.how in ("winner", "ace"):
            last.outcome = "winner"
        elif pt.how in ("forced_error", "unforced_error", "foul", "double_fault", "fault"):
            last.outcome = "error"
            if pt.how == "forced_error" and len(r) > 1:
                r[-2].outcome = "forced_error_induced"

    # Scoreboard (ground truth from broadcast overlay) overrides inferred winners.
    if scoreboard_points:
        _apply_scoreboard(points, scoreboard_points, fps)
    if overrides:
        for idx, w in overrides.items():
            if 0 <= int(idx) < len(points) and w in ("A", "B"):
                points[int(idx)].winner = w
                points[int(idx)].source = "manual"
                points[int(idx)].confidence = 1.0

    for pt in points:
        pt.score_before = scorer.state.to_dict(golden_point)
        if pt.server_team is None:
            pt.server_team = "AB"[scorer.state.server]
        if pt.winner:
            info = scorer.award(0 if pt.winner == "A" else 1)
            pt.game_won, pt.set_won = info["game_won"], info["set_won"]
        if pt.sync_sets:   # the broadcast says what the games score is now: trust it
            _resync(scorer, pt.sync_sets)
            pt.game_won = pt.game_won or pt.winner
        pt.score_after = scorer.state.to_dict(golden_point)
    return points


def _apply_scoreboard(points: List[Point], sb: List[dict], fps: float) -> None:
    """sb: chronological scoreboard events {"frame", "winner", "kind"}.
    'point' events label the latest not-yet-confirmed point that ended within
    25 s before the change. 'game' events (games count went up) label the
    latest point that ended within 120 s before it - the game-winning point -
    even if a point event already labelled it."""
    def relabel(p: Point, w: str) -> None:
        if p.winner != w:
            p.reason = "Scoreboard: " + p.reason + " (inferred result corrected by broadcast score)"
            if p.how in ("winner", "ace"):
                p.how = "forced_error"
            elif p.how in ("unforced_error", "forced_error", "foul"):
                p.how = "winner"
        p.winner = w
        p.source = "scoreboard"
        p.confidence = 0.95

    for ch in sb:
        f = ch["frame"]
        if ch.get("kind") == "game":
            cands = [p for p in points if p.end_frame <= f and f - p.end_frame <= 120 * fps]
        else:
            cands = [p for p in points if p.end_frame <= f and f - p.end_frame <= 25 * fps and p.source != "scoreboard"]
        if cands:
            relabel(cands[-1], ch["winner"])
            if ch.get("kind") == "game" and ch.get("sets"):
                cands[-1].sync_sets = ch["sets"]


def _resync(scorer: PadelScorer, sets: List[List[int]]) -> None:
    """Set the scorer to the broadcast set/games score read at a game change."""
    st = scorer.state
    ocr = [list(map(int, x)) for x in sets if len(x) == 2]
    if not ocr:
        return
    if len(ocr) >= len(st.sets):
        st.sets = ocr
    else:  # broadcast shows only the most recent set(s)
        st.sets[-len(ocr):] = ocr
    st.points = [0, 0]
    won = [0, 0]
    for a, b in st.sets[:-1]:
        won[0 if a > b else 1] += 1
    a, b = st.sets[-1]
    st.tiebreak = (a == 6 and b == 6)
    if (max(a, b) >= 6 and abs(a - b) >= 2) or max(a, b) == 7:
        won[0 if a > b else 1] += 1
        st.sets.append([0, 0])
        st.tiebreak = False
    st.sets_won = won
