"""Match, team and player analytics; weakness detection; scouting & coaching.

All weakness findings are evidence-based: each carries the metric, sample size,
the comparison reference (opponents / match / pro benchmark), an estimated
impact in points and a confidence that grows with the sample size. Rates are
shrunk towards the reference (beta prior) so that small samples cannot create
spurious weaknesses.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .court import zone_of
from .drills import drill_for
from .rally import Point
from .shots import Shot
from .tracks import N_SLOTS, Tracks

BENCHMARK_FILE = Path(__file__).with_name("benchmarks.json")

GROUP_OF = {  # shot_type -> drill category
    "backhand_volley": "backhand_volley", "forehand_volley": "forehand_volley",
    "backhand_drive": "backhand_drive", "forehand_drive": "forehand_drive",
    "backhand_wall": "off_wall", "forehand_wall": "off_wall", "lob": "lob",
    "bandeja": "overhead", "vibora": "overhead", "smash": "smash", "serve": "serve",
    "return": "return", "dropshot": "forehand_volley", "chiquita": "forehand_drive",
}

PRETTY = {
    "serve": "Serve", "return": "Return", "forehand_drive": "Forehand (groundstroke)",
    "backhand_drive": "Backhand (groundstroke)", "forehand_volley": "Forehand volley",
    "backhand_volley": "Backhand volley", "forehand_wall": "Forehand off the glass",
    "backhand_wall": "Backhand off the glass", "lob": "Lob", "smash": "Smash", "bandeja": "Bandeja",
    "vibora": "Vibora", "dropshot": "Drop shot", "chiquita": "Chiquita", "other": "Other",
}


def load_benchmarks() -> Dict:
    if BENCHMARK_FILE.exists():
        return json.loads(BENCHMARK_FILE.read_text())
    return {}


def _shrunk(k: float, n: float, prior: float, strength: float = 8.0) -> float:
    return (k + prior * strength) / (n + strength)


def _conf_from_n(n: int) -> float:
    return float(min(0.95, 0.35 + 0.6 * (1 - np.exp(-n / 15.0))))


# ----------------------------------------------------------------- movement
def _rally_mask(tracks: Tracks, points: List[Point]) -> np.ndarray:
    m = np.zeros(tracks.n_frames, bool)
    for p in points:
        m[p.start_frame:min(tracks.n_frames, p.end_frame + int(0.5 * tracks.fps))] = True
    return m


def movement_stats(tracks: Tracks, points: List[Point]) -> Dict[int, dict]:
    fps = tracks.fps
    mask = _rally_mask(tracks, points)
    out = {}
    for s in range(N_SLOTS):
        xy = tracks.feet[:, s].astype(float)
        ok = np.isfinite(xy[:, 0]) & mask
        if ok.sum() < fps:
            out[s] = {}
            continue
        # smooth positions (pose jitter) with a 5-frame moving average
        k = 5
        sm = xy.copy()
        for d in range(2):
            v = np.where(np.isfinite(xy[:, d]), xy[:, d], np.nan)
            cs = np.convolve(np.nan_to_num(v), np.ones(k) / k, "same")
            sm[:, d] = cs
        step = np.hypot(*(np.diff(sm, axis=0).T))
        valid_step = ok[1:] & ok[:-1] & (step < 0.5)
        dist = float(step[valid_step].sum())
        speeds = step[valid_step] * fps
        dn = np.abs(xy[ok, 1] - C.NET_Y)
        zones = Counter(zone_of(y) for y in xy[ok, 1])
        tot = sum(zones.values())
        heat = np.zeros((20, 10), np.int32)
        gx = np.clip(xy[ok, 0].astype(int), 0, 9)
        gy = np.clip(xy[ok, 1].astype(int), 0, 19)
        np.add.at(heat, (gy, gx), 1)
        rally_seconds = ok.sum() / fps
        out[s] = {
            "distance_m": round(dist, 1),
            "distance_per_rally_minute_m": round(dist / max(rally_seconds / 60, 1e-6), 1),
            "avg_speed_ms": round(float(np.mean(speeds)), 2) if len(speeds) else None,
            "max_speed_ms": round(float(np.percentile(speeds, 99)), 2) if len(speeds) else None,
            "avg_dist_from_net_m": round(float(np.mean(dn)), 2),
            "zone_time_pct": {z: round(100 * zones.get(z, 0) / tot, 1) for z in ("net", "transition", "defense")},
            "heatmap": heat.tolist(),
            "rally_seconds_tracked": round(rally_seconds, 1),
        }
    return out


def pair_stats(tracks: Tracks, points: List[Point]) -> Dict[str, dict]:
    mask = _rally_mask(tracks, points)
    out = {}
    for team, slots in (("A", (0, 1)), ("B", (2, 3))):
        a, b = tracks.feet[:, slots[0]], tracks.feet[:, slots[1]]
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0]) & mask
        if ok.sum() < 30:
            out[team] = {}
            continue
        za = np.array([zone_of(y) for y in a[ok, 1]])
        zb = np.array([zone_of(y) for y in b[ok, 1]])
        both_net = np.mean((za == "net") & (zb == "net"))
        both_back = np.mean((za == "defense") & (zb == "defense"))
        split = np.mean(((za == "net") & (zb == "defense")) | ((za == "defense") & (zb == "net")))
        spacing = np.hypot(*(a[ok] - b[ok]).T)
        depth_gap = np.abs(np.abs(a[ok, 1] - C.NET_Y) - np.abs(b[ok, 1] - C.NET_Y))
        out[team] = {
            "both_at_net_pct": round(100 * float(both_net), 1),
            "both_back_pct": round(100 * float(both_back), 1),
            "split_pct": round(100 * float(split), 1),
            "avg_partner_distance_m": round(float(np.mean(spacing)), 2),
            "wide_gap_pct": round(100 * float(np.mean(spacing > 6.0)), 1),
            "avg_depth_difference_m": round(float(np.mean(depth_gap)), 2),
        }
    return out


# ----------------------------------------------------------------- shot stats
def shot_stats(shots: List[Shot], slot: Optional[int] = None, team: Optional[str] = None) -> dict:
    ss = [s for s in shots if s.outcome != "dead_ball" and s.point is not None and
          (slot is None or s.player == slot) and (team is None or s.team == team)]
    by_type = defaultdict(lambda: Counter())
    for s in ss:
        c = by_type[s.shot_type]
        c["n"] += 1
        c[s.outcome] += 1
        if s.outcome == "error":
            c["unforced" if (s.pressure_in or 0) < 0.5 else "forced"] += 1
    types = {}
    for t, c in by_type.items():
        n = c["n"]
        types[t] = {"n": n, "winners": c["winner"], "errors": c["error"], "unforced_errors": c["unforced"],
                    "forced_errors": c["forced"], "induced_errors": c["forced_error_induced"],
                    "error_rate": round(c["error"] / n, 3), "winner_rate": round(c["winner"] / n, 3),
                    "efficiency": round((c["winner"] + c["forced_error_induced"] - c["error"]) / n, 3)}
    speeds = [s.speed_kmh for s in ss if s.speed_kmh and s.shot_type in ("smash", "vibora", "forehand_drive", "backhand_drive")]
    smash_sp = [s.speed_kmh for s in ss if s.speed_kmh and s.shot_type == "smash"]
    zones = Counter(s.zone for s in ss)
    n = len(ss)
    return {
        "total": n,
        "by_type": dict(sorted(types.items(), key=lambda kv: -kv[1]["n"])),
        "by_stroke": dict(Counter(s.stroke for s in ss)),
        "by_zone_pct": {z: round(100 * zones.get(z, 0) / max(n, 1), 1) for z in ("net", "transition", "defense")},
        "winners": sum(1 for s in ss if s.outcome == "winner"),
        "errors": sum(1 for s in ss if s.outcome == "error"),
        "unforced_errors": sum(1 for s in ss if s.outcome == "error" and (s.pressure_in or 0) < 0.5),
        "forced_errors": sum(1 for s in ss if s.outcome == "error" and (s.pressure_in or 0) >= 0.5),
        "induced_errors": sum(1 for s in ss if s.outcome == "forced_error_induced"),
        "lob_pct": round(100 * sum(1 for s in ss if s.shot_type == "lob") / max(n, 1), 1),
        "overhead_pct": round(100 * sum(1 for s in ss if s.stroke == "overhead") / max(n, 1), 1),
        "avg_attack_speed_kmh": round(float(np.median(speeds)), 1) if speeds else None,
        "top_smash_kmh": round(float(max(smash_sp)), 1) if smash_sp else None,
        "landing_points": [s.landing for s in ss if s.landing][:400],
    }


def biomech_stats(shots: List[Shot], slot: int) -> dict:
    ss = [s for s in shots if s.player == slot and s.point is not None]
    low = [s.biomech.get("knee_flexion_deg") for s in ss
           if s.biomech.get("knee_flexion_deg") and (s.contact.get("height_m") or 9) < 0.85 and s.stroke != "overhead"]
    ov_el = [s.biomech.get("elbow_deg") for s in ss if s.stroke == "overhead" and s.biomech.get("elbow_deg")]
    ov_h = [s.contact.get("height_m") for s in ss if s.stroke == "overhead" and s.contact.get("height_m")]
    stance = [s.biomech.get("stance_width_m") for s in ss if s.biomech.get("stance_width_m") and s.stroke != "overhead"]
    lat = [abs(s.contact.get("lateral_m")) for s in ss if s.contact.get("lateral_m") is not None and s.stroke in ("forehand", "backhand")]
    hand = Counter(s.contact.get("hand") for s in ss).most_common(1)
    return {
        "dominant_hand": hand[0][0] if hand else None,
        "knee_flexion_low_balls_deg": round(float(np.median(low)), 1) if low else None,
        "n_low_balls": len(low),
        "overhead_elbow_deg": round(float(np.median(ov_el)), 1) if ov_el else None,
        "overhead_contact_height_m": round(float(np.median(ov_h)), 2) if ov_h else None,
        "stance_width_m": round(float(np.median(stance)), 2) if stance else None,
        "contact_spacing_m": round(float(np.median(lat)), 2) if lat else None,
    }


def serve_return_stats(shots: List[Shot], points: List[Point], team: str) -> dict:
    by = {s.idx: s for s in shots}
    sv = [p for p in points if p.server_team == team]
    rc = [p for p in points if p.server_team and p.server_team != team]
    won_sv = sum(1 for p in sv if p.winner == team)
    won_rc = sum(1 for p in rc if p.winner == team)
    first_in = sum(1 for p in sv if p.serve_faults == 0 and p.how not in ("fault", "double_fault"))
    ret_err = 0
    for p in rc:
        if len(p.shot_ids) >= 2 and p.rally_length == 2 and p.winner != team:
            ret_err += 1
    bp_faced = bp_saved = bp_had = bp_conv = 0
    for p in points:
        sb = p.score_before
        if not sb or sb.get("tiebreak"):
            continue
        a, b = sb["points"]
        server = sb["server"]
        recv_pts, srv_pts = (b, a) if server == "A" else (a, b)
        # break point: receivers one point from the game (40-x with x<40, or advantage receivers)
        is_bp = recv_pts >= 3 and recv_pts > srv_pts
        if is_bp:
            receiver = "B" if server == "A" else "A"
            if server == team:
                bp_faced += 1
                bp_saved += p.winner == team
            elif receiver == team:
                bp_had += 1
                bp_conv += p.winner == team
    return {
        "service_points": len(sv), "service_points_won_pct": round(100 * won_sv / max(len(sv), 1), 1),
        "first_serve_in_pct": round(100 * first_in / max(len(sv), 1), 1),
        "double_faults": sum(1 for p in sv if p.how == "double_fault"),
        "return_points": len(rc), "return_points_won_pct": round(100 * won_rc / max(len(rc), 1), 1),
        "return_errors": ret_err,
        "break_points_faced": bp_faced, "break_points_saved": bp_saved,
        "break_points_had": bp_had, "break_points_converted": bp_conv,
    }


def lob_defense_stats(shots: List[Shot], points: List[Point], slot: int) -> dict:
    """When the opponents lob over this player's half: keeps the net with an
    overhead, retreats (lets it bounce / off the glass), and who wins the point."""
    by_pt = defaultdict(list)
    for s in shots:
        if s.point is not None:
            by_pt[s.point].append(s)
    lobbed = kept_net = retreat = won = 0
    win_of = {p.idx: p.winner for p in points}
    for pid, ss in by_pt.items():
        ss.sort(key=lambda s: s.frame)
        for a, b in zip(ss[:-1], ss[1:]):
            if a.shot_type == "lob" and b.player == slot:
                lobbed += 1
                if b.stroke == "overhead":
                    kept_net += 1
                else:
                    retreat += 1
                won += win_of.get(pid) == b.team
    return {"times_lobbed": lobbed, "answered_with_overhead_pct": round(100 * kept_net / max(lobbed, 1), 1),
            "retreated_pct": round(100 * retreat / max(lobbed, 1), 1),
            "points_won_after_lob_pct": round(100 * won / max(lobbed, 1), 1)}


# ----------------------------------------------------------------- weaknesses
def _finding(kind, title, category, detail, evidence, impact, conf, rec, who, polarity="weakness"):
    d = drill_for(category)
    return {"id": kind, "title": title, "category": category, "detail": detail, "evidence": evidence,
            "impact_points": round(float(impact), 1), "confidence": round(float(conf), 2),
            "recommendation": rec, "drill": d, "who": who, "polarity": polarity}


def player_findings(slot: int, name: str, stats: dict, ref_types: Dict[str, dict], bio: dict, bio_ref: dict,
                    move: dict, lobd: dict, fouls: List[dict], team_pair: dict, sr: dict,
                    pos_ref: Optional[dict] = None) -> List[dict]:
    pos_ref = pos_ref or {}
    out = []
    bt = stats["by_type"]
    total_err = stats["errors"]
    overall_er = total_err / max(stats["total"], 1)
    for t, v in bt.items():
        n = v["n"]
        if n < 5 or t in ("other",):
            continue
        ref = ref_types.get(t, {}).get("error_rate", overall_er)
        er = _shrunk(v["errors"], n, max(ref, 0.02))
        if er > max(1.35 * ref, ref + 0.05) and v["errors"] >= 2:
            excess = (er - ref) * n
            out.append(_finding(f"err_{t}", f"{PRETTY.get(t, t)} errors", GROUP_OF.get(t, "movement"),
                                f"{name} misses {v['errors']} of {n} {PRETTY.get(t, t).lower()}s ({100 * v['errors'] / n:.0f}%) vs {100 * ref:.0f}% reference.",
                                {"n": n, "errors": v["errors"], "error_rate": v["error_rate"], "reference_rate": round(ref, 3),
                                 "unforced": v["unforced_errors"]},
                                excess, _conf_from_n(n),
                                f"Prioritise margin on the {PRETTY.get(t, t).lower()}: aim higher/deeper and reduce pace until the error rate is below {100 * ref:.0f}%.",
                                slot))
        good = _shrunk(v["winners"] + v["induced_errors"], n, 0.08) - _shrunk(v["errors"], n, max(ref, 0.02))
        if n >= 6 and good > 0.12:
            out.append(_finding(f"str_{t}", f"Effective {PRETTY.get(t, t).lower()}", GROUP_OF.get(t, "movement"),
                                f"{v['winners']} winners and {v['induced_errors']} forced errors from {n} {PRETTY.get(t, t).lower()}s.",
                                {"n": n, "winners": v["winners"], "induced": v["induced_errors"], "errors": v["errors"]},
                                good * n, _conf_from_n(n), "Use this shot as a pattern finisher.", slot, "strength"))
    zt = move.get("zone_time_pct", {})
    tr_ref = pos_ref.get("transition_time_pct") or 20.0
    if zt.get("transition", 0) > tr_ref + 8:
        out.append(_finding("pos_transition", "Too much time in no-man's land", "transition",
                            f"{zt['transition']:.0f}% of rally time spent {C.ZONE_NET_MAX:.1f}-{C.ZONE_TRANSITION_MAX:.1f} m from the net "
                            f"(reference {tr_ref:.0f}%).",
                            {"transition_time_pct": zt["transition"], "reference_pct": round(tr_ref, 1),
                             "shots_from_transition_pct": stats["by_zone_pct"].get("transition")},
                            (zt["transition"] - tr_ref) / 100 * stats["total"] * 0.3, 0.7,
                            "After a lob or a defensive shot, either commit to the net (3-4 m) or stay behind the service line.", slot))
    net_ref = pos_ref.get("net_time_pct")
    if net_ref and zt.get("net", 100) < net_ref - 12:
        out.append(_finding("pos_net", "Not taking the net enough", "transition",
                            f"Only {zt['net']:.0f}% of rally time at the net (reference {net_ref:.0f}%).",
                            {"net_time_pct": zt["net"], "reference_pct": round(net_ref, 1)},
                            (net_ref - zt["net"]) / 100 * stats["total"] * 0.25, 0.65,
                            "Use the lob to take the net and advance together with your partner.", slot))
    lob_ref = pos_ref.get("points_won_after_lob_pct") or 45.0
    if lobd.get("times_lobbed", 0) >= 5:
        if lobd["points_won_after_lob_pct"] < lob_ref - 10:
            out.append(_finding("lob_defense", "Vulnerable to lobs", "lob_defense",
                                f"Lobbed {lobd['times_lobbed']} times; wins only {lobd['points_won_after_lob_pct']:.0f}% of those points "
                                f"(answers with an overhead {lobd['answered_with_overhead_pct']:.0f}% of the time).",
                                lobd, lobd["times_lobbed"] * (0.5 - lobd["points_won_after_lob_pct"] / 100),
                                _conf_from_n(lobd["times_lobbed"]),
                                "Turn sideways immediately on the lob and decide early: bandeja if you reach it in front, otherwise retreat and defend off the glass.",
                                slot))
    kf, kref = bio.get("knee_flexion_low_balls_deg"), bio_ref.get("knee_flexion_low_balls_deg")
    if kf and kref and bio.get("n_low_balls", 0) >= 6 and kf > kref + 8 and kf > 145:
        out.append(_finding("bio_knees", "Stiff legs on low balls", "footwork_low",
                            f"Median knee angle at contact on low balls {kf:.0f}° vs {kref:.0f}° reference (higher = straighter legs).",
                            {"knee_deg": kf, "reference_deg": kref, "n": bio["n_low_balls"]},
                            0.03 * bio["n_low_balls"], _conf_from_n(bio["n_low_balls"]),
                            "Lower the hips on balls below the knee; bend the legs, keep the back straight.", slot))
    # only "likely" infractions count as a weakness; "possible" ones stay in the rules tab for review
    nf = Counter(f["type"] for f in fouls if f["status"] == "likely")
    for t, c in nf.items():
        cat = "rules_serve" if t.startswith("SERVE") or t == "DOUBLE_FAULT" else ("rules_net" if t == "NET_INVASION" else "movement")
        out.append(_finding(f"foul_{t}", f"Rule infractions: {t.replace('_', ' ').lower()} x{c}", cat,
                            f"{c} likely infraction(s) of this type were detected (watch the clips in Rules & fouls).",
                            {"count": c}, c, 0.6, "Review the clips and fix the technique cause.", slot))
    out.sort(key=lambda f: (f["polarity"] != "weakness", -f["impact_points"] * f["confidence"]))
    return out


def team_findings(team: str, pair: dict, sr: dict, stats: dict, opp_stats: dict, name: str,
                  ref: Optional[dict] = None) -> List[dict]:
    out = []
    if pair.get("split_pct", 0) > max(18, (ref or {}).get("split_pct", 0) + 8):
        out.append(_finding(f"{team}_split", "Pair often split (one up, one back)", "pair_positioning",
                            f"{name} are split {pair['split_pct']:.0f}% of rally time; both at the net only {pair['both_at_net_pct']:.0f}%.",
                            pair, 0.04 * stats["total"], 0.7,
                            "Move as a unit: go to the net together after a good lob, retreat together when lobbed.", team))
    if pair.get("wide_gap_pct", 0) > 15:
        out.append(_finding(f"{team}_gap", "Gap through the middle", "middle",
                            f"Partners more than 6 m apart {pair['wide_gap_pct']:.0f}% of rally time.", pair,
                            0.02 * stats["total"], 0.6, "Shift together towards the ball side to close the middle.", team))
    if sr.get("service_points", 0) >= 8 and sr["service_points_won_pct"] < 50:
        out.append(_finding(f"{team}_serve", "Losing too many service points", "serve",
                            f"Won {sr['service_points_won_pct']:.0f}% of service points ({sr['service_points']} played).", sr,
                            sr["service_points"] * (0.6 - sr["service_points_won_pct"] / 100), _conf_from_n(sr["service_points"]),
                            "Serve to the T/body with slice and follow to the net together with the partner.", team))
    if sr.get("double_faults", 0) >= 2:
        out.append(_finding(f"{team}_df", f"Double faults x{sr['double_faults']}", "rules_serve",
                            "Repeated double faults hand free points to the opponents.", sr, sr["double_faults"], 0.7,
                            "Second serve: slower, safer, aim deep to the T.", team))
    ue, oue = stats["unforced_errors"], opp_stats.get("unforced_errors", 0)
    if stats["total"] >= 30 and ue > 1.3 * max(oue, 1) and ue - oue >= 4:
        out.append(_finding(f"{team}_ue", "More unforced errors than the opponents", "forehand_drive",
                            f"{ue} unforced errors vs {oue} for the opponents.", {"unforced": ue, "opponent_unforced": oue},
                            ue - oue, 0.7, "Play with more margin; build the point with lobs and deep balls before attacking.", team))
    return out


def scouting_report(opp_team: str, opp_name: str, players: Dict[int, dict], team_info: dict, names: Dict[int, str]) -> dict:
    """How to beat `opp_team`: who to target, which shots to force, tendencies."""
    slots = [s for s in players if players[s]["team"] == opp_team]
    plan, targets = [], []
    for s in slots:
        p = players[s]
        weak = [f for f in p["findings"] if f["polarity"] == "weakness"]
        score = sum(f["impact_points"] * f["confidence"] for f in weak)
        targets.append((score, s, weak))
    targets.sort(reverse=True)
    if targets and targets[0][0] > 0:
        sc, s, weak = targets[0]
        plan.append({"title": f"Target {names[s]}",
                     "detail": f"{names[s]} has the weaker profile (estimated {sc:.1f} points of weakness). Play the decisive balls to them.",
                     "evidence": [w["title"] for w in weak[:3]]})
    for sc, s, weak in targets:
        for w in weak[:3]:
            t = w["id"]
            if t.startswith("err_"):
                st = t[4:]
                how = {
                    "backhand_volley": "hit low and fast to the backhand side when they are at the net",
                    "forehand_volley": "drive low to their forehand volley",
                    "backhand_drive": "play deep to the backhand corner",
                    "forehand_drive": "move them wide on the forehand side",
                    "backhand_wall": "make them play off the back glass on the backhand side",
                    "forehand_wall": "make them play off the back glass on the forehand side",
                    "lob": "keep the net and force them to lob - their lobs fall short",
                    "bandeja": "lob them to 5-6 m from the net (bandeja zone)",
                    "vibora": "lob to their vibora side",
                    "smash": "lob deep so that the smash is not a put-away",
                    "serve": "attack the return early", "return": "serve to the body/T",
                }.get(st, "pressure that shot")
                plan.append({"title": f"Force {names[s]}'s {PRETTY.get(st, st).lower()}",
                             "detail": f"{w['detail']} Tactic: {how}.", "evidence": [w["title"]]})
            elif t == "lob_defense":
                plan.append({"title": f"Lob {names[s]}", "detail": w["detail"], "evidence": [w["title"]]})
            elif t == "pos_transition":
                plan.append({"title": f"Play at {names[s]}'s feet in transition",
                             "detail": w["detail"] + " Low chiquitas to their feet when they move forward.", "evidence": [w["title"]]})
    for f in team_info.get("findings", []):
        if f["polarity"] == "weakness":
            plan.append({"title": f"Exploit: {f['title'].lower()}", "detail": f["detail"], "evidence": [f["title"]]})
    tendencies = []
    for s in slots:
        p = players[s]
        bt = p["shots"]["by_type"]
        if bt:
            fav = max(bt.items(), key=lambda kv: kv[1]["n"])
            tendencies.append(f"{names[s]}: most used shot {PRETTY.get(fav[0], fav[0]).lower()} ({fav[1]['n']}), "
                              f"lobs {p['shots']['lob_pct']:.0f}% of shots, overheads {p['shots']['overhead_pct']:.0f}%.")
        best = [f for f in p["findings"] if f["polarity"] == "strength"][:1]
        if best:
            tendencies.append(f"{names[s]} danger: {best[0]['title'].lower()} — avoid feeding it ({best[0]['detail']})")
    return {"team": opp_team, "team_name": opp_name, "game_plan": plan[:8], "tendencies": tendencies}


def development_plan(findings: List[dict], who_name: str) -> List[dict]:
    weak = [f for f in findings if f["polarity"] == "weakness" and f.get("drill")]
    seen, weeks = set(), []
    for f in weak:
        cat = f["drill"]["id"]
        if cat in seen:
            continue
        seen.add(cat)
        ev = f["evidence"]
        if "error_rate" in ev and "reference_rate" in ev:
            goal = f"Error rate on this shot from {100 * ev['error_rate']:.0f}% to <= {100 * ev['reference_rate']:.0f}%"
        elif "transition_time_pct" in ev:
            goal = "Time in the transition zone below 20% of rally time"
        elif "knee_deg" in ev:
            goal = f"Median knee angle on low balls <= {ev['reference_deg']:.0f}°"
        elif "points_won_after_lob_pct" in ev:
            goal = "Win at least 50% of points after being lobbed"
        else:
            goal = "Halve the frequency measured in this match"
        weeks.append({"week": len(weeks) + 1, "focus": f["title"], "why": f["detail"], "drill": f["drill"],
                      "goal": goal})
        if len(weeks) == 4:
            break
    while len(weeks) < 4:
        weeks.append({"week": len(weeks) + 1, "focus": "Match play with pattern focus",
                      "why": "Consolidate the improvements in points.", "drill": drill_for("pair_positioning"),
                      "goal": "Play 3 sets tracking the previous weeks' metrics"})
    return weeks


def compute_benchmark(shots: List[Shot], movement: Dict[int, dict], biomech: Dict[int, dict]) -> dict:
    st = shot_stats(shots)
    kv = [b.get("knee_flexion_low_balls_deg") for b in biomech.values() if b.get("knee_flexion_low_balls_deg")]
    return {"by_type": {t: {"error_rate": v["error_rate"], "n": v["n"]} for t, v in st["by_type"].items()},
            "biomech": {"knee_flexion_low_balls_deg": float(np.median(kv)) if kv else None}}


def analyze(tracks: Tracks, shots: List[Shot], points: List[Point], fouls: List[dict],
            names: Optional[Dict[int, str]] = None, team_names: Optional[Dict[str, str]] = None,
            our_team: Optional[str] = None) -> dict:
    names = names or {m["slot"]: m["name"] for m in tracks.slot_meta}
    team_names = team_names or {"A": "Team A", "B": "Team B"}
    move = movement_stats(tracks, points)
    pairs = pair_stats(tracks, points)
    bench = load_benchmarks()
    bio = {s: biomech_stats(shots, s) for s in range(N_SLOTS)}
    match_ref = compute_benchmark(shots, move, bio)
    ref_types = dict(match_ref["by_type"])
    for t, v in bench.get("by_type", {}).items():  # blend with pro benchmark when available
        if t in ref_types and v.get("n", 0) > 10:
            ref_types[t] = {"error_rate": 0.5 * ref_types[t]["error_rate"] + 0.5 * v["error_rate"]}
    pos_ref = dict(bench.get("position", {}))
    if not pos_ref:  # no pro benchmark yet: compare with the match average
        zs = [m.get("zone_time_pct", {}) for m in move.values() if m]
        if zs:
            pos_ref = {"transition_time_pct": float(np.median([z["transition"] for z in zs])),
                       "net_time_pct": float(np.median([z["net"] for z in zs]))}
    bio_ref = dict(match_ref["biomech"])
    if bench.get("biomech", {}).get("knee_flexion_low_balls_deg"):
        bio_ref["knee_flexion_low_balls_deg"] = bench["biomech"]["knee_flexion_low_balls_deg"]

    teams = {}
    for team in ("A", "B"):
        st = shot_stats(shots, team=team)
        sr = serve_return_stats(shots, points, team)
        teams[team] = {"name": team_names.get(team, f"Team {team}"), "shots": st, "serve_return": sr,
                       "pair": pairs.get(team, {}),
                       "points_won": sum(1 for p in points if p.winner == team),
                       "winners": st["winners"], "unforced_errors": st["unforced_errors"],
                       "forced_errors": st["forced_errors"]}
    for team in ("A", "B"):
        o = "B" if team == "A" else "A"
        teams[team]["findings"] = team_findings(team, teams[team]["pair"], teams[team]["serve_return"],
                                                teams[team]["shots"], teams[o]["shots"], teams[team]["name"],
                                                bench.get("pair"))

    players = {}
    for s in range(N_SLOTS):
        team = tracks.team_of_slot[s]
        st = shot_stats(shots, slot=s)
        lobd = lob_defense_stats(shots, points, s)
        pf = [f for f in fouls if f["player"] == s]
        fnd = player_findings(s, names[s], st, ref_types, bio[s], bio_ref, move.get(s, {}), lobd, pf,
                              teams[team]["pair"], teams[team]["serve_return"], pos_ref)
        players[s] = {"slot": s, "name": names[s], "team": team, "meta": tracks.slot_meta[s] if s < len(tracks.slot_meta) else {},
                      "shots": st, "movement": move.get(s, {}), "biomech": bio[s], "lob_defense": lobd,
                      "fouls": pf, "findings": fnd,
                      "plan": development_plan(fnd, names[s])}

    scouting = {t: scouting_report(t, teams[t]["name"], players, teams[t], names) for t in ("A", "B")}
    longest = max(points, key=lambda p: p.rally_length) if points else None
    summary = {
        "points": len(points),
        "points_won": {t: teams[t]["points_won"] for t in ("A", "B")},
        "final_score": points[-1].score_after if points else None,
        "avg_rally_length": round(float(np.mean([p.rally_length for p in points])), 1) if points else 0,
        "longest_rally": {"point": longest.idx, "shots": longest.rally_length, "time_s": longest.start_s} if longest else None,
        "total_shots": sum(1 for s in shots if s.point is not None),
        "how_points_ended": dict(Counter(p.how for p in points)),
        "fouls_reported": sum(1 for f in fouls if f["status"] != "review"),
        "scoreboard_points": sum(1 for p in points if p.source == "scoreboard"),
    }
    out = {"summary": summary, "teams": teams, "players": players, "scouting": scouting,
           "match_reference": match_ref}
    if our_team in ("A", "B"):
        opp = "B" if our_team == "A" else "A"
        ours = [f for s in range(N_SLOTS) if tracks.team_of_slot[s] == our_team for f in players[s]["findings"]]
        ours += teams[our_team]["findings"]
        ours.sort(key=lambda f: (f["polarity"] != "weakness", -f["impact_points"] * f["confidence"]))
        out["our_team"] = {"team": our_team, "name": teams[our_team]["name"], "findings": ours,
                           "plan": development_plan(ours, teams[our_team]["name"]),
                           "how_to_beat_opponents": scouting[opp]}
    return out


def benchmark_from_results(result_dirs: List[Path]) -> dict:
    """Pro reference built from analysed WPT matches (aggregation, no training)."""
    agg = defaultdict(lambda: [0, 0])
    trans, net, lob, knees, split = [], [], [], [], []
    for d in result_dirs:
        a = json.loads((Path(d) / "analysis.json").read_text())
        for p in a["players"].values():
            for t, v in p["shots"]["by_type"].items():
                agg[t][0] += v["errors"]
                agg[t][1] += v["n"]
            z = p["movement"].get("zone_time_pct")
            if z:
                trans.append(z["transition"]); net.append(z["net"])
            if p["lob_defense"]["times_lobbed"] >= 5:
                lob.append(p["lob_defense"]["points_won_after_lob_pct"])
            if p["biomech"].get("knee_flexion_low_balls_deg"):
                knees.append(p["biomech"]["knee_flexion_low_balls_deg"])
        for t in a["teams"].values():
            if t["pair"].get("split_pct") is not None:
                split.append(t["pair"]["split_pct"])
    med = lambda v: round(float(np.median(v)), 2) if v else None
    return {"source": [Path(d).name for d in result_dirs], "level": "WPT Master Final (pro)",
            "by_type": {t: {"error_rate": round(e / n, 4), "n": n} for t, (e, n) in agg.items() if n},
            "position": {"transition_time_pct": med(trans), "net_time_pct": med(net),
                         "points_won_after_lob_pct": med(lob)},
            "biomech": {"knee_flexion_low_balls_deg": med(knees)},
            "pair": {"split_pct": med(split)}}
