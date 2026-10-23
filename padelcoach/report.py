"""Standalone, printable HTML coaching report built from a result directory."""
from __future__ import annotations

import html
import json
from pathlib import Path

HOW = {"winner": "Winner", "forced_error": "Forced error", "unforced_error": "Unforced error", "foul": "Rule infraction",
       "double_fault": "Double fault", "ace": "Ace / unreturned serve", "fault": "Service fault", "unknown": "Unknown"}


def _e(v) -> str:
    return html.escape(str(v if v is not None else "–"))


def _findings(fs, n=6) -> str:
    out = []
    for f in fs[:n]:
        d = f.get("drill") or {}
        out.append(f"<li><b>{_e(f['title'])}</b> <span class='m'>({'strength' if f['polarity'] == 'strength' else f'~{f['impact_points']} pts'}, "
                   f"confidence {round(100 * f['confidence'])}%)</span><br>{_e(f['detail'])}"
                   + (f"<br><i>Fix:</i> {_e(f['recommendation'])}" if f["polarity"] != "strength" else "")
                   + (f"<br><i>Drill:</i> {_e(d.get('name'))} – {_e(d.get('setup'))} ({_e(d.get('dose'))})" if d and f["polarity"] != "strength" else "")
                   + "</li>")
    return "<ul>" + "".join(out) + "</ul>" if out else "<p class='m'>No significant findings.</p>"


def _plan(plan) -> str:
    return "<ol>" + "".join(f"<li><b>{_e(w['focus'])}</b> – {_e((w.get('drill') or {}).get('name'))}. Goal: {_e(w['goal'])}</li>" for w in plan) + "</ol>"


def render_report(rd: Path) -> str:
    a = json.loads((rd / "analysis.json").read_text())
    meta = json.loads((rd / "meta.json").read_text())
    pts = json.loads((rd / "points.json").read_text())
    fouls = [f for f in json.loads((rd / "fouls.json").read_text()) if f["status"] != "review"]
    tn, names = meta["team_names"], {int(k): v for k, v in meta["names"].items()}
    s = a["summary"]
    sc = s.get("final_score") or {}
    A, B = a["teams"]["A"], a["teams"]["B"]

    def row(label, ka, kb):
        return f"<tr><td>{label}</td><td>{_e(ka)}</td><td>{_e(kb)}</td></tr>"

    body = [f"<h1>Padel match report – {_e(meta['video']['name'])}</h1>",
            f"<p class='m'>Generated {_e(meta['generated_at'])} · tracking source {_e(meta['tracks']['source'])} · "
            f"point winners: {'broadcast scoreboard' if s.get('scoreboard_points') else 'inferred from tracking (verify in the app)'}</p>",
            f"<h2>Result</h2><p class='big'>{_e(tn['A'])} &nbsp; {_e(sc.get('games_text'))} &nbsp; {_e(tn['B'])}</p>",
            "<table><tr><th></th><th>" + _e(tn["A"]) + "</th><th>" + _e(tn["B"]) + "</th></tr>",
            row("Points won", A["points_won"], B["points_won"]), row("Winners", A["winners"], B["winners"]),
            row("Unforced errors", A["unforced_errors"], B["unforced_errors"]),
            row("Service points won %", A["serve_return"]["service_points_won_pct"], B["serve_return"]["service_points_won_pct"]),
            row("Return points won %", A["serve_return"]["return_points_won_pct"], B["serve_return"]["return_points_won_pct"]),
            row("Both at the net %", A["pair"].get("both_at_net_pct"), B["pair"].get("both_at_net_pct")),
            row("Lobs % of shots", A["shots"]["lob_pct"], B["shots"]["lob_pct"]), "</table>",
            f"<p>{s['points']} points, {s['total_shots']} shots, average rally {s['avg_rally_length']} shots. "
            + ", ".join(f"{HOW.get(k, k)}: {v}" for k, v in s["how_points_ended"].items()) + "</p>"]
    if "our_team" in a:
        o = a["our_team"]
        body += [f"<h2>Our team – {_e(o['name'])}</h2><h3>Weaknesses</h3>", _findings([f for f in o["findings"] if f["polarity"] == "weakness"], 8),
                 "<h3>Training plan</h3>", _plan(o["plan"]),
                 f"<h3>Game plan vs {_e(o['how_to_beat_opponents']['team_name'])}</h3><ul>"
                 + "".join(f"<li><b>{_e(g['title'])}</b> – {_e(g['detail'])}</li>" for g in o["how_to_beat_opponents"]["game_plan"]) + "</ul>"]
    for t in ("A", "B"):
        sct = a["scouting"][t]
        body += [f"<h2>Scouting – how to beat {_e(tn[t])}</h2><ul>" + "".join(f"<li><b>{_e(g['title'])}</b> – {_e(g['detail'])}</li>" for g in sct["game_plan"]) + "</ul>",
                 "<p class='m'>" + "<br>".join(_e(x) for x in sct["tendencies"]) + "</p>"]
    body.append("<h2>Players</h2>")
    for k, p in a["players"].items():
        st, mv, bm = p["shots"], p.get("movement", {}), p.get("biomech", {})
        z = mv.get("zone_time_pct", {})
        body += [f"<h3>{_e(p['name'])} ({_e(tn[p['team']])})</h3>",
                 f"<p>{st['total']} shots · {st['winners']} winners · {st['errors']} errors ({st['unforced_errors']} unforced) · "
                 f"net {z.get('net', '–')}% / transition {z.get('transition', '–')}% / defence {z.get('defense', '–')}% · "
                 f"{round(mv.get('distance_m') or 0)} m covered · {_e(bm.get('dominant_hand'))}-handed</p>",
                 _findings(p["findings"], 5), "<b>Plan:</b>", _plan(p["plan"])]
    if fouls:
        body.append("<h2>Rule infractions (FIP Rules of Padel 2026)</h2><table><tr><th>Time</th><th>Player</th><th>Rule</th><th>Infraction</th><th>Status</th></tr>")
        for f in fouls:
            m, sec = divmod(int(f["time_s"]), 60)
            body.append(f"<tr><td>{m}:{sec:02d}</td><td>{_e(names.get(f['player'], f['player']))}</td><td>{_e(f['rule'])}</td>"
                        f"<td>{_e(f['name'])}: {_e(f['explanation'])}</td><td>{_e(f['status'])} {round(100 * f['confidence'])}%</td></tr>")
        body.append("</table>")
    body.append("<h2>Point by point</h2><table><tr><th>#</th><th>Score before</th><th>Rally</th><th>Winner</th><th>How</th></tr>")
    for p in pts:
        sb = p.get("score_before") or {}
        body.append(f"<tr><td>{p['idx'] + 1}</td><td>{_e(sb.get('games_text'))} {_e(sb.get('label'))}</td><td>{p['rally_length']}</td>"
                    f"<td>{_e(tn.get(p['winner'], p['winner']))}</td><td>{_e(HOW.get(p['how'], p['how']))}</td></tr>")
    body.append("</table>")
    css = """body{font:14px/1.5 system-ui,Segoe UI,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#17202b;background:#fff}
    h1{font-size:22px}h2{font-size:17px;margin-top:26px;border-bottom:1px solid #dde3ea;padding-bottom:4px}h3{font-size:15px;margin-bottom:4px}
    table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #e5e9ef;padding:4px 6px;text-align:left;vertical-align:top}
    .m{color:#5d6b7c}.big{font-size:20px;font-weight:700}li{margin:4px 0}@media print{h2{break-after:avoid}}"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Padel match report</title><style>{css}</style></head><body>{''.join(body)}</body></html>"
