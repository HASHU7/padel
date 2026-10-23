/* Padel AI Coach - single page app (vanilla JS, no external dependencies). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtT = (s) => { if (s == null) return "–"; s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
const pct = (v, d = 0) => (v == null ? "–" : `${Number(v).toFixed(d)}%`);
const num = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

const SHOT_NAMES = {
  serve: "Serve", return: "Return", forehand_drive: "Forehand", backhand_drive: "Backhand",
  forehand_volley: "Forehand volley", backhand_volley: "Backhand volley", forehand_wall: "Forehand off glass",
  backhand_wall: "Backhand off glass", lob: "Lob", smash: "Smash", bandeja: "Bandeja", vibora: "Víbora",
  dropshot: "Drop shot", chiquita: "Chiquita", other: "Other",
};
const SHOT_COLORS = {
  serve: "#a78bfa", return: "#c084fc", forehand_drive: "#3fcf8e", backhand_drive: "#22a6b3",
  forehand_volley: "#7ee787", backhand_volley: "#56d4dd", forehand_wall: "#9be9a8", backhand_wall: "#8ddbe0",
  lob: "#f2b84b", smash: "#ff6b6b", bandeja: "#ff9f6b", vibora: "#ff7eb6", dropshot: "#e3b341", chiquita: "#d29922", other: "#8b98a8",
};
const HOW = {
  winner: "Winner", forced_error: "Forced error", unforced_error: "Unforced error", foul: "Rule infraction",
  double_fault: "Double fault", ace: "Ace / unreturned serve", fault: "Service fault", unknown: "Unknown",
};
const SKEL = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

const state = { videos: [], data: null, name: null, overlay: new Map(), jobTimer: null };

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function toast(msg, ms = 4000) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), ms);
}

/* ------------------------------------------------------------ theme */
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("pac-theme"); } catch (e) { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeBtn").onclick = () => {
    const cur = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = cur;
    try { localStorage.setItem("pac-theme", cur); } catch (e) { /* ignore */ }
  };
})();

/* ------------------------------------------------------------ jobs */
function watchJob(jid, onDone) {
  const badge = $("#jobBadge");
  badge.classList.remove("hidden");
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jid}`);
      badge.innerHTML = `${esc(j.kind)} · ${esc(j.video)} · ${Math.round(j.progress * 100)}% · ${esc(j.message)}`;
      if (j.status === "done" || j.status === "error") {
        clearInterval(state.jobTimer);
        setTimeout(() => badge.classList.add("hidden"), 3000);
        if (j.status === "error") toast("Job failed: " + j.error, 10000);
        else { toast("Analysis finished"); onDone && onDone(j); }
      }
    } catch (e) { clearInterval(state.jobTimer); }
  }, 1000);
}

/* ------------------------------------------------------------ router */
window.addEventListener("hashchange", route);
window.addEventListener("load", route);
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  if (parts[0] === "match" && parts[1]) return viewMatch(parts[1], parts[2] || "overview");
  if (parts[0] === "calibrate" && parts[1]) return viewCalibrate(parts[1]);
  return viewLibrary();
}
function crumbs(html) { $("#crumbs").innerHTML = html; }

/* ------------------------------------------------------------ library */
async function viewLibrary() {
  crumbs("");
  const v = $("#view");
  v.innerHTML = `<h1>Matches</h1><div class="muted">Loading…</div>`;
  const [videos, status] = await Promise.all([api("/api/videos"), api("/api/status")]);
  state.videos = videos;
  const m = status.models;
  const modelNote = (!m.pose || !m.ball) ? `<div class="notice">
      <b>Raw-video models not installed yet.</b> Matches with PadelTracker100 labels are analysed from the annotations.
      For any other video: train the models with <code>colab/padel_train_models.ipynb</code> and put
      <code>padel_pose.onnx</code> / <code>padel_ball.onnx</code> in <code>models/</code>, or run
      <code>colab/padel_gpu_inference.ipynb</code> and import the detections file here.
      <div class="small muted">Pose model: ${m.pose ? "✔" : "✘"} · Ball model: ${m.ball ? "✔" : "✘"} · Shot model: ${m.shot ? "✔" : "✘ (rule-based classifier in use)"}</div></div>` : "";
  v.innerHTML = `<h1>Matches</h1>${modelNote}
    <div class="card" style="margin-bottom:14px"><div class="row">
      <b>Add a match video</b><input type="file" id="upFile" accept="video/*">
      <button id="upBtn">Upload</button><span id="upMsg" class="muted small"></span></div></div>
    <div class="grid g2" id="vlist"></div>`;
  $("#upBtn").onclick = async () => {
    const f = $("#upFile").files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    $("#upMsg").textContent = "Uploading…";
    try { await api("/api/upload", { method: "POST", body: fd }); viewLibrary(); }
    catch (e) { $("#upMsg").textContent = e.message; }
  };
  const list = $("#vlist");
  for (const vid of videos) {
    const modes = Object.entries(vid.modes).filter(([, ok]) => ok).map(([k]) => k);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="row"><h2 style="margin:0">${esc(vid.name)}</h2><div class="spacer"></div>
        ${vid.analysed ? `<span class="pill good">analysed</span>` : `<span class="pill">not analysed</span>`}</div>
      <div class="small muted">${fmtT(vid.info.duration_s)} · ${vid.info.width}×${vid.info.height} @ ${num(vid.info.fps, 0)} fps ·
        calibration: ${vid.calibrated ? "saved" : "broadcast preset"} · scoreboard: ${vid.scoreboard ? "imported" : "none"}</div>
      ${vid.counts ? `<div class="small" style="margin-top:6px">${vid.counts.points} points · ${vid.counts.shots} shots · ${vid.counts.fouls} rule checks flagged</div>` : ""}
      <div class="row" style="margin-top:10px">
        ${vid.analysed ? `<a href="#/match/${encodeURIComponent(vid.name)}"><button class="primary">Open report</button></a>` : ""}
        <select class="mode">${["auto", "labels", "import", "video"].map((m) => `<option value="${m}" ${!(m === "auto" || vid.modes[m]) ? "disabled" : ""}>${m}</option>`).join("")}</select>
        <button class="run">${vid.analysed ? "Re-run analysis" : "Analyse"}</button>
        <a href="#/calibrate/${encodeURIComponent(vid.name)}"><button>Calibrate court</button></a>
        <label class="small muted">Import Colab file <input type="file" class="imp" accept=".npz,.json"></label>
      </div>
      <div class="small muted" style="margin-top:6px">Available sources: ${modes.length ? modes.join(", ") : "none – needs Colab models or detections"}</div>`;
    card.querySelector(".run").onclick = async () => {
      try {
        const r = await post(`/api/videos/${encodeURIComponent(vid.name)}/analyze`, { mode: card.querySelector(".mode").value });
        watchJob(r.job, () => viewLibrary());
      } catch (e) { toast(e.message); }
    };
    card.querySelector(".imp").onchange = async (ev) => {
      const f = ev.target.files[0]; if (!f) return;
      const fd = new FormData(); fd.append("file", f);
      try { const r = await api(`/api/videos/${encodeURIComponent(vid.name)}/import`, { method: "POST", body: fd }); toast("Imported " + r.saved); viewLibrary(); }
      catch (e) { toast(e.message); }
    };
    list.appendChild(card);
  }
}

/* ------------------------------------------------------------ calibration */
async function viewCalibrate(name) {
  crumbs(`/ <a href="#/">Matches</a> / calibrate ${esc(name)}`);
  const v = $("#view");
  const cal = await api(`/api/videos/${encodeURIComponent(name)}/calibration`);
  const info = (state.videos.find((x) => x.name === name) || {}).info || { n_frames: 1, width: cal.width, height: cal.height };
  const pts = Object.assign({}, cal.image_points);
  let sel = Object.keys(cal.landmarks)[0];
  let roi = cal.scoreboard_roi;
  let roiMode = false;
  let lines = cal.court_lines;
  v.innerHTML = `<h1>Court calibration</h1>
    <p class="muted">Pick a landmark on the right, then click its exact position on the court floor. At least 4 (6+ recommended,
      spread over both halves). The orange model of the court must sit on the real lines. Optionally draw the broadcast scoreboard box.</p>
    <div class="grid" style="grid-template-columns:minmax(0,3fr) minmax(220px,1fr)">
      <div class="card"><div class="row" style="margin-bottom:8px">
        <span class="small muted">Frame</span><input type="range" id="fr" min="0" max="${info.n_frames - 1}" value="0" style="flex:1">
        <span id="frv" class="small"></span></div>
        <div class="calib-wrap" id="cw"><img id="cimg"><svg id="csvg" viewBox="0 0 ${cal.width} ${cal.height}" preserveAspectRatio="none"></svg></div>
      </div>
      <div class="card"><div class="lm-list" id="lms"></div>
        <button id="roiBtn" style="margin-top:8px">Draw scoreboard box</button>
        <div id="err" class="small" style="margin:8px 0"></div>
        <div class="row"><button class="primary" id="saveCal">Save calibration</button><button id="clrCal">Clear points</button></div>
        <p class="small muted">Source: ${esc(cal.source)}</p></div></div>`;
  const img = $("#cimg"), svg = $("#csvg");
  const setFrame = (f) => { img.src = `/api/videos/${encodeURIComponent(name)}/frame?f=${f}&w=1600`; $("#frv").textContent = fmtT(f / (info.fps || 30)); };
  $("#fr").oninput = (e) => setFrame(+e.target.value);
  setFrame(0);
  function drawList() {
    $("#lms").innerHTML = Object.keys(cal.landmarks).map((k) => `<button data-k="${k}" class="${k === sel ? "sel" : ""}">${pts[k] ? "●" : "○"} ${k.replace(/_/g, " ")}</button>`).join("");
    $("#lms").querySelectorAll("button").forEach((b) => (b.onclick = () => { sel = b.dataset.k; roiMode = false; drawList(); }));
  }
  function drawSvg() {
    let s = "";
    for (const [a, b] of lines || []) s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#ff8a3d" stroke-width="3" opacity=".85"/>`;
    for (const [k, p] of Object.entries(pts)) if (p) s += `<circle cx="${p[0]}" cy="${p[1]}" r="9" fill="none" stroke="#3fb6ff" stroke-width="3"/><text x="${p[0] + 12}" y="${p[1] - 10}" style="fill:#3fb6ff;font-size:22px">${k.replace(/_/g, " ")}</text>`;
    if (roi) s += `<rect x="${roi[0]}" y="${roi[1]}" width="${roi[2]}" height="${roi[3]}" fill="none" stroke="#f2b84b" stroke-width="3" stroke-dasharray="8 6"/>`;
    svg.innerHTML = s;
  }
  async function preview() {
    const n = Object.values(pts).filter(Boolean).length;
    if (n < 4) { $("#err").textContent = `${n}/4 points`; lines = []; drawSvg(); return; }
    try {
      const r = await post(`/api/videos/${encodeURIComponent(name)}/calibration/preview`, { image_points: pts });
      lines = r.court_lines;
      $("#err").innerHTML = `Mean error: <b>${(r.reprojection_error_m * 100).toFixed(1)} cm</b> ${r.reprojection_error_m < 0.1 ? "✔" : "– check points"}`;
    } catch (e) { $("#err").textContent = e.message; }
    drawSvg();
  }
  const toImg = (ev) => { const r = svg.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * cal.width, (ev.clientY - r.top) / r.height * cal.height]; };
  let roiStart = null;
  $("#cw").onmousedown = (ev) => { if (roiMode) roiStart = toImg(ev); };
  $("#cw").onmouseup = (ev) => {
    const p = toImg(ev);
    if (roiMode && roiStart) {
      roi = [Math.min(roiStart[0], p[0]), Math.min(roiStart[1], p[1]), Math.abs(p[0] - roiStart[0]), Math.abs(p[1] - roiStart[1])].map(Math.round);
      roiStart = null; roiMode = false; $("#roiBtn").textContent = "Draw scoreboard box"; drawSvg(); return;
    }
    pts[sel] = p.map((x) => Math.round(x * 10) / 10);
    const keys = Object.keys(cal.landmarks); const i = keys.indexOf(sel);
    sel = keys[Math.min(i + 1, keys.length - 1)];
    drawList(); preview();
  };
  $("#roiBtn").onclick = () => { roiMode = true; $("#roiBtn").textContent = "Drag a box over the scoreboard…"; };
  $("#clrCal").onclick = () => { for (const k of Object.keys(pts)) delete pts[k]; lines = []; drawList(); drawSvg(); };
  $("#saveCal").onclick = async () => {
    try { await post(`/api/videos/${encodeURIComponent(name)}/calibration`, { image_points: pts, scoreboard_roi: roi }); toast("Calibration saved – re-run the analysis to apply it."); }
    catch (e) { toast(e.message); }
  };
  drawList(); drawSvg();
}

/* ------------------------------------------------------------ match */
const TABS = [["overview", "Overview"], ["video", "Video & HUD"], ["shots", "Shots"], ["points", "Points & score"], ["fouls", "Rules & fouls"],
  ["players", "Players"], ["scouting", "Opponent scouting"], ["ours", "Our team"], ["settings", "Settings"]];

async function loadMatch(name, force) {
  if (!force && state.data && state.name === name) return state.data;
  const d = await api(`/api/results/${encodeURIComponent(name)}`);
  d.shotById = new Map(d.shots.map((s) => [s.idx, s]));
  state.data = d; state.name = name; state.overlay = new Map();
  return d;
}
const pname = (d, slot) => (slot == null ? "–" : d.meta.names[slot] ?? `P${slot}`);
const tname = (d, t) => (t ? d.meta.team_names[t] || `Team ${t}` : "–");
const teamPill = (d, t) => (t ? `<span class="pill ${t}">${esc(tname(d, t))}</span>` : "");

async function viewMatch(name, tab) {
  crumbs(`/ <a href="#/">Matches</a> / ${esc(name)}`);
  const v = $("#view");
  if (state.name !== name) v.innerHTML = `<div class="muted">Loading analysis…</div>`;
  let d;
  try { d = await loadMatch(name); } catch (e) { v.innerHTML = `<div class="notice">${esc(e.message)}</div><a href="#/">Back</a>`; return; }
  const sc = d.analysis.summary.final_score || {};
  v.innerHTML = `
    <div class="score-head">
      <span class="team" style="color:var(--teamA)">${esc(tname(d, "A"))}</span>
      <span class="team">${esc(sc.games_text || "")} <span class="muted small">(${esc(sc.label || "")})</span></span>
      <span class="team" style="color:var(--teamB)">${esc(tname(d, "B"))}</span>
      <span class="spacer"></span>
      <a href="/api/results/${encodeURIComponent(name)}/report.html" target="_blank"><button>Printable report</button></a>
    </div>
    <div class="tabs">${TABS.map(([k, l]) => `<button data-t="${k}" class="${k === tab ? "active" : ""}">${l}</button>`).join("")}</div>
    <div id="tabBody"></div>`;
  v.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => (location.hash = `#/match/${encodeURIComponent(name)}/${b.dataset.t}`)));
  const body = $("#tabBody");
  const fn = { overview: tabOverview, video: tabVideo, shots: tabShots, points: tabPoints, fouls: tabFouls, players: tabPlayers, scouting: tabScouting, ours: tabOurs, settings: tabSettings }[tab] || tabOverview;
  fn(d, body, name);
}

function statCard(label, value, sub = "") { return `<div class="card stat"><span class="l">${label}</span><span class="v">${value}</span>${sub ? `<span class="small muted">${sub}</span>` : ""}</div>`; }

function tabOverview(d, el) {
  const a = d.analysis, s = a.summary, A = a.teams.A, B = a.teams.B, ev = d.meta.evaluation;
  const rows = [
    ["Points won", A.points_won, B.points_won], ["Winners", A.winners, B.winners], ["Unforced errors", A.unforced_errors, B.unforced_errors],
    ["Forced errors", A.forced_errors, B.forced_errors], ["Service points won", pct(A.serve_return.service_points_won_pct), pct(B.serve_return.service_points_won_pct)],
    ["Return points won", pct(A.serve_return.return_points_won_pct), pct(B.serve_return.return_points_won_pct)],
    ["Break points converted", `${A.serve_return.break_points_converted}/${A.serve_return.break_points_had}`, `${B.serve_return.break_points_converted}/${B.serve_return.break_points_had}`],
    ["Double faults", A.serve_return.double_faults, B.serve_return.double_faults],
    ["Both players at the net", pct(A.pair.both_at_net_pct), pct(B.pair.both_at_net_pct)], ["Pair split (one up / one back)", pct(A.pair.split_pct), pct(B.pair.split_pct)],
    ["Lobs (% of shots)", pct(A.shots.lob_pct), pct(B.shots.lob_pct)], ["Overheads (% of shots)", pct(A.shots.overhead_pct), pct(B.shots.overhead_pct)],
    ["Fastest smash (est.)", A.shots.top_smash_kmh ? `${A.shots.top_smash_kmh} km/h` : "–", B.shots.top_smash_kmh ? `${B.shots.top_smash_kmh} km/h` : "–"],
  ];
  const how = Object.entries(s.how_points_ended || {}).sort((x, y) => y[1] - x[1]);
  const maxHow = Math.max(1, ...how.map((h) => h[1]));
  const ti = d.meta.tracks.info || {};
  const covNote = ti.ball_coverage != null && ti.ball_last_time_s < 0.9 * d.meta.video.duration_s
    ? `<div class="notice small">Ball positions are only available up to ${fmtT(ti.ball_last_time_s)} of ${fmtT(d.meta.video.duration_s)} (${pct(ti.ball_coverage * 100)} of frames) in this source,
       so shots and points after that are not analysed. Run the Colab ball model on this video for full coverage.</div>` : "";
  el.innerHTML = `${covNote}
    ${s.scoreboard_points ? "" : `<div class="notice small">Point winners are inferred from ball and player tracking (no broadcast scoreboard imported). Correct any point in <b>Points & score</b>; the score and every statistic update.</div>`}
    <div class="grid g4">${statCard("Points", s.points)}${statCard("Shots in play", s.total_shots)}${statCard("Average rally", `${num(s.avg_rally_length)} shots`)}
      ${statCard("Longest rally", s.longest_rally ? `${s.longest_rally.shots} shots` : "–", s.longest_rally ? `at ${fmtT(s.longest_rally.time_s)}` : "")}</div>
    <div class="grid g2" style="margin-top:14px">
      <div class="card"><h2>Team comparison</h2><table><tr><th></th><th>${teamPill(d, "A")}</th><th>${teamPill(d, "B")}</th></tr>
        ${rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("")}</table></div>
      <div class="card"><h2>Momentum</h2>${momentumSvg(d)}<h3>How points ended</h3>
        ${how.map(([k, n]) => `<div class="row small"><span style="width:150px">${HOW[k] || k}</span><div class="bar" style="flex:1"><i style="width:${100 * n / maxHow}%"></i></div><span>${n}</span></div>`).join("")}
        <h3>Rule checks</h3><div class="small">${s.fouls_reported} infraction(s) flagged (likely + possible) – see <b>Rules & fouls</b>.</div></div>
    </div>
    ${ev ? `<div class="card" style="margin-top:14px"><h2>Accuracy vs. PadelTracker100 ground truth (annotated part)</h2>
      <div class="grid g4">${statCard("Hit detection F1", num(ev.hit_f1 * 100, 1) + "%", `precision ${num(ev.hit_precision * 100, 0)}% · recall ${num(ev.hit_recall * 100, 0)}%`)}
      ${statCard("Stroke accuracy", num(ev.stroke_accuracy * 100, 1) + "%", `${num(ev.stroke_accuracy_excl_other * 100, 1)}% excluding "Other"`)}
      ${statCard("Labelled shots", ev.gt_shots, `${ev.matched} matched`)}${statCard("Timing error", `${ev.timing_error_frames_median} frames`, "median")}</div></div>` : ""}`;
}

function momentumSvg(d) {
  const pts = d.points; if (!pts.length) return "";
  const W = 600, H = 140; let diff = 0; const ys = [0];
  for (const p of pts) { diff += p.winner === "A" ? 1 : p.winner === "B" ? -1 : 0; ys.push(diff); }
  const m = Math.max(3, ...ys.map(Math.abs));
  const x = (i) => (i / (ys.length - 1)) * W, y = (v) => H / 2 - (v / m) * (H / 2 - 8);
  const path = ys.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const games = pts.map((p, i) => (p.game_won ? `<line x1="${x(i + 1)}" x2="${x(i + 1)}" y1="0" y2="${H}" stroke="var(--line)"/>` : "")).join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${games}<line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="var(--line)"/>
    <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2"/><text x="4" y="12">${esc(tname(d, "A"))} ahead</text><text x="4" y="${H - 4}">${esc(tname(d, "B"))} ahead</text></svg>
    <div class="small muted">Point-by-point difference; vertical lines = games.</div>`;
}

/* ------------------------------------------------------------ video + HUD */
function tabVideo(d, el, name) {
  const fps = d.meta.video.fps, dur = d.meta.video.duration_s;
  el.innerHTML = `<div class="layout-video">
    <div><div class="video-wrap"><video id="vid" src="/video/${encodeURIComponent(name)}" controls preload="metadata"></video><canvas id="ov"></canvas></div>
      <div class="timeline" id="tl"></div>
      <div class="row small" style="margin-top:8px">
        <label><input type="checkbox" id="oSk" checked> skeletons</label><label><input type="checkbox" id="oBall" checked> ball</label>
        <label><input type="checkbox" id="oNames" checked> names</label>
        <button id="pPrev">◀ point</button><button id="sPrev">◀ shot</button><button id="sNext">shot ▶</button><button id="pNext">point ▶</button>
        <span class="muted">Timeline: points (bands) · shots (ticks) · rule flags (red)</span></div></div>
    <div><div class="card" id="hudPoint"></div><div class="card" id="hudShot" style="margin-top:14px"></div></div></div>`;
  const vid = $("#vid"), cv = $("#ov"), tl = $("#tl");
  // timeline
  const span = (t) => `${(100 * t / dur).toFixed(3)}%`;
  let tlHtml = d.points.map((p) => `<div style="position:absolute;top:0;height:100%;left:${span(p.start_s)};width:${span(Math.max(0.5, p.end_s - p.start_s))};background:${p.winner === "A" ? "var(--teamA)" : p.winner === "B" ? "var(--teamB)" : "var(--muted)"};opacity:.18"></div>`).join("");
  tlHtml += d.shots.filter((s) => s.point != null).map((s) => `<div class="tick" style="left:${span(s.time_s)};background:${SHOT_COLORS[s.shot_type] || "#888"};opacity:.7"></div>`).join("");
  tlHtml += d.fouls.filter((f) => f.status !== "review").map((f) => `<div class="tick" style="left:${span(f.time_s)};background:var(--bad);top:0;height:8px;width:4px"></div>`).join("");
  tl.innerHTML = tlHtml + `<div class="cursor" id="tlc"></div>`;
  tl.onclick = (ev) => { const r = tl.getBoundingClientRect(); vid.currentTime = ((ev.clientX - r.left) / r.width) * dur; };
  const shots = d.shots.filter((s) => s.point != null);
  const seekRel = (list, dir) => { const t = vid.currentTime + (dir > 0 ? 0.2 : -0.6); const c = dir > 0 ? list.find((x) => x > t) : [...list].reverse().find((x) => x < t); if (c != null) vid.currentTime = Math.max(0, c - (dir ? 0.4 : 0)); };
  $("#sNext").onclick = () => seekRel(shots.map((s) => s.time_s), 1);
  $("#sPrev").onclick = () => seekRel(shots.map((s) => s.time_s), -1);
  $("#pNext").onclick = () => seekRel(d.points.map((p) => p.start_s), 1);
  $("#pPrev").onclick = () => seekRel(d.points.map((p) => p.start_s), -1);
  if (state.seekTo != null) { const t = state.seekTo; state.seekTo = null; vid.addEventListener("loadedmetadata", () => { vid.currentTime = t; }, { once: true }); }

  const ctx = cv.getContext("2d");
  let lastHud = -1;
  async function getFrame(f) {
    const w = Math.floor(f / 300) * 300;
    if (!state.overlay.has(w)) {
      state.overlay.set(w, null);
      api(`/api/results/${encodeURIComponent(name)}/overlay?start=${w}&end=${w + 300}`).then((r) => state.overlay.set(w, r)).catch(() => state.overlay.delete(w));
    }
    const win = state.overlay.get(w);
    if (!win) return null;
    return win.frames[f - win.start] || null;
  }
  const teamCol = (slot) => getComputedStyle(document.documentElement).getPropertyValue((d.meta.tracks.slots[slot] || {}).team === "B" ? "--teamB" : "--teamA").trim();
  async function draw() {
    if (!document.body.contains(cv)) return;
    const f = Math.round(vid.currentTime * fps);
    cv.width = cv.clientWidth; cv.height = cv.clientHeight;
    const sx = cv.width / d.meta.video.width, sy = cv.height / d.meta.video.height;
    ctx.clearRect(0, 0, cv.width, cv.height);
    $("#tlc").style.left = span(vid.currentTime);
    const fr = await getFrame(f);
    const cur = shots.filter((s) => Math.abs(s.frame - f) <= 6)[0];
    if (fr) {
      if ($("#oSk").checked) fr.p.forEach((p, slot) => {
        if (!p) return;
        const col = teamCol(slot); const xs = p.slice(0, 34), vis = p.slice(34);
        ctx.strokeStyle = col; ctx.lineWidth = cur && cur.player === slot ? 4 : 2;
        for (const [a, b] of SKEL) if (vis[a] && vis[b]) { ctx.beginPath(); ctx.moveTo(xs[2 * a] * sx, xs[2 * a + 1] * sy); ctx.lineTo(xs[2 * b] * sx, xs[2 * b + 1] * sy); ctx.stroke(); }
        if ($("#oNames").checked && vis[0]) { ctx.fillStyle = col; ctx.font = "bold 12px system-ui"; ctx.fillText(pname(d, slot), xs[0] * sx - 20, xs[1] * sy - 22); }
      });
      if ($("#oBall").checked) {
        for (let k = 8; k >= 0; k--) {
          const g = await getFrame(f - k); if (!g || !g.b) continue;
          ctx.fillStyle = k ? `rgba(255,230,80,${0.12 + 0.08 * (8 - k)})` : "#ffe650";
          ctx.beginPath(); ctx.arc(g.b[0] * sx, g.b[1] * sy, k ? 3 : 6, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
    if (Math.abs(f - lastHud) > 2) { lastHud = f; updateHud(d, vid.currentTime); }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
}

function updateHud(d, t) {
  const p = d.points.find((p) => t >= p.start_s - 0.5 && t <= p.end_s + 2.5) || [...d.points].reverse().find((p) => p.start_s <= t);
  const hp = $("#hudPoint"), hs = $("#hudShot");
  if (!hp) return;
  if (p) {
    const sb = p.score_before || {};
    hp.innerHTML = `<h2>Point ${p.idx + 1} <span class="muted small">${esc(sb.games_text || "")} · ${esc(sb.label || "")}</span></h2>
      <div class="kv"><div>Server</div><div>${teamPill(d, p.server_team)} ${esc(pname(d, p.server))}</div>
      <div>Rally</div><div>${p.rally_length} shots · ${num(p.duration_s)} s</div>
      <div>Won by</div><div>${teamPill(d, p.winner)} <span class="small">(${HOW[p.how] || p.how}, ${p.source}, conf ${num(p.confidence * 100, 0)}%)</span></div>
      <div>Why</div><div class="small">${esc(p.reason)}</div></div>
      ${p.fouls.length ? `<h3>Rule flags</h3>${p.fouls.map((f) => `<div class="small"><span class="pill bad">${esc(f.rule)}</span> ${esc(f.name)}</div>`).join("")}` : ""}`;
  } else hp.innerHTML = `<div class="muted">Between points</div>`;
  const s = [...d.shots].reverse().find((s) => s.point != null && s.time_s <= t + 0.15 && t - s.time_s < 2.5);
  if (!s) { hs.innerHTML = `<div class="muted">No shot in progress</div>`; return; }
  const bm = s.biomech || {}, c = s.contact || {};
  hs.innerHTML = `<h2><span class="dot" style="background:${SHOT_COLORS[s.shot_type]}"></span> ${SHOT_NAMES[s.shot_type] || s.shot_type}</h2>
    <div class="kv"><div>Player</div><div>${teamPill(d, s.team)} ${esc(pname(d, s.player))}</div>
    <div>Shot</div><div>#${s.num_in_point} in point · ${esc(s.context.replace("_", " "))} · ${esc(s.stroke)}</div>
    <div>Position</div><div>${s.zone} zone, ${num(s.dist_from_net_m)} m from net, ${s.court_half} side</div>
    <div>Contact</div><div>${c.height_m != null ? num(c.height_m, 2) + " m high" : "–"} · ${c.on_dominant_side ? "forehand side" : "backhand side"} · ${esc(c.hand || "")}-handed</div>
    <div>Ball</div><div>${s.speed_kmh ? `${s.speed_kmh} km/h (avg to bounce)` : "–"}${s.landing ? ` · landed ${s.landing_zone} zone` : ""}</div>
    <div>Knee / elbow</div><div>${bm.knee_flexion_deg ? bm.knee_flexion_deg + "°" : "–"} / ${bm.elbow_deg ? bm.elbow_deg + "°" : "–"}</div>
    <div>Outcome</div><div>${esc(s.outcome.replace(/_/g, " "))}</div><div>Confidence</div><div>${num(s.confidence * 100, 0)}%</div></div>`;
}

function seekVideo(name, t) { state.seekTo = Math.max(0, t - 1.0); location.hash = `#/match/${encodeURIComponent(name)}/video`; }

/* ------------------------------------------------------------ shots */
function tabShots(d, el, name) {
  const types = [...new Set(d.shots.map((s) => s.shot_type))].sort();
  el.innerHTML = `<div class="card"><div class="row" style="margin-bottom:10px">
    <select id="fP"><option value="">All players</option>${d.meta.tracks.slots.map((m) => `<option value="${m.slot}">${esc(pname(d, m.slot))}</option>`).join("")}</select>
    <select id="fT"><option value="">All shot types</option>${types.map((t) => `<option value="${t}">${SHOT_NAMES[t] || t}</option>`).join("")}</select>
    <select id="fO"><option value="">All outcomes</option><option>winner</option><option>error</option><option>forced_error_induced</option><option>in_play</option><option>dead_ball</option></select>
    <span id="cnt" class="muted small"></span></div><div class="tbl-wrap"><table id="st"></table></div></div>`;
  const render = () => {
    const P = $("#fP").value, T = $("#fT").value, O = $("#fO").value;
    const rows = d.shots.filter((s) => (P === "" || String(s.player) === P) && (T === "" || s.shot_type === T) && (O === "" || s.outcome === O));
    $("#cnt").textContent = `${rows.length} shots`;
    $("#st").innerHTML = `<tr><th>Time</th><th>Point</th><th>Player</th><th>Shot</th><th>Context</th><th>Zone</th><th>Contact h</th><th>Speed</th><th>Outcome</th><th>Conf</th></tr>` +
      rows.slice(0, 1500).map((s) => `<tr class="click" data-t="${s.time_s}"><td>${fmtT(s.time_s)}</td><td>${s.point != null ? s.point + 1 : "–"}</td><td>${teamPill(d, s.team)} ${esc(pname(d, s.player))}</td>
        <td><span class="dot" style="background:${SHOT_COLORS[s.shot_type]}"></span> ${SHOT_NAMES[s.shot_type] || s.shot_type}</td><td>${esc(s.context.replace("_", " "))}</td>
        <td>${s.zone} (${num(s.dist_from_net_m)} m)</td><td>${s.contact && s.contact.height_m != null ? num(s.contact.height_m, 2) + " m" : "–"}</td><td>${s.speed_kmh ? s.speed_kmh + " km/h" : "–"}</td>
        <td>${s.outcome === "winner" ? '<span class="pill good">winner</span>' : s.outcome === "error" ? '<span class="pill bad">error</span>' : esc(s.outcome.replace(/_/g, " "))}</td><td>${num(s.confidence * 100, 0)}%</td></tr>`).join("");
    $("#st").querySelectorAll("tr.click").forEach((tr) => (tr.onclick = () => seekVideo(name, +tr.dataset.t)));
  };
  ["fP", "fT", "fO"].forEach((id) => ($("#" + id).onchange = render));
  render();
}

/* ------------------------------------------------------------ points */
function tabPoints(d, el, name) {
  el.innerHTML = `<div class="card"><p class="small muted">Click a row to watch the point. Use the selector to correct the winner – the score, statistics and reports are recomputed.</p>
    <div class="tbl-wrap"><table><tr><th>#</th><th>Time</th><th>Score before</th><th>Server</th><th>Rally</th><th>Won by</th><th>How</th><th>Why</th><th>Source</th></tr>
    ${d.points.map((p) => `<tr class="click" data-t="${p.start_s}"><td>${p.idx + 1}</td><td>${fmtT(p.start_s)}</td>
      <td>${esc((p.score_before || {}).games_text || "")} <b>${esc((p.score_before || {}).label || "")}</b>${p.game_won ? ` <span class="pill">game ${esc(tname(d, p.game_won))}</span>` : ""}</td>
      <td>${teamPill(d, p.server_team)}</td><td>${p.rally_length} · ${(p.shot_ids || []).slice(0, 14).map((i) => { const s = d.shotById.get(i); return s ? `<span class="chip" style="border-color:${SHOT_COLORS[s.shot_type]}">${(SHOT_NAMES[s.shot_type] || s.shot_type).split(" ")[0]}</span>` : ""; }).join("")}${p.shot_ids.length > 14 ? "…" : ""}</td>
      <td><select data-i="${p.idx}" class="pw" onclick="event.stopPropagation()"><option value="">auto</option><option value="A" ${p.winner === "A" ? "selected" : ""}>${esc(tname(d, "A"))}</option><option value="B" ${p.winner === "B" ? "selected" : ""}>${esc(tname(d, "B"))}</option></select></td>
      <td>${HOW[p.how] || p.how}</td><td class="small">${esc(p.reason)}</td><td><span class="pill ${p.source === "inferred" ? "warn" : "good"}">${p.source} ${num(p.confidence * 100, 0)}%</span></td></tr>`).join("")}</table></div></div>`;
  el.querySelectorAll("tr.click").forEach((tr) => (tr.onclick = () => seekVideo(name, +tr.dataset.t)));
  el.querySelectorAll("select.pw").forEach((s) => (s.onchange = async () => {
    try { const r = await post(`/api/results/${encodeURIComponent(name)}/points/${s.dataset.i}`, { winner: s.value || null }); watchJob(r.job, async () => { await loadMatch(name, true); route(); }); }
    catch (e) { toast(e.message); }
  }));
}

/* ------------------------------------------------------------ fouls */
function tabFouls(d, el, name) {
  el.innerHTML = `<div class="card"><div class="row" style="margin-bottom:10px"><select id="fs"><option value="lp">Likely + possible</option><option value="likely">Likely only</option><option value="all">All incl. review</option></select>
    <span class="muted small">Each check uses the FIP Rules of Padel (2026). Measurement margins are applied; "possible" means: watch the clip.</span></div><div id="fl"></div></div>`;
  const render = () => {
    const m = $("#fs").value;
    const list = d.fouls.filter((f) => m === "all" || (m === "likely" ? f.status === "likely" : f.status !== "review"));
    $("#fl").innerHTML = list.length ? list.map((f) => `<div class="finding click" data-t="${f.time_s}" style="cursor:pointer">
      <div class="row"><span class="t">${esc(f.name)}</span><span class="pill bad">${esc(f.rule)}</span><span class="pill ${f.status === "likely" ? "bad" : "warn"}">${f.status} ${num(f.confidence * 100, 0)}%</span>
        <span class="spacer"></span><span class="small">${fmtT(f.time_s)} · ${teamPill(d, f.team)} ${esc(pname(d, f.player))}${f.point != null ? ` · point ${f.point + 1}` : ""}</span></div>
      <div class="small">${esc(f.explanation)}</div><div class="small muted">Rule: ${esc(f.rule_text)}</div>
      <div class="drill">Correction: ${esc(f.correction)}</div></div>`).join("") : `<div class="muted">No infractions at this confidence level.</div>`;
    $("#fl").querySelectorAll(".click").forEach((x) => (x.onclick = () => seekVideo(name, +x.dataset.t)));
  };
  $("#fs").onchange = render; render();
}

/* ------------------------------------------------------------ players */
function findingHtml(f) {
  const dr = f.drill;
  return `<div class="finding ${f.polarity === "strength" ? "strength" : ""}"><div class="row"><span class="t">${esc(f.title)}</span>
    <span class="pill">${f.polarity === "strength" ? "strength" : `~${num(f.impact_points)} pts`}</span><span class="pill">conf ${num(f.confidence * 100, 0)}%</span></div>
    <div class="small">${esc(f.detail)}</div>${f.polarity !== "strength" ? `<div class="small"><b>Fix:</b> ${esc(f.recommendation)}</div>` : ""}
    ${dr && f.polarity !== "strength" ? `<div class="drill"><b>Drill – ${esc(dr.name)}:</b> ${esc(dr.setup)} <i>${esc(dr.dose)}</i>. Key: ${esc(dr.key)}</div>` : ""}</div>`;
}
function planHtml(plan) {
  return plan.map((w) => `<div class="finding strength"><div class="t">Week ${w.week}: ${esc(w.focus)}</div><div class="small">${esc(w.why)}</div>
    ${w.drill ? `<div class="drill"><b>${esc(w.drill.name)}</b> – ${esc(w.drill.setup)} <i>${esc(w.drill.dose)}</i></div>` : ""}<div class="small"><b>Goal:</b> ${esc(w.goal)}</div></div>`).join("");
}
function courtSvg(heat, landings, color) {
  const W = 200, H = 400, cw = W / 10, ch = H / 20;
  const mx = Math.max(1, ...heat.flat());
  let s = `<svg class="court-svg" viewBox="-4 -4 ${W + 8} ${H + 8}" style="width:100%;max-width:220px">`;
  heat.forEach((row, y) => row.forEach((v, x) => { if (v) s += `<rect class="cell" x="${x * cw}" y="${y * ch}" width="${cw}" height="${ch}" fill="${color}" opacity="${(0.08 + 0.85 * v / mx).toFixed(2)}"/>`; }));
  s += `<rect x="0" y="0" width="${W}" height="${H}" fill="none" stroke="var(--muted)"/><line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="var(--text)" stroke-width="2"/>
    <line x1="0" x2="${W}" y1="${3.05 * ch}" y2="${3.05 * ch}" stroke="var(--muted)"/><line x1="0" x2="${W}" y1="${16.95 * ch}" y2="${16.95 * ch}" stroke="var(--muted)"/>
    <line x1="${W / 2}" x2="${W / 2}" y1="${3.05 * ch}" y2="${16.95 * ch}" stroke="var(--muted)"/>`;
  for (const l of landings || []) s += `<circle cx="${l[0] * cw}" cy="${l[1] * ch}" r="2.2" fill="var(--warn)" opacity=".8"/>`;
  return s + `</svg>`;
}

function tabPlayers(d, el) {
  const slots = d.meta.tracks.slots;
  const sel = state.player ?? 0;
  const p = d.analysis.players[sel];
  const tcol = getComputedStyle(document.documentElement).getPropertyValue(p.team === "B" ? "--teamB" : "--teamA").trim();
  const st = p.shots, mv = p.movement || {}, bm = p.biomech || {}, ld = p.lob_defense || {};
  const types = Object.entries(st.by_type);
  const meta = slots[sel] || {};
  el.innerHTML = `<div class="row" style="margin-bottom:12px">${slots.map((m) => `<button data-s="${m.slot}" class="${m.slot === sel ? "primary" : ""}">${esc(pname(d, m.slot))}</button>`).join("")}</div>
    <div class="grid g4">${statCard("Shots", st.total, `${st.winners} winners · ${st.errors} errors`)}${statCard("Unforced errors", st.unforced_errors, `${st.forced_errors} forced`)}
      ${statCard("Distance covered", mv.distance_m ? `${Math.round(mv.distance_m)} m` : "–", mv.distance_per_rally_minute_m ? `${Math.round(mv.distance_per_rally_minute_m)} m per rally-minute` : "")}
      ${statCard("Handedness", esc(bm.dominant_hand || "–"), `${esc(meta.role || "")} side · est. height ${meta.height_m ? meta.height_m + " m" : "–"}`)}</div>
    <div class="grid g3" style="margin-top:14px">
      <div class="card"><h2>${teamPill(d, p.team)} ${esc(p.name)}</h2>
        <h3>Court position</h3><div class="row small"><span>Net ${pct(mv.zone_time_pct?.net)}</span><span>Transition ${pct(mv.zone_time_pct?.transition)}</span><span>Defence ${pct(mv.zone_time_pct?.defense)}</span></div>
        <div class="row" style="align-items:flex-start;margin-top:8px"><div>${courtSvg(mv.heatmap || Array.from({ length: 20 }, () => Array(10).fill(0)), [], tcol)}<div class="small muted">Position heatmap</div></div>
          <div>${courtSvg(Array.from({ length: 20 }, () => Array(10).fill(0)), st.landing_points, tcol)}<div class="small muted">Where the shots landed</div></div></div></div>
      <div class="card"><h2>Shot profile</h2><div class="tbl-wrap"><table><tr><th>Shot</th><th>n</th><th>Win</th><th>Err</th><th>Err %</th></tr>
        ${types.map(([t, v]) => `<tr><td><span class="dot" style="background:${SHOT_COLORS[t]}"></span> ${SHOT_NAMES[t] || t}</td><td>${v.n}</td><td>${v.winners}</td><td>${v.errors}</td><td>${pct(v.error_rate * 100)}</td></tr>`).join("")}</table></div>
        <h3>Overheads & lobs</h3><div class="kv"><div>Lobs</div><div>${pct(st.lob_pct)} of shots</div><div>Overheads</div><div>${pct(st.overhead_pct)} of shots</div>
          <div>Top smash (est.)</div><div>${st.top_smash_kmh ? st.top_smash_kmh + " km/h" : "–"}</div><div>When lobbed</div><div>${ld.times_lobbed} times · overhead ${pct(ld.answered_with_overhead_pct)} · won ${pct(ld.points_won_after_lob_pct)}</div></div></div>
      <div class="card"><h2>Biomechanics</h2><div class="kv"><div>Knee angle, low balls</div><div>${bm.knee_flexion_low_balls_deg ? bm.knee_flexion_low_balls_deg + "°" : "–"} <span class="muted small">(${bm.n_low_balls} balls; lower = deeper)</span></div>
        <div>Overhead elbow</div><div>${bm.overhead_elbow_deg ? bm.overhead_elbow_deg + "°" : "–"} <span class="muted small">(180° = full extension)</span></div>
        <div>Overhead contact height</div><div>${bm.overhead_contact_height_m ? bm.overhead_contact_height_m + " m" : "–"}</div>
        <div>Stance width</div><div>${bm.stance_width_m ? bm.stance_width_m + " m" : "–"}</div><div>Contact spacing</div><div>${bm.contact_spacing_m ? bm.contact_spacing_m + " m from body" : "–"}</div>
        <div>Avg / top speed</div><div>${num(mv.avg_speed_ms)} / ${num(mv.max_speed_ms)} m/s</div></div>
        <p class="small muted">2D estimates from a single broadcast camera.</p></div></div>
    <div class="grid g2" style="margin-top:14px"><div class="card"><h2>Weaknesses & strengths</h2>${p.findings.map(findingHtml).join("") || `<div class="muted">No significant findings.</div>`}</div>
      <div class="card"><h2>4-week development plan</h2>${planHtml(p.plan)}</div></div>`;
  el.querySelectorAll("button[data-s]").forEach((b) => (b.onclick = () => { state.player = +b.dataset.s; tabPlayers(d, el); }));
}

function tabScouting(d, el) {
  const sc = d.analysis.scouting;
  el.innerHTML = `<div class="grid g2">${["A", "B"].map((t) => `<div class="card"><h2>How to beat ${teamPill(d, t)}</h2>
    ${sc[t].game_plan.map((g) => `<div class="finding"><div class="t">${esc(g.title)}</div><div class="small">${esc(g.detail)}</div></div>`).join("") || `<div class="muted">Not enough evidence for a targeted plan.</div>`}
    <h3>Tendencies</h3>${sc[t].tendencies.map((x) => `<div class="small">• ${esc(x)}</div>`).join("")}
    <h3>Team weaknesses</h3>${d.analysis.teams[t].findings.map(findingHtml).join("") || `<div class="muted small">None significant.</div>`}</div>`).join("")}</div>`;
}

function tabOurs(d, el, name) {
  const ours = d.analysis.our_team;
  if (!ours) {
    el.innerHTML = `<div class="card"><h2>Which team is yours?</h2><p class="muted">Pick your team to get your own weaknesses, a training plan and a game plan against your opponents.</p>
      <div class="row">${["A", "B"].map((t) => `<button data-t="${t}">${esc(tname(d, t))}</button>`).join("")}</div></div>`;
    el.querySelectorAll("button[data-t]").forEach((b) => (b.onclick = async () => {
      const r = await post(`/api/results/${encodeURIComponent(name)}/settings`, { our_team: b.dataset.t });
      watchJob(r.job, async () => { await loadMatch(name, true); route(); });
    }));
    return;
  }
  const opp = ours.how_to_beat_opponents;
  el.innerHTML = `<div class="grid g2"><div class="card"><h2>Our weaknesses – ${esc(ours.name)}</h2>${ours.findings.filter((f) => f.polarity === "weakness").map(findingHtml).join("") || `<div class="muted">No significant weaknesses.</div>`}
      <h3>Strengths to keep</h3>${ours.findings.filter((f) => f.polarity === "strength").map(findingHtml).join("") || `<div class="muted small">–</div>`}</div>
    <div class="card"><h2>Game plan vs ${esc(opp.team_name)}</h2>${opp.game_plan.map((g) => `<div class="finding"><div class="t">${esc(g.title)}</div><div class="small">${esc(g.detail)}</div></div>`).join("")}
      <h3>Their tendencies</h3>${opp.tendencies.map((x) => `<div class="small">• ${esc(x)}</div>`).join("")}
      <h3>Team training plan (4 weeks)</h3>${planHtml(ours.plan)}</div></div>`;
}

function tabSettings(d, el, name) {
  const st = d.meta.settings || {};
  el.innerHTML = `<div class="card" style="max-width:720px"><h2>Names & match settings</h2>
    <div class="grid g2">${["A", "B"].map((t) => `<label>Team ${t} name<br><input type="text" data-team="${t}" value="${esc((st.team_names || {})[t] || d.meta.team_names[t] || "")}"></label>`).join("")}
    ${d.meta.tracks.slots.map((m) => `<label>${esc(m.team)} – ${esc(m.role)} player <span class="dot" style="background:${m.kit_color_bgr ? `rgb(${m.kit_color_bgr[2]},${m.kit_color_bgr[1]},${m.kit_color_bgr[0]})` : "#888"}"></span><br>
      <input type="text" data-slot="${m.slot}" value="${esc((st.names || {})[m.slot] || d.meta.names[m.slot] || "")}"></label>`).join("")}</div>
    <div class="row" style="margin-top:12px"><label>Our team <select id="ourT"><option value="">not set</option>${["A", "B"].map((t) => `<option value="${t}" ${st.our_team === t ? "selected" : ""}>${t}</option>`).join("")}</select></label>
      <label><input type="checkbox" id="gp" ${st.golden_point ? "checked" : ""}> Golden point scoring</label></div>
    <div class="row" style="margin-top:12px"><button class="primary" id="saveSt">Save & recompute</button></div>
    <h3>Tracking info</h3><div class="small muted">Source: ${esc(d.meta.tracks.source)} · calibration: ${esc(d.meta.calibration.source)} (error ${num((d.meta.calibration.reprojection_error_m || 0) * 100, 1)} cm) ·
      shot model: ${d.meta.models.shot_model ? "ONNX" : "rule-based"} · scoreboard: ${d.meta.scoreboard.available ? "imported" : "not imported"}</div></div>`;
  $("#saveSt").onclick = async () => {
    const names = {}, team_names = {};
    el.querySelectorAll("input[data-slot]").forEach((i) => (names[i.dataset.slot] = i.value.trim()));
    el.querySelectorAll("input[data-team]").forEach((i) => (team_names[i.dataset.team] = i.value.trim()));
    const r = await post(`/api/results/${encodeURIComponent(name)}/settings`, { names, team_names, our_team: $("#ourT").value || "", golden_point: $("#gp").checked });
    watchJob(r.job, async () => { await loadMatch(name, true); route(); });
  };
}
