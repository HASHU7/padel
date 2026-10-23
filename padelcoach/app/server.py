"""FastAPI backend for the Padel AI Coach web app (local, single user)."""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from .. import config as C
from ..court import CourtCalibration, calibration_path_for, load_calibration_for
from ..perception import models_status
from ..pipeline import (DETECTIONS_DIR, _np, available_modes, load_settings, reanalyze, result_dir, run,
                        save_settings)
from ..scoreboard import timeline_path
from ..tracks import Tracks
from ..video import find_video, list_videos, probe, read_frame

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="Padel AI Coach", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

JOBS: Dict[str, dict] = {}
_job_lock = threading.Lock()
_tracks_cache: Dict[str, tuple] = {}


def _video(name: str) -> Path:
    try:
        return find_video(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Video not found: {name}")


def _start_job(kind: str, video: Path, fn) -> str:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "kind": kind, "video": video.name, "status": "queued", "progress": 0.0,
                 "message": "waiting", "started": time.time(), "error": None, "result": None}

    def work():
        with _job_lock:  # one heavy job at a time on a laptop CPU
            JOBS[jid]["status"] = "running"
            try:
                def prog(p, m):
                    JOBS[jid]["progress"] = round(float(p), 4)
                    JOBS[jid]["message"] = m
                res = fn(prog)
                res = json.loads(json.dumps(res, default=_np))  # numpy -> plain JSON types
                JOBS[jid].update(status="done", progress=1.0, message="done", result=res)
                _tracks_cache.pop(video.name, None)
            except Exception as ex:  # report to UI
                JOBS[jid].update(status="error", error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc())
            JOBS[jid]["elapsed_s"] = round(time.time() - JOBS[jid]["started"], 1)

    threading.Thread(target=work, daemon=True).start()
    return jid


# --------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return {"version": __version__, "models": models_status(),
            "jobs": [j for j in JOBS.values() if j["status"] in ("queued", "running")]}


@app.get("/api/videos")
def videos():
    out = []
    for v in list_videos():
        try:
            info = probe(v).to_dict()
        except IOError:
            continue
        rd = C.RESULTS_DIR / v.stem
        meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").exists() else None
        out.append({"name": v.name, "info": info, "modes": available_modes(v),
                    "calibrated": calibration_path_for(v).exists(), "scoreboard": timeline_path(v).exists(),
                    "analysed": meta is not None, "generated_at": meta.get("generated_at") if meta else None,
                    "counts": meta.get("counts") if meta else None})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in C.VIDEO_EXTENSIONS:
        raise HTTPException(400, "Unsupported video type")
    dst = C.VIDEOS_DIR / name
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f, length=16 * 1024 * 1024)
    return {"name": name}


@app.post("/api/videos/{name}/import")
async def import_file(name: str, file: UploadFile = File(...)):
    """Import Colab outputs: detections (.npz) or scoreboard timeline (.json)."""
    v = _video(name)
    suffix = Path(file.filename).suffix.lower()
    if suffix == ".npz":
        dst = DETECTIONS_DIR / f"{v.stem}.npz"
    elif suffix == ".json":
        dst = timeline_path(v)
    else:
        raise HTTPException(400, "Expected a detections .npz or a scoreboard .json")
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"saved": str(dst.name)}


@app.get("/video/{name}")
def stream_video(name: str):
    return FileResponse(_video(name), media_type="video/mp4")


@app.get("/api/videos/{name}/frame")
def frame(name: str, f: int = 0, w: int = 1280):
    img = read_frame(_video(name), f)
    if img is None:
        raise HTTPException(404, "frame not available")
    if w and img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


# --------------------------------------------------------------- calibration
class CalibIn(BaseModel):
    image_points: Dict[str, list]
    scoreboard_roi: Optional[list] = None


@app.get("/api/videos/{name}/calibration")
def get_calibration(name: str):
    v = _video(name)
    info = probe(v)
    calib, src = load_calibration_for(v, info.width, info.height)
    d = calib.to_dict()
    d.update(source=src, landmarks=C.COURT_LANDMARKS,
             court_lines=[[a.tolist(), b.tolist()] for a, b in calib.court_lines_px()])
    return d


@app.post("/api/videos/{name}/calibration")
def set_calibration(name: str, body: CalibIn):
    v = _video(name)
    info = probe(v)
    pts = {k: tuple(map(float, p)) for k, p in body.image_points.items() if k in C.COURT_LANDMARKS and p}
    try:
        calib = CourtCalibration(pts, info.width, info.height,
                                 tuple(int(x) for x in body.scoreboard_roi) if body.scoreboard_roi else None)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if calib.reprojection_error_m > 0.5:
        raise HTTPException(400, f"Points are inconsistent (error {calib.reprojection_error_m:.2f} m). Check the landmark names.")
    calib.save(calibration_path_for(v))
    d = calib.to_dict()
    d["court_lines"] = [[a.tolist(), b.tolist()] for a, b in calib.court_lines_px()]
    return d


@app.post("/api/videos/{name}/calibration/preview")
def preview_calibration(name: str, body: CalibIn):
    v = _video(name)
    info = probe(v)
    pts = {k: tuple(map(float, p)) for k, p in body.image_points.items() if k in C.COURT_LANDMARKS and p}
    try:
        calib = CourtCalibration(pts, info.width, info.height)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return {"reprojection_error_m": calib.reprojection_error_m,
            "court_lines": [[a.tolist(), b.tolist()] for a, b in calib.court_lines_px()]}


# ------------------------------------------------------------------ analysis
class AnalyzeIn(BaseModel):
    mode: str = "auto"


@app.post("/api/videos/{name}/analyze")
def analyze_video(name: str, body: AnalyzeIn):
    v = _video(name)
    if body.mode not in ("auto", "labels", "import", "video"):
        raise HTTPException(400, "bad mode")
    return {"job": _start_job("analyze", v, lambda prog: run(v, body.mode, prog))}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    j = dict(JOBS[jid])
    j.pop("trace", None)
    return j


@app.get("/api/results/{name}")
def results(name: str):
    v = _video(name)
    rd = C.RESULTS_DIR / v.stem
    if not (rd / "analysis.json").exists():
        raise HTTPException(404, "Not analysed yet")
    parts = {}
    for k in ("meta", "analysis", "points", "fouls", "shots"):
        parts[k] = json.loads((rd / f"{k}.json").read_text())
    return JSONResponse(parts)


@app.get("/api/results/{name}/events")
def events(name: str):
    rd = C.RESULTS_DIR / _video(name).stem
    return JSONResponse(json.loads((rd / "events.json").read_text()))


def _tracks(name: str) -> Tracks:
    v = _video(name)
    p = C.RESULTS_DIR / v.stem / "tracks.npz"
    if not p.exists():
        raise HTTPException(404, "Not analysed yet")
    mt = p.stat().st_mtime
    hit = _tracks_cache.get(name)
    if hit and hit[0] == mt:
        return hit[1]
    tr = Tracks.load(p)
    _tracks_cache[name] = (mt, tr)
    return tr


@app.get("/api/results/{name}/overlay")
def overlay(name: str, start: int = 0, end: int = 300):
    """Per-frame skeletons + ball for frames [start, end) (max 900 frames)."""
    tr = _tracks(name)
    start = max(0, start)
    end = min(tr.n_frames, max(start + 1, min(end, start + 900)))
    kp = tr.kp[start:end]
    ball = tr.ball[start:end]
    frames = []
    for i in range(end - start):
        pl = []
        for s in range(kp.shape[1]):
            k = kp[i, s]
            if not np.isfinite(k[0, 0]):
                pl.append(None)
                continue
            pl.append(np.round(np.nan_to_num(k[:, :2], nan=-1)).astype(int).ravel().tolist() +
                      [int(c > 0.25) for c in np.nan_to_num(k[:, 2])])
        b = ball[i]
        frames.append({"p": pl, "b": None if not np.isfinite(b[0]) else [int(b[0]), int(b[1])]})
    return {"start": start, "end": end, "fps": tr.fps, "frames": frames, "width": tr.width, "height": tr.height}


class SettingsIn(BaseModel):
    names: Optional[Dict[str, str]] = None
    team_names: Optional[Dict[str, str]] = None
    our_team: Optional[str] = None
    golden_point: Optional[bool] = None


@app.get("/api/results/{name}/settings")
def get_settings(name: str):
    return load_settings(_video(name))


@app.post("/api/results/{name}/settings")
def set_settings(name: str, body: SettingsIn):
    v = _video(name)
    st = load_settings(v)
    for k in ("names", "team_names"):
        if getattr(body, k) is not None:
            st[k] = {kk: vv for kk, vv in getattr(body, k).items()}
    if body.our_team is not None:
        st["our_team"] = body.our_team if body.our_team in ("A", "B") else None
    if body.golden_point is not None:
        st["golden_point"] = bool(body.golden_point)
    save_settings(v, st)
    return {"job": _start_job("reanalyze", v, lambda prog: reanalyze(v, progress=prog))}


class PointIn(BaseModel):
    winner: Optional[str] = None


@app.post("/api/results/{name}/points/{idx}")
def set_point(name: str, idx: int, body: PointIn):
    v = _video(name)
    st = load_settings(v)
    ov = st.setdefault("overrides", {})
    if body.winner in ("A", "B"):
        ov[str(idx)] = body.winner
    else:
        ov.pop(str(idx), None)
    save_settings(v, st)
    return {"job": _start_job("reanalyze", v, lambda prog: reanalyze(v, progress=prog))}


@app.get("/api/results/{name}/report.html", response_class=HTMLResponse)
def report_html(name: str):
    from ..report import render_report
    v = _video(name)
    return render_report(C.RESULTS_DIR / v.stem)
