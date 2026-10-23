"""End-to-end orchestration.

Perception sources (mode):
  labels  - PadelTracker100 annotations for this video (exact poses + ball)
  import  - detections file produced by colab/padel_gpu_inference.ipynb
            (workspace/detections/<video>.npz)
  video   - local CPU inference with the ONNX models in models/
  auto    - labels > import > video
Results go to workspace/results/<video stem>/. User settings (names, our
team, golden point, point corrections) live in settings.json there and are
applied by `reanalyze`, which reuses the saved tracks (seconds, not minutes).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from . import config as C
from .analytics import analyze
from .ball import clean_ball
from .court import CourtCalibration, load_calibration_for
from .events import detect_events
from .fouls import detect_fouls
from .identity import resolve_identities
from .rally import build_points
from .scoreboard import scoreboard_points
from .shots import ShotModel, build_shots
from .sources import padeltracker as pt
from .tracks import RawDetections, Tracks
from .video import probe

DETECTIONS_DIR = C.WORKSPACE / "detections"
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

Progress = Optional[Callable[[float, str], None]]


def result_dir(video: Path, tag: Optional[str] = None) -> Path:
    """workspace/results/<stem>[@tag]; tagged results (e.g. evaluation runs on
    raw detections of a labelled video) do not replace the main result."""
    d = C.RESULTS_DIR / (Path(video).stem + (f"@{tag}" if tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_settings(video: Path) -> dict:
    p = result_dir(video) / "settings.json"
    base = {"names": {}, "team_names": {}, "our_team": None, "golden_point": False, "overrides": {}}
    if p.exists():
        base.update(json.loads(p.read_text()))
    return base


def save_settings(video: Path, settings: dict) -> None:
    (result_dir(video) / "settings.json").write_text(json.dumps(settings, indent=2))


def available_modes(video: Path) -> Dict[str, bool]:
    from .perception import models_status
    return {"labels": pt.has_labels(video.name), "import": (DETECTIONS_DIR / f"{video.stem}.npz").exists(),
            "video": models_status()["pose"]}


def _stage(progress: Progress, lo: float, hi: float) -> Progress:
    if progress is None:
        return None
    return lambda p, m: progress(lo + (hi - lo) * max(0.0, min(1.0, p)), m)


def get_raw(video: Path, mode: str, calib: CourtCalibration, progress: Progress = None) -> RawDetections:
    info = probe(video)
    modes = available_modes(video)
    if mode == "auto":
        mode = "labels" if modes["labels"] else ("import" if modes["import"] else "video")
    if mode == "labels":
        key = pt.dataset_for_video(video.name)
        raw = pt.load_raw(key, info.fps, info.width, info.height)
        if info.n_frames < raw.n_frames:  # e.g. the 407-frame sample clip of a labelled match
            raw = raw.crop(info.n_frames)
        return raw
    if mode == "import":
        return RawDetections.load(DETECTIONS_DIR / f"{video.stem}.npz")
    from .perception import run_perception
    raw = run_perception(video, calib, progress)
    raw.save(DETECTIONS_DIR / f"{video.stem}.npz")
    return raw


def run(video: Path, mode: str = "auto", progress: Progress = None, tag: Optional[str] = None) -> dict:
    """Full analysis from perception to report. Returns the analysis summary."""
    t0 = time.time()
    video = Path(video)
    info = probe(video)
    calib, calib_src = load_calibration_for(video, info.width, info.height)
    out = result_dir(video, tag)
    if progress:
        progress(0.01, "loading detections")
    raw = get_raw(video, mode, calib, _stage(progress, 0.01, 0.45))
    tracks = resolve_identities(raw, calib, video, _stage(progress, 0.45, 0.85))
    tracks.ball = clean_ball(tracks.ball, tracks.fps)
    bv = np.isfinite(tracks.ball[:, 0])
    tracks.info.update({"calibration": calib_src, "video": video.name, "mode": mode,
                        "reprojection_error_m": round(calib.reprojection_error_m, 3),
                        "ball_coverage": round(float(bv.mean()), 3),
                        "ball_last_time_s": round(float(np.where(bv)[0].max() / tracks.fps), 1) if bv.any() else 0.0,
                        "players_coverage": round(float(np.isfinite(tracks.feet[:, :, 0]).any(1).mean()), 3)})
    tracks.save(out / "tracks.npz")
    if progress:
        progress(0.86, "detecting ball events")
    events = detect_events(tracks, calib)
    (out / "events.json").write_text(json.dumps([e.to_dict() for e in events]))
    res = reanalyze(video, tracks=tracks, events_cache=events, progress=_stage(progress, 0.9, 1.0), tag=tag)
    res["timing_s"] = round(time.time() - t0, 1)
    meta_p = result_dir(video, tag) / "meta.json"
    meta = json.loads(meta_p.read_text())
    meta["timing_s"] = res["timing_s"]
    meta_p.write_text(json.dumps(meta, indent=2))
    return res


def reanalyze(video: Path, tracks: Optional[Tracks] = None, events_cache=None, progress: Progress = None,
              tag: Optional[str] = None) -> dict:
    """Shots -> points -> fouls -> analytics with the current settings."""
    from .events import BallEvent
    video = Path(video)
    out = result_dir(video, tag)
    info = probe(video)
    calib, calib_src = load_calibration_for(video, info.width, info.height)
    if tracks is None:
        tracks = Tracks.load(out / "tracks.npz")
    if "ball_coverage" not in tracks.info:
        bv = np.isfinite(tracks.ball[:, 0])
        tracks.info["ball_coverage"] = round(float(bv.mean()), 3)
        tracks.info["ball_last_time_s"] = round(float(np.where(bv)[0].max() / tracks.fps), 1) if bv.any() else 0.0
    if events_cache is None:
        evs = json.loads((out / "events.json").read_text())
        events_cache = [BallEvent(e["frame"], e["kind"], tuple(e["px"]), tuple(v if v is not None else np.nan for v in e["court"]),
                                  e["side"], e["player"], e["wall"], e["confidence"],
                                  e["reach_ratio"] if e["reach_ratio"] is not None else np.nan,
                                  tuple(e["v_in"]), tuple(e["v_out"])) for e in evs]
    events = events_cache
    settings = load_settings(video)
    model = ShotModel()
    if progress:
        progress(0.1, "classifying shots")
    shots = build_shots(tracks, calib, events, model)
    # first pass (inferred winners) to map scoreboard rows to teams
    points = build_points(shots, events, tracks, settings.get("golden_point", False))
    sb, sb_info = scoreboard_points(video, points, tracks.fps)
    if sb or settings.get("overrides"):
        shots = build_shots(tracks, calib, events, model)
        points = build_points(shots, events, tracks, settings.get("golden_point", False), scoreboard_points=sb,
                              overrides={int(k): v for k, v in settings.get("overrides", {}).items()})
    if progress:
        progress(0.5, "checking FIP rules")
    fouls = detect_fouls(tracks, calib, shots, points, events)
    names = {m["slot"]: m["name"] for m in tracks.slot_meta}
    names.update({int(k): v for k, v in settings.get("names", {}).items() if v})
    team_names = {"A": "Team A", "B": "Team B"}
    team_names.update(sb_info.get("team_names", {}))
    team_names.update({k: v for k, v in settings.get("team_names", {}).items() if v})
    if progress:
        progress(0.7, "building player and team reports")
    analysis = analyze(tracks, shots, points, fouls, names, team_names, settings.get("our_team"))

    evaluation = None
    key = pt.dataset_for_video(video.name)
    if key and pt.has_labels(video.name):
        from .evaluation import evaluate_against_labels
        evaluation = evaluate_against_labels(key, shots)

    meta = {"video": info.to_dict(), "calibration": {"source": calib_src, **calib.to_dict()},
            "tracks": {"source": tracks.source, "info": tracks.info, "slots": tracks.slot_meta},
            "names": names, "team_names": team_names, "settings": settings, "scoreboard": sb_info,
            "models": {"shot_model": model.available}, "evaluation": evaluation,
            "counts": {"events": len(events), "shots": len(shots), "points": len(points), "fouls": len(fouls)},
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "shots.json").write_text(json.dumps([s.to_dict() for s in shots], default=_np))
    (out / "points.json").write_text(json.dumps([p.to_dict() for p in points], default=_np))
    (out / "fouls.json").write_text(json.dumps(fouls, default=_np))
    (out / "analysis.json").write_text(json.dumps(analysis, default=_np))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_np))
    if progress:
        progress(1.0, "done")
    return {"result_dir": str(out), "summary": analysis["summary"], "evaluation": evaluation}


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
