"""Command line interface.

  python -m padelcoach serve [--port 8000]          start the web app
  python -m padelcoach analyze VIDEO [--mode auto|labels|import|video]
  python -m padelcoach reanalyze VIDEO              re-run analytics with saved settings
  python -m padelcoach benchmark                    rebuild the pro benchmark from labelled matches
  python -m padelcoach videos                       list videos and available analysis modes
  python -m padelcoach export-training              build colab_bundle/padel_bundle.zip for the Colab notebooks
  python -m padelcoach evaluate-models              accuracy of imported Kaggle/Colab detections on the labelled finals
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config as C


def _progress(p, msg):
    sys.stdout.write(f"\r  {p:6.1%}  {msg:<40}")
    sys.stdout.flush()


def main(argv=None):
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="padelcoach")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.add_argument("--host", default="127.0.0.1")
    a = sub.add_parser("analyze"); a.add_argument("video"); a.add_argument("--mode", default="auto")
    r = sub.add_parser("reanalyze"); r.add_argument("video")
    sub.add_parser("benchmark")
    sub.add_parser("export-training")
    sub.add_parser("evaluate-models")
    sub.add_parser("videos")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn
        from .app.server import app
        print(f"Padel AI Coach running at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    elif args.cmd in ("analyze", "reanalyze"):
        from .pipeline import reanalyze, run
        from .video import find_video
        v = find_video(args.video)
        res = run(v, args.mode, _progress) if args.cmd == "analyze" else reanalyze(v, progress=_progress)
        print()
        print(json.dumps(res, indent=1, default=str))
    elif args.cmd == "benchmark":
        from .analytics import BENCHMARK_FILE, benchmark_from_results
        from .sources import padeltracker as pt
        dirs = [C.RESULTS_DIR / k for k in pt.DATASETS if (C.RESULTS_DIR / k / "analysis.json").exists()]
        if not dirs:
            print("Analyse the labelled PadelTracker100 matches first (python -m padelcoach analyze <video> --mode labels).")
            return 1
        bench = benchmark_from_results(dirs)
        BENCHMARK_FILE.write_text(json.dumps(bench, indent=1))
        print(json.dumps(bench, indent=1))
    elif args.cmd == "export-training":
        from .training_data import build_bundle
        build_bundle()
    elif args.cmd == "evaluate-models":
        from .pipeline import DETECTIONS_DIR, run
        from .eval_detections import evaluate_detections
        from .sources import padeltracker as pt
        from .tracks import RawDetections
        from .video import find_video
        report = {}
        for key in pt.DATASETS:
            det_p = DETECTIONS_DIR / f"{key}.npz"
            if not det_p.exists():
                print(f"no detections for {key} (expected {det_p})")
                continue
            v = find_video(f"{key}.mp4")
            det = RawDetections.load(det_p)
            gt = pt.load_raw(key)
            r = {"detections": evaluate_detections(det, gt)}
            print()
            print(f"{key}: detection accuracy {r['detections']}")
            res = run(v, "import", _progress, tag="models")
            print()
            r["end_to_end"] = {k: res["evaluation"][k] for k in ("hit_precision", "hit_recall", "hit_f1",
                                                              "stroke_accuracy", "stroke_accuracy_excl_other")}
            r["points"] = res["summary"]["points"]
            report[key] = r
            print(f"{key}: end-to-end {r['end_to_end']}")
        out = C.RESULTS_DIR / "model_evaluation.json"
        out.write_text(json.dumps(report, indent=1))
        print("saved", out)
    elif args.cmd == "videos":
        from .pipeline import available_modes
        from .video import list_videos
        for v in list_videos():
            print(v.name, available_modes(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
