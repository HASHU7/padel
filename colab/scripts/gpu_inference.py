"""Run perception for a full match on a Colab GPU and write files the local app imports.

  python colab_scripts/gpu_inference.py --video /content/drive/MyDrive/padel/videos/my_match.mp4 \
      --pose /content/models/padel_pose.pt --ball /content/models/padel_ball.pt --out /content/out

Outputs (import both in the app: Matches -> "Import Colab file"):
  <stem>.npz   detections (players + 17 keypoints + ball) in padelcoach RawDetections format
  <stem>.json  broadcast scoreboard timeline (if --scoreboard and easyocr available)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")  # bundle root (contains padelcoach/)
from padelcoach.court import CourtCalibration  # noqa: E402
from padelcoach.perception import DEFAULT_BALL_METHOD, MainViewChecker, select_ball  # noqa: E402
from padelcoach.scoreboard import parse_ocr_tokens  # noqa: E402
from padelcoach.tracks import RawDetections  # noqa: E402


def load_calib(video: Path, calib_dir: Path, w: int, h: int) -> CourtCalibration:
    p = calib_dir / f"{video.stem}.json"
    if p.exists():
        return CourtCalibration.load(p)
    print("[calib] no saved calibration for this video - using the broadcast preset."
          " Calibrate in the local app first if the camera differs, then copy workspace/calibrations/<video>.json.")
    return CourtCalibration.padeltracker_broadcast(w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--ball", required=True)
    ap.add_argument("--out", default="/content/out")
    ap.add_argument("--calib-dir", default="calibrations")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scoreboard", action="store_true")
    ap.add_argument("--ball-method", default=DEFAULT_BALL_METHOD)
    args = ap.parse_args()
    from ultralytics import YOLO

    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    calib = load_calib(video, Path(args.calib_dir), W, H)
    view = MainViewChecker(calib, W, H)
    pose_m, ball_m = YOLO(args.pose), YOLO(args.ball)
    raw = RawDetections(fps, n, W, H, source=f"colab-gpu:{video.name}")
    ocr = None
    samples = []
    raw_ocr = []
    if args.scoreboard:
        try:
            import easyocr
            ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
        except Exception as ex:
            print("[scoreboard] easyocr unavailable:", ex)
    roi = calib.scoreboard_roi
    prev_ball = None
    ball_cands = {}
    buf, idx = [], []
    view_ok = True

    def flush():
        nonlocal prev_ball
        if not buf:
            return
        pr = pose_m.predict(buf, imgsz=args.imgsz, conf=0.3, half=True, verbose=False)
        br = ball_m.predict(buf, imgsz=args.imgsz, conf=0.05, half=True, verbose=False)
        for f, p, b in zip(idx, pr, br):
            if p.keypoints is not None and len(p.boxes):
                kp = p.keypoints.data.cpu().numpy().astype(np.float32)
                xywh = p.boxes.xywh.cpu().numpy()
                bb = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], 1).astype(np.float32)
                raw.persons_kp[f], raw.persons_bbox[f] = kp, bb
                raw.persons_score[f] = p.boxes.conf.cpu().numpy().astype(np.float32)
            if len(b.boxes):   # keep the top-5 candidates; linked over time after the pass
                xy = b.boxes.xywh.cpu().numpy()[:, :2]
                cf = b.boxes.conf.cpu().numpy()
                top = np.argsort(-cf)[:5]
                ball_cands[f] = np.column_stack([xy[top], cf[top]]).astype(np.float32)
        buf.clear()
        idx.clear()

    for f in range(n):
        ok, img = cap.read()
        if not ok:
            break
        if f % 15 == 0:
            view_ok = view(img) >= 0.07
            if ocr is not None and roi:
                x, y, w, h = roi
                # small broadcast text: upscale 2x, restrict the alphabet, accept lower confidence
                crop = cv2.resize(img[y:y + h, x:x + w], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                res = ocr.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
                toks = [(t, float(np.mean([q[1] for q in bx])) / 2, float(np.mean([q[0] for q in bx])) / 2)
                        for bx, t, c in res if c > 0.15]
                r = parse_ocr_tokens(toks)
                if r:
                    samples.append({"frame": f, "rows": r["rows"], "names": r["names"]})
                if f % 150 == 0:   # keep raw OCR every 5 s for diagnosis
                    raw_ocr.append({"frame": f, "tokens": [[t, round(c, 2)] for _, t, c in res]})
        if not view_ok:
            continue
        buf.append(img)
        idx.append(f)
        if len(buf) == args.batch:
            flush()
        if f % 3000 == 0:
            print(f"{f}/{n} frames")
    flush()
    cap.release()
    fr = np.array(sorted(ball_cands), np.int32)
    np.savez_compressed(out / f"{video.stem}_ballcands.npz", frames=fr,
                        counts=np.array([len(ball_cands[f]) for f in fr], np.int16),
                        cands=np.concatenate([ball_cands[f] for f in fr]) if len(fr) else np.zeros((0, 3), np.float32))
    raw.ball = select_ball(ball_cands, n, args.ball_method)
    print(f"ball: {sum(len(v) for v in ball_cands.values())} candidates in {len(ball_cands)} frames -> "
          f"{int(np.isfinite(raw.ball[:, 0]).sum())} linked ball positions", flush=True)
    raw.save(out / f"{video.stem}.npz")
    print("saved", out / f"{video.stem}.npz")
    if samples:
        names = next((s["names"] for s in samples if all(s["names"])), ["Row 1", "Row 2"])
        tl = {"rows": names, "samples": [{"frame": s["frame"], "rows": s["rows"]} for s in samples], "source": "colab-easyocr"}
        tl["raw_ocr"] = raw_ocr
        (out / f"{video.stem}.json").write_text(json.dumps(tl))
        print("saved", out / f"{video.stem}.json", len(samples), "scoreboard readings")


if __name__ == "__main__":
    main()
