import os, glob, time, json, shutil, subprocess, zipfile, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("nvidia-smi")
sh("pip install -q ultralytics")

INP = "/kaggle/input"
W = "/kaggle/working"
B = "/tmp/padel"
os.makedirs(B, exist_ok=True)
os.makedirs(f"{W}/detections", exist_ok=True)

# 1. Locate video and model weights
video_files = glob.glob(f"{INP}/**/the_wildest_match.mp4", recursive=True)
if not video_files:
    video_files = glob.glob(f"{INP}/**/*.mp4", recursive=True)
video_path = video_files[0]
print(f"Target Video: {video_path}", flush=True)

pose_weights = glob.glob(f"{INP}/**/pose_best.pt", recursive=True)[0]
ball_weights = glob.glob(f"{INP}/**/ball_best.pt", recursive=True)[0]
print(f"Pose Weights: {pose_weights}", flush=True)
print(f"Ball Weights: {ball_weights}", flush=True)

# 2. Extract package from padel-coach-data
z = glob.glob(f"{INP}/**/padel_bundle.zip", recursive=True)
if z:
    zipfile.ZipFile(z[0]).extractall(B)
else:
    src_dirs = glob.glob(f"{INP}/**/padelcoach", recursive=True)
    if src_dirs:
        shutil.copytree(os.path.dirname(src_dirs[0]), B, dirs_exist_ok=True)

import sys
sys.path.insert(0, B)
from padelcoach.tracks import RawDetections
from ultralytics import YOLO

cap = cv2.VideoCapture(video_path)
fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video metadata: {width}x{height} @ {fps} fps, {n_frames} frames", flush=True)

pose_m = YOLO(pose_weights)
ball_m = YOLO(ball_weights)
raw = RawDetections(fps, n_frames, width, height, source=f"kaggle-gpu:{os.path.basename(video_path)}")

batch_size = 16
buf, idx = [], []
ball_cands = {}

t0 = time.time()
print("Starting GPU inference on Tesla T4...", flush=True)

def flush():
    if not buf:
        return
    pr = pose_m.predict(buf, imgsz=1280, conf=0.25, verbose=False)
    br = ball_m.predict(buf, imgsz=1280, conf=0.05, verbose=False)
    for f, p, b in zip(idx, pr, br):
        if p.keypoints is not None and len(p.boxes):
            kp = p.keypoints.data.cpu().numpy().astype(np.float32)
            xywh = p.boxes.xywh.cpu().numpy()
            bb = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]], 1).astype(np.float32)
            raw.persons_kp[f], raw.persons_bbox[f] = kp, bb
            raw.persons_score[f] = p.boxes.conf.cpu().numpy().astype(np.float32)
        if len(b.boxes):
            xy = b.boxes.xywh.cpu().numpy()[:, :2]
            cf = b.boxes.conf.cpu().numpy()
            top = np.argsort(-cf)[:5]
            ball_cands[f] = np.column_stack([xy[top], cf[top]]).astype(np.float32)
    buf.clear()
    idx.clear()

for f in range(n_frames):
    ok, img = cap.read()
    if not ok:
        break
    buf.append(img)
    idx.append(f)
    if len(buf) == batch_size:
        flush()
    if f > 0 and f % 2000 == 0:
        elapsed = time.time() - t0
        fps_proc = f / elapsed
        eta_s = (n_frames - f) / max(fps_proc, 1)
        print(f"  Frame {f}/{n_frames} ({f/n_frames*100:.1f}%) - {fps_proc:.1f} FPS - ETA: {eta_s:.0f}s", flush=True)

flush()
cap.release()

total_time = time.time() - t0
print(f"GPU Inference Finished: {n_frames} frames in {total_time:.1f}s ({n_frames/total_time:.1f} FPS average)", flush=True)

# Self-contained ball tracklet linking algorithm
def link_candidates(cands, n_frames, max_gap=5, single_conf=0.45, min_conf=0.1, static_px=25, min_mean_conf=0.15):
    active, done = [], []
    for f in range(n_frames):
        c = cands.get(f)
        if c is not None and min_conf > 0:
            c = c[c[:, 2] >= min_conf]
        used = set()
        if c is not None and len(c):
            for t in sorted(active, key=lambda t: -sum(t["s"])):
                gap = f - t["f"][-1]
                if gap > max_gap:
                    continue
                pred = t["p"][-1] + t["v"] * gap
                d = np.hypot(*(c[:, :2] - pred).T)
                gate = 30 + 0.5 * np.hypot(*t["v"]) * gap + 10 * gap
                for j in np.argsort(d):
                    if j in used or d[j] > gate:
                        continue
                    used.add(j)
                    nv = (c[j, :2] - t["p"][-1]) / gap
                    t["v"] = 0.5 * t["v"] + 0.5 * nv if len(t["f"]) > 1 else nv
                    t["f"].append(f); t["p"].append(c[j, :2].copy()); t["s"].append(float(c[j, 2]))
                    break
            for j in range(len(c)):
                if j not in used:
                    active.append({"f": [f], "p": [c[j, :2].copy()], "s": [float(c[j, 2])], "v": np.zeros(2)})
        still = []
        for t in active:
            (still if f - t["f"][-1] <= max_gap else done).append(t)
        active = still
    done += active
    scored = []
    for t in done:
        n = len(t["f"])
        if n < 3 and max(t["s"]) < single_conf:
            continue
        if min_mean_conf and float(np.mean(t["s"])) < min_mean_conf:
            continue
        if static_px and n >= 8:
            P = np.array(t["p"])
            if np.hypot(*(P.max(0) - P.min(0))) < static_px:
                continue
        scored.append((float(np.sum(t["s"])) * (1.0 + 0.15 * min(n, 10)), t))
    scored.sort(key=lambda st: -st[0])
    ball = np.full((n_frames, 3), np.nan, np.float32)
    for _, t in scored:
        fr = np.array(t["f"])
        if np.isnan(ball[fr, 0]).mean() < 0.6:
            continue
        for f, p, s in zip(t["f"], t["p"], t["s"]):
            if np.isnan(ball[f, 0]):
                ball[f] = (p[0], p[1], s)
    return ball

raw.ball = link_candidates(ball_cands, n_frames)
valid_ball = int(np.isfinite(raw.ball[:, 0]).sum())
print(f"Ball track linked. Valid ball frames: {valid_ball}/{n_frames} ({valid_ball/n_frames*100:.1f}%)", flush=True)

out_npz = f"{W}/the_wildest_match.npz"
raw.save(out_npz)
print(f"SUCCESS: Saved RawDetections to {out_npz} (size: {os.path.getsize(out_npz) / (1024*1024):.2f} MB)", flush=True)

# Also copy to detections/ directory
shutil.copy(out_npz, f"{W}/detections/the_wildest_match.npz")

summary = {
    "video": os.path.basename(video_path),
    "n_frames": n_frames,
    "fps": fps,
    "total_time_s": round(total_time, 2),
    "avg_fps": round(n_frames / total_time, 2),
    "persons_frames_detected": len(raw.persons_kp),
    "ball_frames_detected": valid_ball,
    "ball_coverage_pct": round(float(valid_ball / n_frames) * 100, 2),
}
with open(f"{W}/inference_summary.json", "w") as fp:
    json.dump(summary, fp, indent=2)
print("INFERENCE COMPLETED SUCCESSFULLY!")
print(json.dumps(summary, indent=2), flush=True)
