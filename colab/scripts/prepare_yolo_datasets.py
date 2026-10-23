"""Build YOLO datasets (ball detection + 17-keypoint pose) from PadelTracker100.

Run on Colab:  python colab_scripts/prepare_yolo_datasets.py --videos /content/drive/MyDrive/padel/videos \
                   --labels labels --out /content/datasets
Validation = last 15 % of each match (temporal split, no leakage between
neighbouring frames).
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

MATCHES = ["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"]
# video frame = annotation frame + offset (men's pose/ball JSON starts 65 frames late)
FRAME_OFFSET = {"2022_BCN_FinalM_1": 65}
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def load_coco(p, offset=0):
    d = json.load(open(p, encoding="utf-8"))
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"] - 1 + offset, []).append(a)
    return d, by


def extract(video, frames, out_dir, quality=92):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(frames))
    cap = cv2.VideoCapture(str(video))
    i, j, saved = 0, 0, {}
    while j < len(frames):
        if i == frames[j]:
            ok, img = cap.read()
            if not ok:
                break
            p = out_dir / f"{video.stem}_{i:06d}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved[i] = p
            j += 1
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--out", default="/content/datasets")
    ap.add_argument("--ball-step", type=int, default=3)
    ap.add_argument("--pose-step", type=int, default=12)
    ap.add_argument("--only", choices=["ball", "pose", "both"], default="both")
    ap.add_argument("--test-match", default=None,
                    help="cross-match protocol: this whole match becomes the unseen TEST split; the other match "
                         "is split temporally into train (85%%) / val (15%%, early stopping only)")
    args = ap.parse_args()
    out = Path(args.out)
    random.seed(0)
    for m in MATCHES:
        video = Path(args.videos) / f"{m}.mp4"
        if not video.exists():
            print("missing video", video)
            continue
        n = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
        split_at = int(0.85 * n)
        off = FRAME_OFFSET.get(m, 0)
        _, balls = load_coco(Path(args.labels) / f"{m}_ball.json", off)
        _, poses = load_coco(Path(args.labels) / f"{m}_pose.json", off)
        # ---- ball
        with_ball = sorted(f for f in balls if f < n)[:: args.ball_step]
        no_ball = [f for f in range(0, n, 7) if f not in balls]
        neg = random.sample(no_ball, min(len(no_ball), len(with_ball) // 10))
        ball_frames = with_ball + neg if args.only in ("ball", "both") else []
        # ---- pose
        pose_frames = list(range(0, n, args.pose_step)) if args.only in ("pose", "both") else []
        ball_set, pose_set = set(ball_frames), set(pose_frames)
        saved = extract(video, ball_set | pose_set, out / "_frames" / m)
        for f, p in saved.items():
            if args.test_match:
                split = "test" if m == args.test_match else ("val" if f >= split_at else "train")
            else:
                split = "val" if f >= split_at else "train"
            if f in ball_set:
                d = out / "ball" / "images" / split
                d.mkdir(parents=True, exist_ok=True)
                (out / "ball" / "labels" / split).mkdir(parents=True, exist_ok=True)
                dst = d / p.name
                if not dst.exists():
                    dst.write_bytes(p.read_bytes())
                lines = []
                for a in balls.get(f, []):
                    x, y, w, h = a["bbox"]
                    w, h = max(w, 6), max(h, 6)  # tiny balls: keep a minimum box
                    lines.append(f"0 {(x + w / 2) / 1920:.6f} {(y + h / 2) / 1080:.6f} {w / 1920:.6f} {h / 1080:.6f}")
                (out / "ball" / "labels" / split / (p.stem + ".txt")).write_text("\n".join(lines))
            if f in pose_set and f in poses:
                d = out / "pose" / "images" / split
                d.mkdir(parents=True, exist_ok=True)
                (out / "pose" / "labels" / split).mkdir(parents=True, exist_ok=True)
                dst = d / p.name
                if not dst.exists():
                    dst.write_bytes(p.read_bytes())
                lines = []
                for a in poses[f]:
                    x, y, w, h = a["bbox"]
                    kp = np.array(a["keypoints"], float).reshape(17, 3)
                    vals = [f"{(x + w / 2) / 1920:.6f}", f"{(y + h / 2) / 1080:.6f}", f"{w / 1920:.6f}", f"{h / 1080:.6f}"]
                    for kx, ky, kv in kp:
                        vals += [f"{kx / 1920:.6f}", f"{ky / 1080:.6f}", "2" if kv > 0 else "0"]
                    lines.append("0 " + " ".join(vals))
                (out / "pose" / "labels" / split / (p.stem + ".txt")).write_text("\n".join(lines))
            p.unlink()
        print(m, "ball frames", len(ball_frames), "pose frames", len(pose_frames))
    test = "test: images/test\n" if args.test_match else ""
    (out / "ball.yaml").write_text(f"path: {out / 'ball'}\ntrain: images/train\nval: images/val\n{test}names:\n  0: ball\n")
    (out / "pose.yaml").write_text(f"path: {out / 'pose'}\ntrain: images/train\nval: images/val\n{test}kpt_shape: [17, 3]\n"
                                   f"flip_idx: {FLIP_IDX}\nnames:\n  0: player\n")
    print("datasets ready in", out)


if __name__ == "__main__":
    main()
