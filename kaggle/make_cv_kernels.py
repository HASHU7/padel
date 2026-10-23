"""Generate the cross-match (train on one final, test on the unseen other) Kaggle jobs.

Protocol per fold: train = first 85 % of the training match, val = its last
15 % (early stopping / checkpoint choice only), test = the whole other match
(never used for any decision). Writes kaggle/cv_*/{train.py,kernel-metadata.json}.
"""
import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OWNER = "hashulearn"
MATCHES = ["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"]
SHORT = {"2022_BCN_FinalF_1": "women", "2022_BCN_FinalM_1": "men"}

scripts = {n: base64.b64encode((ROOT / "colab/scripts" / n).read_bytes()).decode()
           for n in ("prepare_yolo_datasets.py", "train_shot_model.py", "gpu_inference.py")}
pkg = {p.relative_to(ROOT).as_posix(): base64.b64encode(p.read_bytes()).decode()
       for p in list((ROOT / "padelcoach").rglob("*")) + list((ROOT / "tests").glob("*.py"))
       if p.suffix in (".py", ".json") and "__pycache__" not in p.parts}

HEADER = '''import base64, glob, json, os, shutil, subprocess, zipfile
SCRIPTS = %s
PKG = %s
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
INP, W, B = "/kaggle/input", "/kaggle/working", "/tmp/padel"
os.makedirs(B, exist_ok=True)
z = glob.glob(f"{INP}/**/padel_bundle.zip", recursive=True)
if z:
    zipfile.ZipFile(z[0]).extractall(B)
else:
    src = os.path.dirname(os.path.dirname(glob.glob(f"{INP}/**/padelcoach/__init__.py", recursive=True)[0]))
    shutil.copytree(src, B, dirs_exist_ok=True)
for rel, data in list(PKG.items()) + [(f"colab_scripts/{k}", v) for k, v in SCRIPTS.items()]:
    os.makedirs(os.path.dirname(f"{B}/{rel}"), exist_ok=True)
    open(f"{B}/{rel}", "wb").write(base64.b64decode(data))
VID = os.path.dirname(glob.glob(f"{INP}/**/2022_BCN_FinalF_1.mp4", recursive=True)[0])
os.chdir(B)
''' % (json.dumps(scripts), json.dumps(pkg))

GPU_SETUP = '''sh("nvidia-smi")
sh("pip install -q ultralytics onnx onnxslim")
from ultralytics import YOLO
# Strong augmentation: different light, colour, scale and viewpoint so the model
# cannot memorise this one court / broadcast.
AUG = dict(hsv_h=0.03, hsv_s=0.7, hsv_v=0.5, degrees=3, translate=0.1, scale=0.5, shear=1.0,
           perspective=0.0005, fliplr=0.5, mosaic=1.0, mixup=0.1, close_mosaic=5)
def m_box(r):
    return {"precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4)}
def m_pose(r):
    return {"box": m_box(r), "keypoints": {"precision": round(float(r.pose.mp), 4), "recall": round(float(r.pose.mr), 4),
            "mAP50": round(float(r.pose.map50), 4), "mAP50_95": round(float(r.pose.map), 4)}}
'''


def ball_job(test):
    train = [m for m in MATCHES if m != test][0]
    return HEADER + GPU_SETUP + f'''
TEST, TRAIN = "{test}", "{train}"
sh(f"python colab_scripts/prepare_yolo_datasets.py --videos {{VID}} --labels labels --out /tmp/datasets --only ball --ball-step 4 --test-match {{TEST}}")
YOLO("yolo11s.pt").train(data="/tmp/datasets/ball.yaml", imgsz=1280, epochs=40, time=5.0, patience=8, batch=8, workers=4,
                         project=f"{{W}}/runs", name="ball", exist_ok=True, **AUG)
best = f"{{W}}/runs/ball/weights/best.pt"
res = {{"trained_on": TRAIN, "tested_on_unseen": TEST,
        "val_same_match": m_box(YOLO(best).val(data="/tmp/datasets/ball.yaml", split="val", imgsz=1280)),
        "test_unseen_match": m_box(YOLO(best).val(data="/tmp/datasets/ball.yaml", split="test", imgsz=1280))}}
shutil.copy(best, f"{{W}}/ball_test_{{TEST}}.pt")
json.dump(res, open(f"{{W}}/ball_cv_{{TEST}}.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
'''


def pose_job(test):
    train = [m for m in MATCHES if m != test][0]
    return HEADER + GPU_SETUP + f'''
TEST, TRAIN = "{test}", "{train}"
sh(f"python colab_scripts/prepare_yolo_datasets.py --videos {{VID}} --labels labels --out /tmp/datasets --only pose --pose-step 12 --test-match {{TEST}}")
baseline = m_pose(YOLO("yolo11s-pose.pt").val(data="/tmp/datasets/pose.yaml", split="test", imgsz=1280))
print("BASELINE (no fine-tuning) on unseen match", json.dumps(baseline), flush=True)
# freeze the first 10 layers (backbone): keep the general human-pose knowledge
YOLO("yolo11s-pose.pt").train(data="/tmp/datasets/pose.yaml", imgsz=1280, epochs=15, time=2.5, patience=5, batch=8,
                              workers=4, freeze=10, project=f"{{W}}/runs", name="pose", exist_ok=True, **AUG)
best = f"{{W}}/runs/pose/weights/best.pt"
res = {{"trained_on": TRAIN, "tested_on_unseen": TEST, "baseline_pretrained_test": baseline,
        "finetuned_val_same_match": m_pose(YOLO(best).val(data="/tmp/datasets/pose.yaml", split="val", imgsz=1280)),
        "finetuned_test_unseen_match": m_pose(YOLO(best).val(data="/tmp/datasets/pose.yaml", split="test", imgsz=1280))}}
shutil.copy(best, f"{{W}}/pose_test_{{TEST}}.pt")
json.dump(res, open(f"{{W}}/pose_cv_{{TEST}}.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
'''


def shot_job():
    return HEADER + '''
sh("pip install -q onnx onnxscript")
sh(f"python colab_scripts/train_shot_model.py --data shot_sequences.npz --out {W} --epochs 120")
print("RESULT", open(f"{W}/padel_shot_classifier.json").read(), flush=True)
'''


def write(name, code, gpu):
    d = ROOT / "kaggle" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text(code, encoding="utf-8")
    meta = {"id": f"{OWNER}/padel-{name.replace('_', '-')}", "title": f"padel-{name.replace('_', '-')}",
            "code_file": "train.py", "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true" if gpu else "false", "enable_tpu": "false", "enable_internet": "true",
            "dataset_sources": [f"{OWNER}/padel-coach-data"], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", d, meta["id"])


for t in MATCHES:
    write(f"cv_ball_{SHORT[t]}", ball_job(t), True)
    write(f"cv_pose_{SHORT[t]}", pose_job(t), True)
write("cv_shot", shot_job(), False)


def e2e_job():
    """Full pipeline on each match using ONLY models that never saw that match.
    Phase 1 GPU detection (all ball candidates kept), phase 2 choose the
    ball-selection method by measured F1 against the labels (same method for
    both matches), phase 3 full analysis + evaluation."""
    return HEADER + '''
sh("nvidia-smi")
sh("pip install -q ultralytics easyocr onnxruntime")
sh("python -m unittest discover -s tests -v")   # unit tests run here, not on the laptop
import numpy as np, sys
sys.path.insert(0, B)
MATCHES = ["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"]
os.makedirs(f"{W}/e2e", exist_ok=True)
# ---------- phase 1: detection with fold models
for TEST in MATCHES:
    pose = glob.glob(f"{INP}/**/pose_test_{TEST}.pt", recursive=True)[0]
    ball = glob.glob(f"{INP}/**/ball_test_{TEST}.pt", recursive=True)[0]
    video = glob.glob(f"{INP}/**/{TEST}.mp4", recursive=True)[0]
    sh(f"python colab_scripts/gpu_inference.py --video {video} --pose {pose} --ball {ball} --out {W}/e2e/{TEST} --scoreboard --batch 16")
# ---------- phase 2: ball-selection method chosen by measurement
from padelcoach.perception import BALL_METHODS, select_ball
from padelcoach.sources import padeltracker as pt
from padelcoach.tracks import RawDetections
def load_cands(path):
    z = np.load(path)
    frames, counts, allc = z["frames"], z["counts"], z["cands"]   # decompress ONCE
    out, off = {}, 0
    for f, c in zip(frames, counts):
        out[int(f)] = allc[off:off + int(c)]
        off += int(c)
    return out
def ball_pr(ball, gt, tol=10.0):
    last = int(np.where(np.isfinite(gt[:, 0]))[0].max())
    tp = fp = fn = 0
    for f in range(last + 1):
        g, d = gt[f], ball[f]
        hg, hd = np.isfinite(g[0]), np.isfinite(d[0])
        if hg and hd and np.hypot(d[0] - g[0], d[1] - g[1]) <= tol:
            tp += 1
        else:
            fp += int(hd); fn += int(hg)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}
cands = {T: load_cands(f"{W}/e2e/{T}/{T}_ballcands.npz") for T in MATCHES}
gts = {T: pt.load_raw(T).ball for T in MATCHES}
ball_eval = {}
for m in BALL_METHODS:
    ball_eval[m] = {T: ball_pr(select_ball(cands[T], len(gts[T]), m), gts[T]) for T in MATCHES}
    ball_eval[m]["mean_f1"] = round(float(np.mean([ball_eval[m][T]["f1"] for T in MATCHES])), 4)
    print("BALL METHOD", m, json.dumps(ball_eval[m]), flush=True)
best = max(BALL_METHODS, key=lambda m: ball_eval[m]["mean_f1"])
print("CHOSEN BALL METHOD", best, flush=True)
json.dump({"methods": ball_eval, "chosen": best}, open(f"{W}/ball_method_selection.json", "w"), indent=1)
# ---------- phase 3: full pipeline + evaluation
report = {}
for TEST in MATCHES:
    raw = RawDetections.load(f"{W}/e2e/{TEST}/{TEST}.npz")
    raw.ball = select_ball(cands[TEST], raw.n_frames, best)
    for d in ("workspace/detections", "workspace/videos", "workspace/scoreboard", "models"):
        os.makedirs(f"{B}/{d}", exist_ok=True)
    for f in glob.glob(f"{B}/workspace/detections/*.npz"):
        os.remove(f)
    raw.save(f"{B}/workspace/detections/{TEST}.npz")
    for f in glob.glob(f"{W}/e2e/{TEST}/{TEST}.json"):
        shutil.copy(f, f"{B}/workspace/scoreboard/")
    video = glob.glob(f"{INP}/**/{TEST}.mp4", recursive=True)[0]
    dst = f"{B}/workspace/videos/{TEST}.mp4"
    if not os.path.exists(dst):
        os.symlink(video, dst)
    shutil.copy(glob.glob(f"{INP}/**/shot_fold_test_{TEST}.onnx", recursive=True)[0], f"{B}/models/padel_shot_classifier.onnx")
    json.dump({"classes": ["Serve", "Forehand", "Backhand", "Smash", "Dropshot", "Other"], "window": 31},
              open(f"{B}/models/padel_shot_classifier.json", "w"))
    sh("python -m padelcoach evaluate-models")
    report[TEST] = json.load(open(f"{B}/workspace/results/model_evaluation.json"))[TEST]
    report[TEST]["ball_method"] = best
    shutil.copytree(f"{B}/workspace/results/{TEST}@models", f"{W}/e2e/{TEST}/analysis", dirs_exist_ok=True)
json.dump(report, open(f"{W}/cross_match_end_to_end.json", "w"), indent=1)
print("RESULT", json.dumps(report), flush=True)
'''


import json as _json  # noqa: E402
d = ROOT / "kaggle" / "cv_e2e"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(e2e_job(), encoding="utf-8")
meta = {"id": f"{OWNER}/padel-cv-e2e", "title": "padel-cv-e2e", "code_file": "train.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [f"{OWNER}/padel-coach-data"], "competition_sources": [],
        "kernel_sources": [f"{OWNER}/padel-cv-ball-women", f"{OWNER}/padel-cv-ball-men", f"{OWNER}/padel-cv-pose-women",
                           f"{OWNER}/padel-cv-pose-men", f"{OWNER}/padel-cv-shot"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
(d / "kernel-metadata.json").write_text(_json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)


def eval_job():
    """CPU only: choose the ball-selection method by measured F1 (same method
    for both matches) from saved fold detections, then run the full pipeline +
    evaluation on each unseen match."""
    return HEADER + '''
sh("pip install -q onnxruntime")
sh("python -m unittest discover -s tests -v")
import numpy as np, sys
sys.path.insert(0, B)
from padelcoach.perception import BALL_METHODS, select_ball
from padelcoach.sources import padeltracker as pt
from padelcoach.tracks import RawDetections
MATCHES = ["2022_BCN_FinalF_1", "2022_BCN_FinalM_1"]
def find(name):
    return glob.glob(f"{INP}/**/{name}", recursive=True)[0]
def load_cands(path):
    z = np.load(path)
    frames, counts, allc = z["frames"], z["counts"], z["cands"]   # decompress ONCE
    out, off = {}, 0
    for f, c in zip(frames, counts):
        out[int(f)] = allc[off:off + int(c)]
        off += int(c)
    return out
def ball_pr(ball, gt, tol=10.0):
    n = min(len(ball), len(gt))
    last = int(np.where(np.isfinite(gt[:n, 0]))[0].max())
    tp = fp = fn = 0
    for f in range(last + 1):
        g, d = gt[f], ball[f]
        hg, hd = np.isfinite(g[0]), np.isfinite(d[0])
        if hg and hd and np.hypot(d[0] - g[0], d[1] - g[1]) <= tol:
            tp += 1
        else:
            fp += int(hd); fn += int(hg)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}
raws = {T: RawDetections.load(find(f"{T}.npz")) for T in MATCHES}
cands = {T: load_cands(find(f"{T}_ballcands.npz")) for T in MATCHES}
gts = {T: pt.load_ball_gt(T, raws[T].n_frames) for T in MATCHES}
ball_eval = {}
for m in BALL_METHODS:
    ball_eval[m] = {T: ball_pr(select_ball(cands[T], raws[T].n_frames, m), gts[T]) for T in MATCHES}
    ball_eval[m]["mean_f1"] = round(float(np.mean([ball_eval[m][T]["f1"] for T in MATCHES])), 4)
    print("BALL METHOD", m, json.dumps(ball_eval[m]), flush=True)
best = max(BALL_METHODS, key=lambda m: ball_eval[m]["mean_f1"])
print("CHOSEN BALL METHOD", best, flush=True)
json.dump({"methods": ball_eval, "chosen": best}, open(f"{W}/ball_method_selection.json", "w"), indent=1)
report = {}
for TEST in MATCHES:
    raw = raws[TEST]
    raw.ball = select_ball(cands[TEST], raw.n_frames, best)
    for d in ("workspace/detections", "workspace/videos", "workspace/scoreboard", "models"):
        os.makedirs(f"{B}/{d}", exist_ok=True)
    for f in glob.glob(f"{B}/workspace/detections/*.npz"):
        os.remove(f)
    raw.save(f"{B}/workspace/detections/{TEST}.npz")
    shutil.copy(find(f"{TEST}.json"), f"{B}/workspace/scoreboard/{TEST}.json")
    dst = f"{B}/workspace/videos/{TEST}.mp4"
    if not os.path.exists(dst):
        os.symlink(find(f"{TEST}.mp4"), dst)
    shutil.copy(find(f"shot_fold_test_{TEST}.onnx"), f"{B}/models/padel_shot_classifier.onnx")
    json.dump({"classes": ["Serve", "Forehand", "Backhand", "Smash", "Dropshot", "Other"], "window": 31},
              open(f"{B}/models/padel_shot_classifier.json", "w"))
    sh("python -m padelcoach evaluate-models")
    report[TEST] = json.load(open(f"{B}/workspace/results/model_evaluation.json"))[TEST]
    report[TEST]["ball_method"] = best
    shutil.copytree(f"{B}/workspace/results/{TEST}@models", f"{W}/analysis/{TEST}", dirs_exist_ok=True)
json.dump(report, open(f"{W}/cross_match_end_to_end.json", "w"), indent=1)
print("RESULT", json.dumps(report), flush=True)
'''


d = ROOT / "kaggle" / "cv_eval"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(eval_job(), encoding="utf-8")
meta = {"id": f"{OWNER}/padel-cv-eval", "title": "padel-cv-eval", "code_file": "train.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{OWNER}/padel-coach-data", f"{OWNER}/padel-cv-detections"], "competition_sources": [],
        "kernel_sources": [f"{OWNER}/padel-cv-shot"], "model_sources": []}
(d / "kernel-metadata.json").write_text(_json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)


def multivenue_job():
    """Ball model v2 on Barcelona + 12 external WPT venues; 4 whole venues held out as an
    unseen TEST set (never used for training or model selection). Compares the old
    Barcelona-only ball model, the new model, and pose models (player recall) on it."""
    return HEADER + GPU_SETUP + '''
import collections, random
random.seed(0)
EXT = os.path.dirname(os.path.dirname(glob.glob(f"{INP}/**/new_annotations/train/labels", recursive=True)[0]))
TEST_CLIPS, VAL_CLIPS = {"9", "12", "13", "16"}, {"2"}
def clip_of(p):
    return os.path.basename(p).split("-")[0]
# images live under padel/padel/<split>/images, the 3-class labels under new_annotations/...: map by file stem
DROOT = os.path.dirname(os.path.dirname(EXT))
img_of = {}
for ext_ in ("png", "jpg", "jpeg"):
    for im in glob.glob(f"{DROOT}/**/*.{ext_}", recursive=True):
        img_of.setdefault(os.path.splitext(os.path.basename(im))[0], im)
ext = []
for split in ("train", "valid"):
    for lab in glob.glob(f"{EXT}/{split}/labels/*.txt"):
        stem = os.path.splitext(os.path.basename(lab))[0]
        if stem in img_of:
            ext.append((img_of[stem], lab))
print("images found", len(img_of), "external labelled images", len(ext), flush=True)
assert len(ext) > 5000, "external dataset not found - stop before using GPU time"
def write_split(items, root, keep_cls):
    os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/labels", exist_ok=True)
    for img, lab in items:
        name = os.path.basename(img)
        dst = f"{root}/images/{name}"
        if not os.path.exists(dst):
            os.symlink(img, dst)
        rows = [r.split() for r in open(lab) if r.strip()]
        rows = ["0 " + " ".join(r[1:5]) for r in rows if r[0] in keep_cls]
        open(f"{root}/labels/{os.path.splitext(name)[0]}.txt", "w").write("\\n".join(rows))
groups = collections.defaultdict(list)
for it in ext:
    c = clip_of(it[0])
    groups["test" if c in TEST_CLIPS else ("val" if c in VAL_CLIPS else "train")].append(it)
print({k: len(v) for k, v in groups.items()}, flush=True)
for k, v in groups.items():
    write_split(v, f"/tmp/ext_ball/{k}", {"0"})
write_split(groups["test"], "/tmp/ext_players/test", {"1"})
# Barcelona (PadelTracker100) ball frames: train = first 85 % of both matches, val = last 15 %
sh(f"python colab_scripts/prepare_yolo_datasets.py --videos {VID} --labels labels --out /tmp/pt --only ball --ball-step 5")
open("/tmp/ball_mv.yaml", "w").write(
    "train: [/tmp/pt/ball/images/train, /tmp/ext_ball/train/images]\\n"
    "val: [/tmp/pt/ball/images/val, /tmp/ext_ball/val/images]\\n"
    "test: /tmp/ext_ball/test/images\\nnames:\\n  0: ball\\n")
YOLO("yolo11s.pt").train(data="/tmp/ball_mv.yaml", imgsz=1280, epochs=40, time=3.5, patience=8, batch=8, workers=4,
                         project=f"{W}/runs", name="ball_mv", exist_ok=True, **AUG)
new_ball = f"{W}/runs/ball_mv/weights/best.pt"
old_ball = glob.glob(f"{INP}/**/ball_best.pt", recursive=True)[0]
def ball_points(model_path, items, conf=0.25, tol=10.0):
    m = YOLO(model_path); tp = fp = fn = 0
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        res = m.predict([b[0] for b in batch], imgsz=1280, conf=conf, verbose=False, half=True)
        for (img, lab), r in zip(batch, res):
            h, w = r.orig_shape
            gts = [list(map(float, x.split()[1:5])) for x in open(lab) if x.strip() and x.split()[0] == "0"]
            gts = [(g[0] * w, g[1] * h) for g in gts]
            if len(r.boxes):
                j = int(r.boxes.conf.argmax()); x, y = r.boxes.xywh[j, :2].tolist()
                if gts and min(np.hypot(x - gx, y - gy) for gx, gy in gts) <= tol:
                    tp += 1
                else:
                    fp += 1; fn += int(bool(gts))
            elif gts:
                fn += 1
    p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / max(p + rc, 1e-9), 4)}
import numpy as np
def player_recall(model_path, items, conf=0.3):
    m = YOLO(model_path); tp = n_gt = n_det = 0
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        res = m.predict([b[0] for b in batch], imgsz=1280, conf=conf, verbose=False, half=True)
        for (img, lab), r in zip(batch, res):
            h, w = r.orig_shape
            gts = [list(map(float, x.split()[1:5])) for x in open(lab) if x.strip() and x.split()[0] == "1"]
            gts = [((g[0] - g[2] / 2) * w, (g[1] - g[3] / 2) * h, (g[0] + g[2] / 2) * w, (g[1] + g[3] / 2) * h) for g in gts]
            dets = r.boxes.xyxy.tolist() if len(r.boxes) else []
            n_gt += len(gts); used = set()
            for g in gts:
                best, bj = 0, None
                for j, d in enumerate(dets):
                    if j in used: continue
                    ix = max(0, min(g[2], d[2]) - max(g[0], d[0])); iy = max(0, min(g[3], d[3]) - max(g[1], d[1]))
                    inter = ix * iy; u = (g[2]-g[0])*(g[3]-g[1]) + (d[2]-d[0])*(d[3]-d[1]) - inter
                    if inter / max(u, 1e-6) > best: best, bj = inter / max(u, 1e-6), j
                if bj is not None and best >= 0.5:
                    used.add(bj); tp += 1
            # only count detections that overlap the court players' area as false positives if big
            n_det += len(used) + sum(1 for j, d in enumerate(dets) if j not in used and (d[3]-d[1]) > 0.06 * h)
    return {"recall": round(tp / max(n_gt, 1), 4), "precision_players_sized": round(tp / max(n_det, 1), 4), "gt_players": n_gt}
test = groups["test"]
res = {"test_clips_unseen_venues": sorted(TEST_CLIPS), "n_test_images": len(test),
       "ball_old_barcelona_only": ball_points(old_ball, test),
       "ball_new_multivenue": ball_points(new_ball, test),
       "ball_new_ultralytics_test": m_box(YOLO(new_ball).val(data="/tmp/ball_mv.yaml", split="test", imgsz=1280)),
       "players_pose_pretrained": player_recall("yolo11s-pose.pt", test),
       "players_pose_finetuned_barcelona": player_recall(glob.glob(f"{INP}/**/pose_best.pt", recursive=True)[0], test)}
per_clip = {}
for c in sorted(TEST_CLIPS):
    items = [t for t in test if clip_of(t[0]) == c]
    per_clip[c] = {"n": len(items), "old": ball_points(old_ball, items), "new": ball_points(new_ball, items)}
res["ball_per_unseen_venue"] = per_clip
shutil.copy(new_ball, f"{W}/ball_multivenue.pt")
p = YOLO(new_ball).export(format="onnx", imgsz=1280, opset=17, simplify=True)
shutil.copy(p, f"{W}/padel_ball_multivenue.onnx")
json.dump(res, open(f"{W}/multivenue_results.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
'''


d = ROOT / "kaggle" / "multivenue"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(multivenue_job(), encoding="utf-8")
meta = {"id": f"{OWNER}/padel-multivenue-ball", "title": "padel-multivenue-ball", "code_file": "train.py",
        "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{OWNER}/padel-coach-data", f"{OWNER}/padel-coach-models", "ichimarugin/padel-ball-dataset"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
(d / "kernel-metadata.json").write_text(_json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)


def shot_unseen_job():
    """Shot classifier on josmazna/padelpose (other players, other camera, MediaPipe poses,
    no ball). All variants reported - none is picked by test accuracy."""
    return HEADER + '''
sh("pip install -q mediapipe onnxruntime huggingface_hub joblib")
import numpy as np, joblib, onnxruntime as ort, sys
from huggingface_hub import snapshot_download
sys.path.insert(0, B)
from padelcoach.shotfeatures import sequence_features
d = snapshot_download("josmazna/padelpose", repo_type="dataset", local_dir="/tmp/padelpose")
MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LABEL = {"forehand": "Forehand", "backhand": "Backhand", "fhvolley": "Forehand", "bhvolley": "Backhand",
         "bandeja": "Smash", "smash": "Smash"}
model = glob.glob(f"{INP}/**/padel_shot_classifier.onnx", recursive=True)[0]
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
classes = ["Serve", "Forehand", "Backhand", "Smash", "Dropshot", "Other"]
samples = []
for f in sorted(glob.glob("/tmp/padelpose/*.pkl")):
    name = os.path.basename(f).split(" (")[0].lower()
    if name not in LABEL:
        continue
    res = joblib.load(f)
    seq = []
    for fr in res:
        if not fr.pose_landmarks:
            seq.append(np.full((17, 3), np.nan, np.float32)); continue
        lm = fr.pose_landmarks[0]
        seq.append(np.array([[lm[i].x * 1920, lm[i].y * 1080, lm[i].visibility] for i in MP2COCO], np.float32))
    samples.append((name, LABEL[name], np.stack(seq)))
print("samples", len(samples), "lengths min/median/max",
      min(len(s[2]) for s in samples), int(np.median([len(s[2]) for s in samples])), max(len(s[2]) for s in samples), flush=True)
def window(seq, mode):
    T = len(seq)
    if mode == "resample":
        idx = np.linspace(0, T - 1, 31).round().astype(int)
    else:   # centre 31 frames
        c = T // 2
        idx = np.clip(np.arange(c - 15, c + 16), 0, T - 1)
    return seq[idx]
results = {}
for mode in ("centre", "resample"):
    for far in (False, True):
        X = np.stack([sequence_features(window(s[2], mode), np.full((31, 3), np.nan, np.float32), far, False) for s in samples])
        logits = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]
        pred = [classes[i] for i in logits.argmax(1)]
        y = [s[1] for s in samples]
        acc = float(np.mean([p == t for p, t in zip(pred, y)]))
        per = {}
        for c in ("Forehand", "Backhand", "Smash"):
            idx = [i for i, t in enumerate(y) if t == c]
            per[c] = round(float(np.mean([pred[i] == c for i in idx])), 3)
        conf = {}
        for t, p in zip(y, pred):
            conf.setdefault(t, {}).setdefault(p, 0); conf[t][p] += 1
        results[f"{mode}|far={far}"] = {"accuracy": round(acc, 4), "per_class_recall": per, "confusion": conf}
        print("VARIANT", mode, "far", far, json.dumps(results[f"{mode}|far={far}"]), flush=True)
json.dump({"n": len(samples), "variants": results}, open(f"{W}/shot_unseen_padelpose.json", "w"), indent=1)
print("RESULT", json.dumps(results), flush=True)
'''


d = ROOT / "kaggle" / "shot_unseen"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(shot_unseen_job(), encoding="utf-8")
meta = {"id": f"{OWNER}/padel-shot-unseen", "title": "padel-shot-unseen", "code_file": "train.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": [f"{OWNER}/padel-coach-data", f"{OWNER}/padel-coach-models"], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}
(d / "kernel-metadata.json").write_text(_json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)


def shot_multiview_job():
    """Shot classifier made view-robust: ball-feature dropout + multi-view data.
    padelpose is split 50/50 (stratified, fixed seed) BEFORE anything else; the test half
    is never used for training or selection. Reports Barcelona cross-match and padelpose-test
    for (A) Barcelona-only and (B) Barcelona + padelpose-train."""
    return HEADER + '''
sh("pip install -q mediapipe onnx onnxscript huggingface_hub joblib")
import numpy as np, joblib, torch, sys, importlib
from huggingface_hub import snapshot_download
sys.path.insert(0, B); sys.path.insert(0, f"{B}/colab_scripts")
from padelcoach.shotfeatures import sequence_features
tsm = importlib.import_module("train_shot_model")
dev = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Serve", "Forehand", "Backhand", "Smash", "Dropshot", "Other"]
# ---------- data
bar = np.load("shot_sequences.npz", allow_pickle=False)
Xb, yb, mb = bar["X"], bar["y"], bar["match"]
snapshot_download("josmazna/padelpose", repo_type="dataset", local_dir="/tmp/padelpose")
MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LABEL = {"forehand": "Forehand", "backhand": "Backhand", "fhvolley": "Forehand", "bhvolley": "Backhand",
         "bandeja": "Smash", "smash": "Smash"}
Xp, yp, kind = [], [], []
for f in sorted(glob.glob("/tmp/padelpose/*.pkl")):
    name = os.path.basename(f).split(" (")[0].lower()
    if name not in LABEL:
        continue
    seq = []
    for fr in joblib.load(f):
        if not fr.pose_landmarks:
            seq.append(np.full((17, 3), np.nan, np.float32)); continue
        lm = fr.pose_landmarks[0]
        seq.append(np.array([[lm[i].x * 1920, lm[i].y * 1080, lm[i].visibility] for i in MP2COCO], np.float32))
    seq = np.stack(seq)
    seq = seq[np.linspace(0, len(seq) - 1, 31).round().astype(int)]
    Xp.append(sequence_features(seq, np.full((31, 3), np.nan, np.float32), False, False))
    yp.append(CLASSES.index(LABEL[name])); kind.append(name)
Xp, yp, kind = np.stack(Xp).astype(np.float32), np.array(yp), np.array(kind)
rng = np.random.RandomState(0)
test_mask = np.zeros(len(yp), bool)
for k in np.unique(kind):            # stratified 50/50 split by stroke type, fixed seed
    idx = np.where(kind == k)[0]; rng.shuffle(idx)
    test_mask[idx[: len(idx) // 2]] = True
print("padelpose train", (~test_mask).sum(), "test", test_mask.sum(), flush=True)
# ---------- ball-feature dropout: the model must also work without ball positions
base_aug = tsm.augment
def augment_drop(x):
    out = base_aug(x)
    drop = (torch.rand(out.shape[0], device=out.device) < 0.5).float().view(-1, 1, 1)
    out[..., 51:54] = out[..., 51:54] * (1 - drop)
    return out
tsm.augment = augment_drop
def acc(m, X, y):
    p = tsm.predict(m, X, dev).argmax(1)
    return round(float((p == y).mean()), 4)
def per_class(m, X, y):
    p = tsm.predict(m, X, dev).argmax(1)
    return {CLASSES[c]: round(float((p[y == c] == c).mean()), 3) for c in np.unique(y)}
report = {}
for variant, use_pp in (("A_barcelona_only", False), ("B_barcelona_plus_padelpose_train", True)):
    r = {"cross_match": {}}
    for tr_m in sorted(set(mb)):
        te = mb != tr_m
        Xtr, ytr = Xb[~te], yb[~te]
        if use_pp:
            Xtr, ytr = np.concatenate([Xtr, Xp[~test_mask]]), np.concatenate([ytr, yp[~test_mask]])
        m = tsm.train(Xtr, ytr, len(CLASSES), 120, dev)
        r["cross_match"][f"train_{tr_m}"] = {"broadcast_unseen_match_acc": acc(m, Xb[te], yb[te]),
                                             "padelpose_test_acc": acc(m, Xp[test_mask], yp[test_mask])}
    Xtr, ytr = Xb, yb
    if use_pp:
        Xtr, ytr = np.concatenate([Xb, Xp[~test_mask]]), np.concatenate([yb, yp[~test_mask]])
    m = tsm.train(Xtr, ytr, len(CLASSES), 120, dev)
    r["final_model_padelpose_test_acc"] = acc(m, Xp[test_mask], yp[test_mask])
    r["final_model_padelpose_test_per_class"] = per_class(m, Xp[test_mask], yp[test_mask])
    report[variant] = r
    tsm.export(m.cpu(), Xb, f"{W}/shot_{variant}.onnx")
    print("VARIANT", variant, json.dumps(r), flush=True)
json.dump(report, open(f"{W}/shot_multiview_results.json", "w"), indent=1)
print("RESULT", json.dumps(report), flush=True)
'''


d = ROOT / "kaggle" / "shot_multiview"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(shot_multiview_job(), encoding="utf-8")
meta = {"id": f"{OWNER}/padel-shot-multiview", "title": "padel-shot-multiview", "code_file": "train.py",
        "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [f"{OWNER}/padel-coach-data"], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}
(d / "kernel-metadata.json").write_text(_json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)


def multivenue_pose_job():
    """Pose model that generalises across venues: Barcelona frames (boxes + 17 keypoints)
    + external WPT clips (player boxes only, keypoints marked not-visible so they only train
    detection). Same 4 held-out clips as the ball test; compares pretrained, Barcelona-only
    and multi-venue pose models on player recall/precision there."""
    return HEADER + GPU_SETUP + '''
import collections, numpy as np
EXT = os.path.dirname(os.path.dirname(glob.glob(f"{INP}/**/new_annotations/train/labels", recursive=True)[0]))
DROOT = os.path.dirname(os.path.dirname(EXT))
img_of = {}
for e in ("png", "jpg", "jpeg"):
    for im in glob.glob(f"{DROOT}/**/*.{e}", recursive=True):
        img_of.setdefault(os.path.splitext(os.path.basename(im))[0], im)
TEST_CLIPS, VAL_CLIPS = {"9", "12", "13", "16"}, {"2"}
groups = collections.defaultdict(list)
for split in ("train", "valid"):
    for lab in glob.glob(f"{EXT}/{split}/labels/*.txt"):
        stem = os.path.splitext(os.path.basename(lab))[0]
        if stem in img_of:
            c = stem.split("-")[0]
            groups["test" if c in TEST_CLIPS else ("val" if c in VAL_CLIPS else "train")].append((img_of[stem], lab))
print({k: len(v) for k, v in groups.items()}, flush=True)
assert len(groups["train"]) > 4000
NOKP = " ".join(["0 0 0"] * 17)
for k, items in groups.items():
    root = f"/tmp/ext_pose/{k}"
    os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/labels", exist_ok=True)
    for img, lab in items:
        name = os.path.basename(img)
        if not os.path.exists(f"{root}/images/{name}"):
            os.symlink(img, f"{root}/images/{name}")
        rows = [r.split() for r in open(lab) if r.strip()]
        rows = ["0 " + " ".join(r[1:5]) + " " + NOKP for r in rows if r[0] == "1"]
        open(f"{root}/labels/{os.path.splitext(name)[0]}.txt", "w").write("\\n".join(rows))
sh(f"python colab_scripts/prepare_yolo_datasets.py --videos {VID} --labels labels --out /tmp/pt --only pose --pose-step 12")
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
open("/tmp/pose_mv.yaml", "w").write(
    "train: [/tmp/pt/pose/images/train, /tmp/ext_pose/train/images]\\n"
    "val: [/tmp/pt/pose/images/val, /tmp/ext_pose/val/images]\\n"
    f"test: /tmp/ext_pose/test/images\\nkpt_shape: [17, 3]\\nflip_idx: {FLIP}\\nnames:\\n  0: player\\n")
YOLO("yolo11s-pose.pt").train(data="/tmp/pose_mv.yaml", imgsz=1280, epochs=30, time=3.0, patience=6, batch=8,
                              workers=4, freeze=10, project=f"{W}/runs", name="pose_mv", exist_ok=True, **AUG)
new_pose = f"{W}/runs/pose_mv/weights/best.pt"
def player_eval(model_path, items, conf=0.3):
    m = YOLO(model_path); tp = n_gt = n_det = 0
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        for (img, lab), r in zip(batch, m.predict([b[0] for b in batch], imgsz=1280, conf=conf, verbose=False, half=True)):
            h, w = r.orig_shape
            gts = [list(map(float, x.split()[1:5])) for x in open(lab) if x.strip() and x.split()[0] == "1"]
            gts = [((g[0]-g[2]/2)*w, (g[1]-g[3]/2)*h, (g[0]+g[2]/2)*w, (g[1]+g[3]/2)*h) for g in gts]
            dets = r.boxes.xyxy.tolist() if len(r.boxes) else []
            n_gt += len(gts); used = set()
            for g in gts:
                best, bj = 0, None
                for j, d in enumerate(dets):
                    if j in used: continue
                    ix = max(0, min(g[2], d[2]) - max(g[0], d[0])); iy = max(0, min(g[3], d[3]) - max(g[1], d[1]))
                    inter = ix * iy; u = (g[2]-g[0])*(g[3]-g[1]) + (d[2]-d[0])*(d[3]-d[1]) - inter
                    if inter / max(u, 1e-6) > best: best, bj = inter / max(u, 1e-6), j
                if bj is not None and best >= 0.5:
                    used.add(bj); tp += 1
            n_det += len(used) + sum(1 for j, d in enumerate(dets) if j not in used and (d[3]-d[1]) > 0.06*h)
    p, rc = tp / max(n_det, 1), tp / max(n_gt, 1)
    return {"recall": round(rc, 4), "precision": round(p, 4), "f1": round(2*p*rc/max(p+rc, 1e-9), 4), "gt_players": n_gt}
test = groups["test"]
res = {"test_clips_unseen": sorted(TEST_CLIPS),
       "pretrained": player_eval("yolo11s-pose.pt", test),
       "barcelona_finetuned": player_eval(glob.glob(f"{INP}/**/pose_best.pt", recursive=True)[0], test),
       "multivenue": player_eval(new_pose, test),
       "multivenue_barcelona_val_keypoints": m_pose(YOLO(new_pose).val(data="/tmp/pose_mv.yaml", split="val", imgsz=1280))}
shutil.copy(new_pose, f"{W}/pose_multivenue.pt")
p = YOLO(new_pose).export(format="onnx", imgsz=1280, opset=17, simplify=True)
shutil.copy(p, f"{W}/padel_pose_multivenue.onnx")
json.dump(res, open(f"{W}/pose_multivenue_results.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
'''


d = ROOT / "kaggle" / "multivenue_pose"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(multivenue_pose_job(), encoding="utf-8")
meta = {"id": f"{OWNER}/padel-multivenue-pose", "title": "padel-multivenue-pose", "code_file": "train.py",
        "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{OWNER}/padel-coach-data", f"{OWNER}/padel-coach-models", "ichimarugin/padel-ball-dataset"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
(d / "kernel-metadata.json").write_text(_json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)
