"""Generate the multi-venue pose job with PSEUDO-LABELLED keypoints.

Boxes-only labels for the external venues made the keypoint head predict "not visible"
(Barcelona keypoint mAP50-95 fell 0.80 -> 0.30). Here every external player box gets the
keypoints of the teacher (COCO-pretrained or Barcelona-fine-tuned pose model) whose matching
detection (IoU >= 0.5) has the higher mean keypoint confidence; only keypoints with
confidence >= 0.5 inside the box are kept as visible.

Usage: python kaggle/make_pose_pseudolabel.py <owner>   (hashulearn | hashuu7)
"""
import json
import sys

import make_cv_kernels as M  # noqa: E402  (regenerates the other jobs too - harmless)

OWNER = sys.argv[1] if len(sys.argv) > 1 else "hashulearn"

BODY = '''
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
assert len(groups["train"]) > 4000, "external dataset not found"
teacher_paths = glob.glob(f"{INP}/**/pose_best.pt", recursive=True) + glob.glob(f"{INP}/**/pose/weights/best.pt", recursive=True)
assert teacher_paths, "Barcelona pose model not found"
teachers = [YOLO("yolo11s-pose.pt"), YOLO(teacher_paths[0])]
print("teachers:", ["yolo11s-pose.pt", teacher_paths[0]], flush=True)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1e-6)
stats = collections.Counter()
for k in ("train", "val"):
    root = f"/tmp/ext_pose/{k}"
    os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/labels", exist_ok=True)
    items = groups[k]
    for i in range(0, len(items), 16):
        batch = items[i:i + 16]
        preds = [t.predict([b[0] for b in batch], imgsz=1280, conf=0.2, verbose=False, half=True) for t in teachers]
        for bi, (img, lab) in enumerate(batch):
            name = os.path.basename(img)
            if not os.path.exists(f"{root}/images/{name}"):
                os.symlink(img, f"{root}/images/{name}")
            h, w = preds[0][bi].orig_shape
            rows = []
            for r in open(lab):
                r = r.split()
                if not r or r[0] != "1":
                    continue
                cx, cy, bw, bh = map(float, r[1:5])
                g = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                best, best_conf = None, 0.0
                for tp in preds:
                    p = tp[bi]
                    if p.keypoints is None or not len(p.boxes):
                        continue
                    for j, d in enumerate(p.boxes.xyxy.tolist()):
                        if iou(g, d) >= 0.5:
                            kp = p.keypoints.data[j].cpu().numpy()
                            if kp[:, 2].mean() > best_conf:
                                best, best_conf = kp, float(kp[:, 2].mean())
                vals = []
                if best is not None:
                    stats["players_with_pseudo_kpts"] += 1
                    for x, y, c in best:
                        ok = c >= 0.5 and g[0] - 5 <= x <= g[2] + 5 and g[1] - 5 <= y <= g[3] + 5
                        stats["kpts_kept" if ok else "kpts_dropped"] += 1
                        vals += [f"{x / w:.6f}", f"{y / h:.6f}", "2"] if ok else ["0", "0", "0"]
                else:
                    stats["players_box_only"] += 1
                    vals = ["0"] * 51
                rows.append("0 " + " ".join(r[1:5]) + " " + " ".join(vals))
            open(f"{root}/labels/{os.path.splitext(name)[0]}.txt", "w").write(chr(10).join(rows))
print("PSEUDO-LABEL STATS", dict(stats), flush=True)
os.makedirs("/tmp/ext_pose/test/images", exist_ok=True); os.makedirs("/tmp/ext_pose/test/labels", exist_ok=True)
for img, lab in groups["test"]:     # held-out venues: player boxes only (detection metrics)
    name = os.path.basename(img)
    if not os.path.exists(f"/tmp/ext_pose/test/images/{name}"):
        os.symlink(img, f"/tmp/ext_pose/test/images/{name}")
    rows = ["0 " + " ".join(r.split()[1:5]) + " " + " ".join(["0"] * 51) for r in open(lab) if r.strip() and r.split()[0] == "1"]
    open(f"/tmp/ext_pose/test/labels/{os.path.splitext(name)[0]}.txt", "w").write(chr(10).join(rows))
sh(f"python colab_scripts/prepare_yolo_datasets.py --videos {VID} --labels labels --out /tmp/pt --only pose --pose-step 12")
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
NL = chr(10)
open("/tmp/pose_mv.yaml", "w").write(
    f"train: [/tmp/pt/pose/images/train, /tmp/ext_pose/train/images]{NL}"
    f"val: /tmp/pt/pose/images/val{NL}"
    f"test: /tmp/ext_pose/test/images{NL}kpt_shape: [17, 3]{NL}flip_idx: {FLIP}{NL}names:{NL}  0: player{NL}")
YOLO("yolo11s-pose.pt").train(data="/tmp/pose_mv.yaml", imgsz=1280, epochs=30, time=2.8, patience=6, batch=8,
                              workers=4, freeze=10, project=f"{W}/runs", name="pose_pl", exist_ok=True, **AUG)
new_pose = f"{W}/runs/pose_pl/weights/best.pt"
def player_eval(model_path, items, conf=0.3):
    m = YOLO(model_path); tp = n_gt = n_det = 0; kconf = []
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
                    if j not in used and iou(g, d) > best:
                        best, bj = iou(g, d), j
                if bj is not None and best >= 0.5:
                    used.add(bj); tp += 1
                    if r.keypoints is not None:
                        kconf.append(float(r.keypoints.data[bj][:, 2].mean()))
            n_det += len(used) + sum(1 for j, d in enumerate(dets) if j not in used and (d[3]-d[1]) > 0.06*h)
    p, rc = tp / max(n_det, 1), tp / max(n_gt, 1)
    return {"recall": round(rc, 4), "precision": round(p, 4), "f1": round(2*p*rc/max(p+rc, 1e-9), 4),
            "mean_keypoint_conf_on_found_players": round(float(np.mean(kconf)), 3) if kconf else None}
test = groups["test"]
res = {"test_clips_unseen": sorted(TEST_CLIPS), "pseudo_label_stats": dict(stats),
       "pretrained": player_eval("yolo11s-pose.pt", test),
       "barcelona_finetuned": player_eval(teacher_paths[0], test),
       "multivenue_pseudolabel": player_eval(new_pose, test),
       "barcelona_val_keypoints_new": m_pose(YOLO(new_pose).val(data="/tmp/pose_mv.yaml", split="val", imgsz=1280)),
       "barcelona_val_keypoints_barcelona_only": m_pose(YOLO(teacher_paths[0]).val(data="/tmp/pose_mv.yaml", split="val", imgsz=1280))}
shutil.copy(new_pose, f"{W}/pose_multivenue_pl.pt")
p = YOLO(new_pose).export(format="onnx", imgsz=1280, opset=17, simplify=True)
shutil.copy(p, f"{W}/padel_pose_multivenue_pl.onnx")
json.dump(res, open(f"{W}/pose_pseudolabel_results.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
'''

SOURCES = {"hashulearn": (["hashulearn/padel-coach-data", "hashulearn/padel-coach-models"], []),
           "hashuu7": (["hashuu7/padel-coach-data"], ["hashuu7/padel-coach-train-pose-shot"])}

d = M.ROOT / "kaggle" / f"pose_pseudolabel_{OWNER}"
d.mkdir(exist_ok=True)
(d / "train.py").write_text(M.HEADER + M.GPU_SETUP + BODY, encoding="utf-8")
meta = {"id": f"{OWNER}/padel-pose-pseudolabel", "title": "padel-pose-pseudolabel", "code_file": "train.py",
        "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": SOURCES[OWNER][0] + ["ichimarugin/padel-ball-dataset"],
        "competition_sources": [], "kernel_sources": SOURCES[OWNER][1], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", d)
