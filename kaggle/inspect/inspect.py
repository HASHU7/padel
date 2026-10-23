import glob, json, os, random, collections
import cv2, numpy as np
INP = "/kaggle/input"
W = "/kaggle/working"
root = [d for d in glob.glob(f"{INP}/*") if "ball" in d.lower()]
print("inputs:", os.listdir(INP), flush=True)
labels = glob.glob(f"{INP}/**/labels/**/*.txt", recursive=True)
images = glob.glob(f"{INP}/**/*.jpg", recursive=True) + glob.glob(f"{INP}/**/*.png", recursive=True)
print("label files", len(labels), "images", len(images), flush=True)
dirs = collections.Counter(os.path.dirname(p).replace(INP, "") for p in labels)
print("label dirs:", dirs.most_common(20), flush=True)
clips = collections.Counter(os.path.basename(p).split("-")[0] for p in labels)
print("clips (prefix before '-'):", len(clips), clips.most_common(60), flush=True)
cls = collections.Counter(); per_img = []
for p in random.sample(labels, min(3000, len(labels))):
    rows = [l.split() for l in open(p) if l.strip()]
    per_img.append(len(rows))
    for r in rows:
        cls[r[0]] += 1
print("class ids (sample):", dict(cls), "objects/img mean", np.mean(per_img), flush=True)
yaml = glob.glob(f"{INP}/**/*.yaml", recursive=True) + glob.glob(f"{INP}/**/*.txt", recursive=True)[:0]
for y in yaml[:5]:
    print("YAML", y, open(y).read()[:600], flush=True)
# montage of 16 images from different clips with boxes drawn
by_clip = collections.defaultdict(list)
for p in images:
    by_clip[os.path.basename(p).split("-")[0]].append(p)
pick = [random.choice(v) for k, v in sorted(by_clip.items())][:16]
tiles = []
for p in pick:
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    lab = glob.glob(f"{INP}/**/{os.path.splitext(os.path.basename(p))[0]}.txt", recursive=True)
    if lab:
        for r in open(lab[0]):
            r = r.split()
            if len(r) >= 5:
                c, x, y, bw, bh = r[0], *map(float, r[1:5])
                col = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)][int(c) % 4]
                cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)), col, 3)
    img = cv2.resize(img, (480, 270))
    cv2.putText(img, os.path.basename(p)[:20] + f" {w}x{h}", (5, 20), 0, 0.6, (0, 255, 255), 2)
    tiles.append(img)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f"{W}/montage.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump({"labels": len(labels), "images": len(images), "clips": len(clips), "classes": dict(cls)}, open(f"{W}/summary.json", "w"))
print("DONE", flush=True)
