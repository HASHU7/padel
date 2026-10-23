"""Train the temporal shot classifier (PyTorch, GPU) and export ONNX.

Input: shot_sequences.npz from the bundle (X: (n, 31, 55), y, match).
Evaluation: cross-match (train women -> test men and vice versa) - the honest
number for a new match - plus the final model trained on both matches.
Outputs: padel_shot_classifier.onnx + padel_shot_classifier.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

N_FEAT = 55


class ShotNet(nn.Module):
    def __init__(self, n_feat=N_FEAT, n_cls=6, h=96, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, h, 5, padding=2), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(h, h, 3, padding=4, dilation=4), nn.BatchNorm1d(h), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, n_cls))

    def forward(self, x):            # x: (B, T, F)
        z = self.conv(x.transpose(1, 2))
        z = torch.cat([z.mean(-1), z.amax(-1)], 1)
        return self.head(z)


def augment(x):
    """Time shift (+-4 frames), keypoint noise, scale and dropout of joints."""
    B, T, F = x.shape
    shift = torch.randint(-4, 5, (B,))
    out = torch.stack([torch.roll(x[i], int(shift[i]), 0) for i in range(B)])
    coords = torch.ones(F, dtype=torch.bool, device=x.device)
    coords[2:51:3] = False           # keep visibility flags exact
    noise = torch.randn_like(out) * 0.04
    out = out + noise * coords
    out[..., :51] = out[..., :51] * (1 + 0.1 * torch.randn(B, 1, 1, device=x.device)).clamp(0.8, 1.2)
    return out


def train(X, y, n_cls, epochs=120, device="cuda", seed=0):
    torch.manual_seed(seed)
    m = ShotNet(n_cls=n_cls).to(device)
    w = torch.tensor(len(y) / (n_cls * np.maximum(np.bincount(y, minlength=n_cls), 1)), dtype=torch.float32, device=device)
    w = w.clamp(max=8.0)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(y) // 64 + 1))
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(yt), device=device)
        for i in range(0, len(yt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(m(augment(Xt[b])), yt[b])
            loss.backward()
            opt.step()
            sched.step()
    return m.eval()


@torch.no_grad()
def predict(m, X, device="cuda"):
    return m(torch.tensor(X, device=device)).softmax(-1).cpu().numpy()


def export(m, X, path):
    dummy = torch.zeros(1, X.shape[1], X.shape[2])
    kw = dict(input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
              opset_version=17)
    try:  # classic TorchScript exporter (no onnxscript dependency)
        torch.onnx.export(m, dummy, path, dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(m, dummy, path, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="shot_sequences.npz")
    ap.add_argument("--out", default="/content/models")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data, allow_pickle=False)
    X, y, match, classes = d["X"], d["y"], d["match"], [str(c) for c in d["classes"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_cls = len(classes)
    report = {"classes": classes, "n": int(len(y)), "cross_match": {}}
    for tr_m in sorted(set(match)):
        te = match != tr_m
        tr = ~te
        if tr.sum() < 50 or te.sum() < 50:
            continue
        m = train(X[tr], y[tr], n_cls, args.epochs, dev)
        p = predict(m, X[te], dev).argmax(1)
        acc = float((p == y[te]).mean())
        no_other = y[te] != classes.index("Other")
        acc_no = float((p[no_other] == y[te][no_other]).mean())
        report["cross_match"][f"train_{tr_m}"] = {"accuracy": round(acc, 4), "accuracy_excl_other": round(acc_no, 4)}
        print(f"train {tr_m} -> other match: accuracy {acc:.3f} (excl. Other {acc_no:.3f})")
        # fold model (never saw the other match) for honest end-to-end testing
        test_m = [x for x in sorted(set(match)) if x != tr_m][0]
        export(m.cpu(), X, f"{args.out}/shot_fold_test_{test_m}.onnx")
    m = train(X, y, n_cls, args.epochs, dev)
    export(m.cpu(), X, f"{args.out}/padel_shot_classifier.onnx")
    report["window"] = int(X.shape[1])
    json.dump(report, open(f"{args.out}/padel_shot_classifier.json", "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("Rule-based local classifier reference (cross-match, no training): ~80% (women) - compare before installing.")


if __name__ == "__main__":
    main()
