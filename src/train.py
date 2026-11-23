"""Row B (direct VICReg training) and row C (distillation from the frozen teacher).

    python -m src.train --row B --size S --seed 0 [--data-frac 0.5]
    python -m src.train --row C --size L --seed 1

Both rows share EVERYTHING except the objective: same student architecture and
initialization (seeded), same cached augmented views, same epochs, optimizer steps,
batch size, optimizer, LR schedule. Checkpoints every epoch; re-running the same
command resumes from the last finished epoch.
"""
import argparse
import csv
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .common import (DEVICE, autocast, needs_scaler, resize_normalize, seed_everything,
                     to_device_uint8)
from .diag import diagnose, encode
from .models import build_student, distill_projector, n_params, vicreg_projector


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return f


def collapse_check(model, x_tr, y_tr, diag_idx):
    """kNN / emb-std / k-means NMI on a fixed labeled subset of the TRAIN split
    (4000 to fit the kNN, the rest to score). Monitoring only, no gradients."""
    f = encode(model, x_tr, C.IMG_SIZE, C.STUDENT_MEAN, C.STUDENT_STD, idx=diag_idx)
    y = y_tr[diag_idx]
    cut = int(0.8 * len(y))
    return diagnose(f[:cut], y[:cut], f[cut:], y[cut:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", choices=["B", "C"], required=True)
    ap.add_argument("--size", choices=list(C.SIZES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--wd", type=float, default=C.WD)
    a = ap.parse_args()

    name = C.tag(a.row, a.size, a.seed, a.data_frac)
    out = C.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "backbone.pt").exists():
        print(f"[{name}] already finished, skipping"); return

    # ---------------- data: cached views (+ teacher targets for C), nested subset
    views = np.load(C.CACHE / f"views_{C.IMG_SIZE}.npy", mmap_mode="r")
    n = int(round(len(views) * a.data_frac))
    views = np.ascontiguousarray(views[:n])                 # into RAM
    K = views.shape[1]
    if a.row == "C":
        tfeat = torch.from_numpy(np.load(C.CACHE / "teacher_feats.npy")[:n].astype(np.float32))
    from .data import load_split
    x_tr, y_tr = load_split("train")
    diag_idx = np.load(C.SPLITS / "diag_idx.npy")

    # ---------------- model: identical init for B and C given the seed
    seed_everything(a.seed)
    model = build_student(a.size).to(DEVICE)
    if a.row == "B":
        from lightly.loss import VICRegLoss
        head = vicreg_projector(model.num_features).to(DEVICE)
        crit = VICRegLoss()
    else:
        head = distill_projector(model.num_features, tfeat.shape[-1]).to(DEVICE)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = n // a.batch
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total, C.WARMUP_EPOCHS * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler())

    start_epoch, train_seconds = 0, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=DEVICE)
        model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, train_seconds = ck["epoch"] + 1, ck["train_seconds"]
        print(f"[{name}] resuming at epoch {start_epoch}")

    meta = {"name": name, "row": a.row, "size": a.size, "seed": a.seed,
            "data_frac": a.data_frac, "n_unlabeled": n, "k_views": K,
            "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "wd": a.wd,
            "steps_per_epoch": steps_per_epoch, "total_steps": total,
            "img_size": C.IMG_SIZE, "params_backbone": n_params(model),
            "params_head": n_params(head), "feat_dim": model.num_features,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"}
    print(json.dumps(meta, indent=1))
    log_path = out / "log.csv"
    new_log = not log_path.exists() or start_epoch == 0
    logf = open(log_path, "w" if new_log else "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["epoch", "loss", "lr", "epoch_seconds", "knn_acc", "emb_std",
                       "emb_std_x_sqrt_d", "kmeans_nmi"])

    def batch_x(idx, k):
        x = to_device_uint8(views[idx, k])                   # (B,3,S,S) in [0,1]
        return resize_normalize(x, C.IMG_SIZE, C.STUDENT_MEAN, C.STUDENT_STD)

    for epoch in range(start_epoch, a.epochs):
        model.train(); head.train()
        rng = np.random.default_rng([a.seed, epoch])        # order/view choice: seed+epoch
        perm = rng.permutation(n)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0, run = time.time(), 0.0
        for st in range(steps_per_epoch):
            idx = np.sort(perm[st * a.batch:(st + 1) * a.batch])
            k1 = rng.integers(K, size=len(idx))
            with autocast():
                if a.row == "B":
                    k2 = (k1 + rng.integers(1, K, size=len(idx))) % K   # a different view
                    z1 = head(model(batch_x(idx, k1)))
                    z2 = head(model(batch_x(idx, k2)))
                    loss = crit(z1.float(), z2.float())
                else:
                    t = tfeat[torch.from_numpy(idx), torch.from_numpy(k1)].to(DEVICE, non_blocking=True)
                    s = head(model(batch_x(idx, k1)))
                    loss = 1 - F.cosine_similarity(s.float(), t, dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if st % 10 == 0:
                print(f"\r[{name}] ep {epoch + 1}/{a.epochs} step {st}/{steps_per_epoch} "
                      f"loss {loss.item():.4f}", end="", flush=True)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_seconds += dt
        row = [epoch + 1, run / steps_per_epoch, sched.get_last_lr()[0], dt]
        last = epoch + 1 == a.epochs
        if (epoch + 1) % C.DIAG_EVERY == 0 or last:
            d = collapse_check(model, x_tr, y_tr, diag_idx)
            row += [d["knn_acc"], d["emb_std"], d["emb_std_x_sqrt_d"], d["kmeans_nmi"]]
            print(f"\n[{name}] collapse check: {d}")
        else:
            row += ["", "", "", ""]
            print()
        logw.writerow(row); logf.flush()
        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "train_seconds": train_seconds}, out / "last.pt")

    logf.close()
    if "d" not in locals():                      # resumed after the last epoch had finished
        d = collapse_check(model, x_tr, y_tr, diag_idx)
    meta["train_gpu_hours"] = train_seconds / 3600
    meta["final_collapse_check"] = d
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out / "backbone.pt")      # projector is discarded
    (out / "last.pt").unlink()
    print(f"[{name}] done in {train_seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
