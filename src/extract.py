"""Frozen-feature extraction for every encoder, cached to features/<tag>.npz.

    python -m src.extract --row D                 # teacher (+ sanity check)
    python -m src.extract --row A --size S --seed 0
    python -m src.extract --row B --size S --seed 0 [--data-frac 0.5]
    python -m src.extract --all                   # D, every A seed, every finished B/C run

Same deterministic eval transform for every encoder: native 128 px image ->
bilinear antialiased resize to the encoder's input size -> normalize. Nothing random.
Teacher feature = final-norm [CLS] token (timm, num_classes=0).
"""
import argparse
import json

import numpy as np
import torch

from . import config as C
from .common import DEVICE, seed_everything
from .data import load_split, probe_pool
from .diag import diagnose, encode
from .models import build_student, load_teacher


def get_encoder(row, size=None, seed=None, data_frac=1.0):
    """Returns (model, input_size, mean, std)."""
    if row == "D":
        m, mean, std, _ = load_teacher()
        return m.to(DEVICE), C.TEACHER_SIZE, mean, std
    if row == "A":
        seed_everything(seed)                       # same init as B/C with this seed
        m = build_student(size)
    else:
        m = build_student(size)
        sd = torch.load(C.RUNS / C.tag(row, size, seed, data_frac) / "backbone.pt",
                        map_location="cpu")
        m.load_state_dict(sd)
    return m.to(DEVICE), C.IMG_SIZE, C.STUDENT_MEAN, C.STUDENT_STD


def extract(row, size=None, seed=None, data_frac=1.0, force=False):
    t = C.tag(row, size, seed, data_frac)
    path = C.FEATS / f"{t}.npz"
    if path.exists() and not force:
        print(f"[extract] {t} cached"); return path
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    pool = probe_pool()
    model, sz, mean, std = get_encoder(row, size, seed, data_frac)
    bs = 128 if row == "D" else 512
    f_tr = encode(model, x_tr, sz, mean, std, idx=pool, bs=bs)
    f_te = encode(model, x_te, sz, mean, std, bs=bs)
    np.savez(path, train=f_tr.astype(np.float32), y_train=y_tr[pool],
             test=f_te.astype(np.float32), y_test=y_te)
    print(f"[extract] {t}: train {f_tr.shape} test {f_te.shape} -> {path.name}")
    if row in ("D", "A"):
        d = diagnose(f_tr, y_tr[pool], f_te, y_te)
        print(f"[sanity] {t}: {json.dumps(d)}")
        (C.RESULTS / f"sanity_{t}.json").write_text(json.dumps(d, indent=2))
    return path


def all_encoders():
    jobs = [("D", None, None, 1.0)]
    for size, seeds in C.SEEDS.items():
        jobs += [("A", size, s, 1.0) for s in seeds]
    for run in sorted(C.RUNS.iterdir()):
        if (run / "backbone.pt").exists():
            p = C.parse_tag(run.name)
            jobs.append((p["row"], p["size"], p["seed"], p["data_frac"]))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", choices=["A", "B", "C", "D"])
    ap.add_argument("--size", choices=list(C.SIZES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--data-frac", type=float, default=1.0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.all:
        for j in all_encoders():
            extract(*j, force=a.force)
    else:
        extract(a.row, a.size, a.seed, a.data_frac, force=a.force)
