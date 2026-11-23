"""Data pipeline.

    python -m src.data splits   # .npy caches + every fixed index file (run once)
    python -m src.data views    # K augmented views per unlabeled image + teacher features

Split usage (state this in the report):
  * pretraining / distillation: N_UNLABELED images of the TRAIN split, labels discarded;
  * probes: label fractions of the TRAIN split;
  * testing: official TEST split, never touched before probing.
The MedMNIST validation split is not used.
"""
import argparse
import json
import pickle
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .common import DEVICE, autocast, resize_normalize, to_device_uint8


# ------------------------------------------------------------------ raw arrays
def _npy(split, what):
    return C.DATA / f"{split}_{what}.npy"


def load_split(split):
    """Returns (images uint8 NHWC memmap, labels int64). Converts the npz once."""
    if not _npy(split, "images").exists():
        print(f"[data] converting {C.NPZ.name}:{split} -> .npy (one-off)")
        z = np.load(C.NPZ)
        np.save(_npy(split, "images"), z[f"{split}_images"])
        np.save(_npy(split, "labels"), z[f"{split}_labels"].reshape(-1).astype(np.int64))
    return (np.load(_npy(split, "images"), mmap_mode="r"),
            np.load(_npy(split, "labels")))


def probe_pool():
    """Indices of the train split used as 'the labeled training set' (all, by default)."""
    path = C.SPLITS / "probe_pool.npy"
    return np.load(path)


# ------------------------------------------------------------------ fixed splits
def make_splits():
    _, y_tr = load_split("train")
    _, y_te = load_split("test")
    print(f"[data] train {len(y_tr)}  test {len(y_te)}")
    print(f"[data] train class counts {np.bincount(y_tr).tolist()}")
    print(f"[data] test  class counts {np.bincount(y_te).tolist()}")

    # 1) unlabeled subset (nested: the 50% ablation is its first half)
    rng = np.random.default_rng(C.SPLIT_SEED)
    unl = rng.permutation(len(y_tr))[:C.N_UNLABELED]
    np.save(C.SPLITS / "unlabeled_idx.npy", unl)

    # 2) probe pool = "labeled training set"
    if C.PROBE_POOL and C.PROBE_POOL < len(y_tr):
        pool, _ = train_test_split(np.arange(len(y_tr)), train_size=C.PROBE_POOL,
                                   stratify=y_tr, random_state=C.SPLIT_SEED)
        pool = np.sort(pool)
    else:
        pool = np.arange(len(y_tr))
    np.save(C.SPLITS / "probe_pool.npy", pool)

    # 3) label subsets: generated ONCE, stratified, shared by every encoder.
    #    Indices are positions inside the probe pool (Section 3.1 of the statement).
    y_pool = y_tr[pool]
    subsets = {}
    for frac in C.LABEL_FRACS:
        for draw in range(C.N_DRAWS):
            if frac == 1.0:
                subsets[(frac, draw)] = np.arange(len(y_pool))
            else:
                idx, _ = train_test_split(np.arange(len(y_pool)), train_size=frac,
                                          stratify=y_pool, random_state=draw)
                subsets[(frac, draw)] = np.sort(idx)
    with open(C.SPLITS / "label_subsets.pkl", "wb") as f:
        pickle.dump(subsets, f)

    # 4) small fixed labeled set from TRAIN for collapse checks during training
    diag, _ = train_test_split(np.arange(len(y_tr)), train_size=min(5000, len(y_tr) // 2),
                               stratify=y_tr, random_state=123)
    np.save(C.SPLITS / "diag_idx.npy", np.sort(diag))

    info = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "n_unlabeled": int(len(unl)), "probe_pool": int(len(pool)),
            "label_subset_sizes": {f"{k[0]}": int(len(v)) for k, v in subsets.items()}}
    (C.SPLITS / "splits_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# ------------------------------------------------------------------ augmented views
def view_transform():
    """The training transform of row B (VICReg, from lightly), applied at native 128 px.
    Histology has no canonical orientation, so vertical flips and 90-degree rotations
    are added; min_scale is raised from 0.08 because the images are small."""
    from lightly.transforms.vicreg_transform import VICRegViewTransform
    return VICRegViewTransform(input_size=C.NATIVE_SIZE, min_scale=C.VIEW_MIN_SCALE,
                               vf_prob=0.5, hf_prob=0.5, rr_prob=0.5, normalize=None)


class _ViewDS(Dataset):
    def __init__(self, images, idx, k, base_seed=C.SPLIT_SEED):
        self.images, self.idx, self.k, self.base = images, idx, k, base_seed
        self.t = view_transform()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = self.base * 10_000_019 + int(self.idx[i])  # per-image seed: reproducible
        random.seed(s); np.random.seed(s % 2**32); torch.manual_seed(s)
        img = Image.fromarray(np.asarray(self.images[self.idx[i]]))
        views = [self.t(img) for _ in range(self.k)]          # K x (3,128,128) in [0,1]
        return torch.stack(views).mul_(255).round_().to(torch.uint8)


@torch.no_grad()
def make_views(batch=32, workers=8):
    """Generates K views per unlabeled image ONCE (shared by B and C, every seed and
    size), stores them at the student resolution, and caches the teacher's features of
    exactly the same views (128 px view -> 224 px for the teacher)."""
    from .models import load_teacher

    x_tr, _ = load_split("train")
    unl = np.load(C.SPLITS / "unlabeled_idx.npy")
    teacher, t_mean, t_std, t_dim = load_teacher()
    teacher = teacher.to(DEVICE)

    n, k, s = len(unl), C.K_VIEWS, C.IMG_SIZE
    views = np.lib.format.open_memmap(C.CACHE / f"views_{s}.npy", mode="w+",
                                      dtype=np.uint8, shape=(n, k, 3, s, s))
    tfeat = np.lib.format.open_memmap(C.CACHE / "teacher_feats.npy", mode="w+",
                                      dtype=np.float16, shape=(n, k, t_dim))
    loader = DataLoader(_ViewDS(x_tr, unl, k), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=DEVICE.type == "cuda")
    t_teacher, t0, pos = 0.0, time.time(), 0
    for vb in loader:                                     # (B,K,3,128,128) uint8
        b = vb.shape[0]
        flat = vb.flatten(0, 1).to(DEVICE, non_blocking=True).float().div_(255)
        # student copy at its own resolution, stored as uint8
        small = torch.nn.functional.interpolate(flat, size=(s, s), mode="bilinear",
                                                antialias=True, align_corners=False)
        views[pos:pos + b] = (small.mul(255).round().clamp(0, 255).to(torch.uint8)
                              .view(b, k, 3, s, s).cpu().numpy())
        # teacher features of the very same views
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        tt = time.time()
        with autocast():
            f = teacher(resize_normalize(flat, C.TEACHER_SIZE, t_mean, t_std))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_teacher += time.time() - tt
        tfeat[pos:pos + b] = f.float().view(b, k, t_dim).cpu().numpy().astype(np.float16)
        pos += b
        print(f"\r[views] {pos}/{n} images  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    views.flush(); tfeat.flush()
    meta = {"n": n, "k": k, "img_size": s, "view_size": C.NATIVE_SIZE,
            "teacher": C.TEACHER_NAME, "teacher_dim": t_dim,
            "teacher_forward_gpu_hours": t_teacher / 3600,
            "total_seconds": time.time() - t0, "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"}
    (C.CACHE / "views_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["splits", "views"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32, help="images per batch (x K views)")
    a = ap.parse_args()
    if a.step == "splits":
        make_splits()
    else:
        make_views(batch=a.batch, workers=a.workers)
