"""Parameters, FLOPs, latency of every encoder + GPU-hours of every training.

    python -m src.cost            # -> results/cost_models.csv, results/cost_runs.csv

Latency: fixed device (this GPU), fixed batch (LAT_BATCH), mixed precision as in
training, eval mode, LAT_WARMUP warm-up iterations then the mean over LAT_ITERS.
"""
import json
import time

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from . import config as C
from .common import AMP_DTYPE, DEVICE, autocast
from .models import build_student, load_teacher, n_params

LAT_BATCH, LAT_WARMUP, LAT_ITERS = 128, 10, 50


def flops_per_image(model, size):
    x = torch.randn(1, 3, size, size, device=DEVICE)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


@torch.no_grad()
def latency_ms(model, size):
    x = torch.randn(LAT_BATCH, 3, size, size, device=DEVICE).contiguous(
        memory_format=torch.channels_last)
    model.eval()
    for _ in range(LAT_WARMUP):
        with autocast():
            model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(LAT_ITERS):
        with autocast():
            model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / LAT_ITERS * 1000


def main():
    rows = []
    encs = [(f"student_{s}", build_student(s), C.IMG_SIZE) for s in C.SIZES]
    teacher, *_ = load_teacher()
    encs.append(("teacher", teacher, C.TEACHER_SIZE))
    for name, m, size in encs:
        m = m.to(DEVICE).eval()
        lat = latency_ms(m, size)
        rows.append({"encoder": name, "input_size": size, "params_M": n_params(m) / 1e6,
                     "gflops_per_image": flops_per_image(m, size) / 1e9,
                     "feat_dim": m.num_features,
                     f"latency_ms_batch{LAT_BATCH}": lat,
                     "throughput_img_s": LAT_BATCH / lat * 1000,
                     "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
                     "amp": str(AMP_DTYPE or "fp32")})
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "cost_models.csv", index=False)
    print(df.to_string(index=False))

    runs = []
    for r in sorted(C.RUNS.iterdir()):
        if (r / "meta.json").exists():
            m = json.loads((r / "meta.json").read_text())
            runs.append({k: m[k] for k in ("name", "row", "size", "seed", "data_frac",
                                            "epochs", "total_steps", "train_gpu_hours")})
    vm = C.CACHE / "views_meta.json"
    if vm.exists():
        v = json.loads(vm.read_text())
        runs.append({"name": "teacher_forward_cache", "row": "C(shared)",
                     "train_gpu_hours": v["teacher_forward_gpu_hours"]})
    if runs:
        dr = pd.DataFrame(runs)
        dr.to_csv(C.RESULTS / "cost_runs.csv", index=False)
        print(dr.to_string(index=False))


if __name__ == "__main__":
    main()
