"""Small shared helpers: device, mixed precision, seeding, GPU preprocessing."""
import random

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
elif DEVICE.type == "cuda":
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = None  # CPU: plain fp32


def autocast():
    if AMP_DTYPE is None:
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=AMP_DTYPE)


def needs_scaler():
    return AMP_DTYPE == torch.float16


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device_uint8(x):
    """numpy uint8 batch -> float tensor in [0,1] on DEVICE, NCHW.
    Accepts NHWC (raw dataset) or NCHW (cached views)."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if DEVICE.type == "cuda":
        t = t.pin_memory().to(DEVICE, non_blocking=True)
    if t.shape[-1] == 3:  # NHWC -> NCHW
        t = t.permute(0, 3, 1, 2)
    return t.float().div_(255.0)


def resize_normalize(x, size, mean, std):
    """Deterministic: bilinear (antialiased) resize to ``size`` then normalize."""
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True,
                          align_corners=False)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous(memory_format=torch.channels_last)
