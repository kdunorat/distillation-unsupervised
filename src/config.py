"""Central configuration. Every value can be overridden by an environment
variable with the same name prefixed by ``DU_`` (e.g. ``DU_EPOCHS=20``), so the
whole pipeline can be shrunk for a smoke test without editing code."""
import os
from pathlib import Path


def _env(name, default, cast=None):
    raw = os.environ.get(f"DU_{name}")
    if raw is None:
        return default
    cast = cast or type(default)
    return cast(raw)


ROOT = Path(_env("ROOT", str(Path(__file__).resolve().parents[1]), str))
DATA = ROOT / "data"          # raw dataset (pathmnist_128.npz) and .npy caches
SPLITS = ROOT / "splits"      # fixed indices: unlabeled subset, label subsets
CACHE = ROOT / "cache"        # pre-generated augmented views + teacher features
RUNS = ROOT / "runs"          # one folder per student training
FEATS = ROOT / "features"     # frozen features per encoder (train/test)
RESULTS = ROOT / "results"    # csv tables
FIGS = ROOT / "figures"

NPZ = DATA / _env("NPZ_NAME", "pathmnist_128.npz", str)
NUM_CLASSES = 9
NATIVE_SIZE = 128             # resolution of the npz images

# ---------------------------------------------------------------- unlabeled data
N_UNLABELED = _env("N_UNLABELED", 89996)   # subset of the train split, labels discarded
K_VIEWS = _env("K_VIEWS", 8)               # pre-generated augmented views per image
DATA_FRACS = [1.0, 0.5]                    # ablation: nested 100% / 50% of the subset
SPLIT_SEED = 0

# ---------------------------------------------------------------- resolution
IMG_SIZE = _env("IMG_SIZE", 64)            # student input (views are made at 128, then resized)
VIEW_MIN_SCALE = _env("VIEW_MIN_SCALE", 0.2)

# ---------------------------------------------------------------- teacher (row D)
TEACHER_NAME = _env("TEACHER_NAME", "vit_small_patch16_224.dino", str)
TEACHER_SIZE = 224
TEACHER_PRETRAINED = bool(_env("TEACHER_PRETRAINED", 1, int))  # 0 only for offline smoke tests

# ---------------------------------------------------------------- student family
# one scaling rule: width multiplier on the channel widths, everything else fixed
BASE_WIDTHS = (32, 64, 128, 256)
DEPTHS = (2, 2, 2, 2)
SIZES = {"S": 0.5, "L": 1.5}
SEEDS = {"S": [0, 1, 2], "L": [0, 1]}
STUDENT_MEAN = (0.485, 0.456, 0.406)       # ImageNet stats, same for every student
STUDENT_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- training budget (identical for B and C)
EPOCHS = _env("EPOCHS", 40)
BATCH = _env("BATCH", 256)
LR = _env("LR", 1e-3)
WD = _env("WD", 0.05)
WARMUP_EPOCHS = _env("WARMUP_EPOCHS", 4)
PROJ_HIDDEN = _env("PROJ_HIDDEN", 2048)    # hidden size of both projectors
VICREG_OUT = _env("VICREG_OUT", 2048)
DIAG_EVERY = _env("DIAG_EVERY", 10)        # epochs between collapse checks

# ---------------------------------------------------------------- probes
LABEL_FRACS = [0.01, 0.10, 0.50, 1.00]
N_DRAWS = 3
KNN_MAX_K = 20
RBF_CAP = _env("RBF_CAP", 10000)           # RBF SVM is subsampled above this many samples
PROBE_POOL = _env("PROBE_POOL", 0)          # 0 = full train split; >0 = fixed stratified pool
N_JOBS = _env("N_JOBS", 4)


def tag(row, size=None, seed=None, data_frac=1.0):
    """Canonical encoder name: D | A_S_s0 | B_L_s1_d100 | C_S_s2_d50."""
    if row == "D":
        return "D"
    if row == "A":
        return f"A_{size}_s{seed}"
    return f"{row}_{size}_s{seed}_d{int(round(data_frac * 100))}"


def parse_tag(t):
    parts = t.split("_")
    out = {"row": parts[0], "size": None, "seed": None, "data_frac": 1.0}
    if len(parts) >= 3:
        out["size"], out["seed"] = parts[1], int(parts[2][1:])
    if len(parts) == 4:
        out["data_frac"] = int(parts[3][1:]) / 100
    return out


for _d in (DATA, SPLITS, CACHE, RUNS, FEATS, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)
