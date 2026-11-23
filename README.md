# Distilling Unsupervised Representations into Small Models — PathMNIST

Controlled, budget-matched comparison (MO433 project):

| Row | Encoder | Notes |
|---|---|---|
| A | student, random init | BatchNorm in `eval()` with default running stats (mean 0, var 1) |
| B | student trained from scratch with **VICReg** (lightly) | same views, epochs, steps, batch, optimizer as C |
| C | same student distilled from the frozen teacher | cosine loss on projected features, projector discarded |
| D | frozen teacher, **DINO ViT-S/16** (`timm: vit_small_patch16_224.dino`) | feature = final-norm [CLS] token |

Dataset: **PathMNIST** (MedMNIST v2, 128 px, 9 classes) — outside ImageNet, so the
teacher has not seen these images.

## Setup

```bash
pip install -r requirements.txt      # keeps your existing torch/torchvision
# dataset (4.3 GB): data/pathmnist_128.npz
python -c "from medmnist import PathMNIST; PathMNIST(split='train', size=128, download=True, root='data')"
```

## Run

Everything is cached and resumable (re-running a command skips finished work; an
interrupted training resumes from its last epoch). Run inside `tmux`.

```bash
python -m src.benchmark            # throughput first (random data, ~2 min)
bash scripts/run_all.sh            # the whole study, in order
```

Step by step (what `run_all.sh` does):

| Step | Command | Output |
|---|---|---|
| splits | `python -m src.data splits` | `splits/` (unlabeled subset, label subsets, diag set) |
| views + teacher cache | `python -m src.data views` | `cache/views_64.npy`, `cache/teacher_feats.npy` |
| teacher sanity | `python -m src.extract --row D` | `results/sanity_D.json` |
| train | `python -m src.train --row {B,C} --size {S,L} --seed N [--data-frac 0.5]` | `runs/<tag>/` |
| features | `python -m src.extract --all` | `features/<tag>.npz` |
| SVM hparams (once) | `python -m src.probe select --tag B_S_s0_d100` | `results/probe_hparams.json` |
| probes + k-means | `python -m src.probe run --all` | `results/probes/`, `results/clusters/` |
| cost | `python -m src.cost` | `results/cost_models.csv`, `results/cost_runs.csv` |
| report | `python -m src.report` | `results/main_table.md`, `results/*.csv`, `figures/*.png` |

Tags: `A_S_s0`, `B_L_s1_d100` (row, size, seed, % of unlabeled data), `D`.

Any value in `src/config.py` can be overridden with an env var, e.g.
`DU_EPOCHS=20 DU_N_UNLABELED=10000 bash scripts/run_all.sh`. If a budget reduction is
needed, reduce in this order: `DU_N_UNLABELED`, then `DU_EPOCHS`, then `DU_IMG_SIZE`
— never the seeds — and keep it identical for B and C.

Quick smoke test on the real data (separate folder, minutes):
```bash
mkdir -p /tmp/du/data && ln -s $PWD/data/pathmnist_128.npz /tmp/du/data/
export DU_ROOT=/tmp/du DU_N_UNLABELED=512 DU_EPOCHS=2 DU_K_VIEWS=2 DU_PROBE_POOL=3000
python -m src.data splits && python -m src.data views && python -m src.extract --row D
python -m src.train --row B --size S --seed 0 && python -m src.train --row C --size S --seed 0
unset DU_ROOT DU_N_UNLABELED DU_EPOCHS DU_K_VIEWS DU_PROBE_POOL
```

## Design decisions (state these in the report)

**Splits.** Pretraining/distillation: `N_UNLABELED` = 15,000 images of the *train*
split, labels discarded (nested 7,500 for the 50% ablation). Probes: 1/10/50/100% of the
train split, stratified, 3 draws per fraction, generated once (`splits/label_subsets.pkl`)
and shared by every encoder. Test: official test split (7,180), used only for probing,
k-means and projections. MedMNIST's val split is not used. Pretraining and probe images
overlap (standard; no labels are used in pretraining).

**Augmentation and teacher caching.** One training transform: lightly's
`VICRegViewTransform` at native 128 px (min_scale 0.2, horizontal + vertical flips and
90° rotations since histology has no canonical orientation, colour jitter, grayscale,
solarize, blur). `K = 8` views per image are generated once with per-image seeds.
The student sees each view resized to 64 px; the teacher sees *the same view* resized
to 224 px; teacher features are computed once and reused by every C run. Row B samples
two different views of the same K per image, row C one view, so both rows use the same
finite set of augmented images.

**Student family** (`src/models.py`). Residual convnet: 3×3 stride-2 stem (no max-pool:
a ResNet 7×7/2 + max-pool stem would divide a 64 px image by 4 immediately), four stages
of two basic blocks (3×3 convs, BatchNorm, ReLU), stages 2–4 downsample by 2
(64 → 32, 16, 8, 4), global average pooling, no head. Single scaling rule: channel
widths (32, 64, 128, 256) × width multiplier, everything else fixed.
S = ×0.5 (16–128 ch, 0.70 M params, 128-d features); L = ×1.5 (48–384 ch, 6.28 M params,
384-d features). Teacher: 21.7 M params, 384-d. Report FLOPs/latency from `cost_models.csv`.

**Budget (identical in B and C).** 30 epochs, batch 256, AdamW (lr 1e-3, wd 0.05),
3 warm-up epochs then cosine, bf16 autocast, `channels_last`, same seed → same
initialization (row A uses that same initialization). Projectors: VICReg expander
3×2048; distillation projector Linear-BN-GELU-Linear (2048 hidden). Seeds: 3 for S,
2 for L.

**Probes.** L2-normalize → StandardScaler fitted on train-split features. k-NN
(cosine, k = min(20, smallest class count in the subset)); LinearSVC; RBF SVC
(subsampled to 10,000 above that size). C and γ chosen once by 3-fold CV inside the
train split on `B_S_s0_d100` at 100% labels, then frozen for every encoder.
k-means (k = 9) NMI/ARI and UMAP on test features, same parameters for all encoders.

**Collapse checks.** Every 10 epochs and at the end of each training: k-NN accuracy,
per-dimension std of L2-normalized embeddings and k-means NMI on a fixed 5,000-image
labeled subset of the *train* split (monitoring only) → `runs/<tag>/log.csv`,
`results/collapse_checks.csv`.

**Caveats to discuss.** C vs B differs in supervision source *and* objective (DINO
teacher vs VICReg student). Probe capacity differs with feature dimension (128 / 384 /
384). Short schedules: results are undertrained and noisy; differences smaller than the
seed-to-seed spread are inconclusive. Teacher pretraining cost: quote it from the DINO
paper (Caron et al., 2021) — **TODO: fill in GPU-hours for ViT-S/16** — and add the
teacher-forward GPU-hours from `cache/views_meta.json`.

## Required ablation

Amount of unlabeled data: 100% vs 50% (nested subset), same epochs, both sizes, same
seeds and probes → `results/gap.csv` (rows with `data_frac` 0.5 / 1.0) and
`figures/ablation_gap.png`.
