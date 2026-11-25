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
# dataset (4.3 GB) -> data/pathmnist_128.npz
mkdir -p data && python -c "from medmnist import PathMNIST; PathMNIST(split='train', size=128, download=True, root='data')"
```

## Run

Open `notebooks/pipeline.ipynb` and run the sections in order. Every value comes from
`src/config.py`. Everything is cached and resumable: re-running a cell skips finished
work, and an interrupted training resumes from its last epoch.

| Section | What it does | Output |
|---|---|---|
| 1. Data | `.npy` caches + fixed splits | `splits/` (unlabeled subset, label subsets, diag set) |
| 2. Views + teacher | K augmented views per image + teacher features of those views | `cache/views_64.npy`, `cache/teacher_feats.npy` |
| 3. Row D | teacher features + sanity check | `features/D.npz`, `results/sanity_D.json` |
| 4. Row A | random-init student features | `features/A_*.npz`, `results/sanity_A_*.json` |
| 5. Rows B, C | one training each (`SIZE`, `SEED`) + loss/collapse curves | `runs/<tag>/` |
| 6. Probes | SVM C/γ chosen once on B, then k-NN / linear / RBF + k-means | `results/probe_hparams.json`, `results/probes/`, `results/clusters/` |
| 7. Projection | UMAP of the four encoders | `figures/projection_S.png` |
| 8. Cost | params, FLOPs, latency, GPU-hours | `results/cost_models.csv`, `results/cost_runs.csv` |
| 9. Full study | every B/C run (main, then ablation), features, probes, cost, report | `results/main_table.md`, `results/*.csv`, `figures/*.png` |

Tags: `A_S_s0`, `B_L_s1_d100` (row, size, seed, % of unlabeled data), `D`.

The budget was sized with `python -m src.benchmark` (throughput of view generation, of
one B/C training step per student size, and of the teacher forward, on random data).
If a reduction is ever needed, change `src/config.py` in this order: `N_UNLABELED`,
then `EPOCHS`, then `IMG_SIZE` — never the seeds — and keep it identical for B and C.

Not versioned (`.gitignore`): `data/`, `cache/`, `runs/`, `features/`.

## Design decisions

**Splits.** Pretraining/distillation: the whole *train* split (89,996 images), labels
discarded (nested 44,998 for the 50% ablation). Probes: 1/10/50/100% of the train split,
stratified, 3 draws per fraction, generated once (`splits/label_subsets.pkl`) and shared
by every encoder. Test: official test split (7,180), used only for probing, k-means and
projections. MedMNIST's val split is not used. Pretraining and probe images overlap
(standard; no labels are used in pretraining).

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
384-d features). Teacher: 21.7 M params, 384-d. FLOPs/latency in `results/cost_models.csv`.

**Budget (identical in B and C).** 40 epochs, batch 256, AdamW (lr 1e-3, wd 0.05),
4 warm-up epochs then cosine, bf16 autocast, `channels_last`, same seed → same
initialization (row A uses that same initialization). Projectors: VICReg expander
3×2048; distillation projector Linear-BN-GELU-Linear (2048 hidden). Seeds: 3 for S,
2 for L. Hardware: one RTX 3050 Laptop (4 GB).

**Probes.** L2-normalize → StandardScaler fitted on train-split features. k-NN
(cosine, k = min(20, smallest class count in the subset)); LinearSVC; RBF SVC
(subsampled to 10,000 above that size, so 50% and 100% labels are equivalent for it).
C and γ chosen once by 3-fold CV inside the train split on `B_S_s0_d100` at 100% labels
(grids: linear C ∈ {0.001, 0.01, 0.1, 1}; RBF C ∈ {0.1, 1, 10, 100}, γ = m/d with
m ∈ {0.5, 1, 2, 4, 8}), then frozen for every encoder.
k-means (k = 9) NMI/ARI and UMAP on test features, same parameters for all encoders.

**Collapse checks.** Every 10 epochs and at the end of each training: k-NN accuracy,
per-dimension std of L2-normalized embeddings and k-means NMI on a fixed 5,000-image
labeled subset of the *train* split (monitoring only) → `runs/<tag>/log.csv`,
`results/collapse_checks.csv`.

**Caveats to discuss.** C vs B differs in supervision source *and* objective (DINO
teacher vs VICReg student). Probe capacity differs with feature dimension (128 / 384 /
384). Differences smaller than the seed-to-seed spread are inconclusive. Teacher
pretraining cost: quote it from the DINO paper (Caron et al., 2021) — **TODO: fill in
GPU-hours for ViT-S/16** — and add the teacher-forward GPU-hours from
`cache/views_meta.json`.

## Extensions (`notebooks/extensions.ipynb`)

Beyond the required study; same code, same probes, results land in the same `results/`
and `figures/`.

- **Row P — raw pixels:** no encoder; the image resized to 32×32 and flattened (3072-d)
  goes through the same L2 + StandardScaler + probes. Separates what row A gets from the
  architecture from what is already in the pixels.
- **Extra student sizes (stretch goal):** M = ×1.0 (2.8 M params) and XL = ×1.8
  (9.1 M) of the same family, main comparison only (100% unlabeled data, 2 seeds, no
  ablation), so the C − B gap vs. parameters has four points. Declared in `src/config.py`:

```python
EXTRA_SIZES = {"M": 1.0, "XL": 1.8}
EXTRA_SEEDS = {"M": [0, 1], "XL": [0, 1]}
```

## Required ablation

Amount of unlabeled data: 100% vs 50% (nested subset), same epochs, both sizes, same
seeds and probes → `results/gap.csv` (rows with `data_frac` 0.5 / 1.0) and
`figures/ablation_gap.png`.
