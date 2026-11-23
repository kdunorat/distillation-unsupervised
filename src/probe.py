"""Probes and representation characterization, identical for all four rows.

    python -m src.probe select --tag B_S_s0_d100   # choose SVM C / gamma ONCE, then frozen
    python -m src.probe run --all                   # every cached feature file

Feature preprocessing (everywhere): L2-normalize, then StandardScaler fitted on the
training-split features only. Probes: k-NN (cosine, k = min(20, n_per_class), never
tuned), LinearSVC, RBF SVC (subsampled to RBF_CAP above that size). k-means NMI/ARI and
the 2-D projection use the TEST features.
"""
import argparse
import csv
import json
import pickle

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import config as C
from .diag import l2n

HP_PATH = C.RESULTS / "probe_hparams.json"
LIN_GRID = [0.001, 0.01, 0.1, 1.0]
RBF_C_GRID = [1.0, 10.0, 100.0]
RBF_GAMMA_MULT = [0.5, 1.0, 2.0]          # gamma = mult / n_features (features are standardized)


def load(tag):
    z = np.load(C.FEATS / f"{tag}.npz")
    sc = StandardScaler().fit(l2n(z["train"]))
    return (sc.transform(l2n(z["train"])), z["y_train"],
            sc.transform(l2n(z["test"])), z["y_test"])


def _cap(idx, y, cap, seed):
    if len(idx) <= cap:
        return idx
    sub, _ = train_test_split(idx, train_size=cap, stratify=y[idx], random_state=seed)
    return np.sort(sub)


# ------------------------------------------------------------------ hyper-parameters
def select(tag):
    """3-fold CV inside the training split only (never the test split), on one
    encoder (row B at 100% labels), on a stratified RBF_CAP subsample. Then frozen."""
    xtr, ytr, _, _ = load(tag)
    idx = _cap(np.arange(len(ytr)), ytr, C.RBF_CAP, 0)
    x, y = xtr[idx], ytr[idx]
    d = x.shape[1]
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    lin = GridSearchCV(LinearSVC(dual="auto", max_iter=5000), {"C": LIN_GRID}, cv=cv,
                       n_jobs=C.N_JOBS).fit(x, y)
    rbf = GridSearchCV(SVC(kernel="rbf"),
                       {"C": RBF_C_GRID, "gamma": [m / d for m in RBF_GAMMA_MULT]},
                       cv=cv, n_jobs=C.N_JOBS).fit(x, y)
    hp = {"selected_on": tag, "n_cv": int(len(y)),
          "linear_C": lin.best_params_["C"], "linear_cv_acc": lin.best_score_,
          "rbf_C": rbf.best_params_["C"],
          "rbf_gamma_mult": rbf.best_params_["gamma"] * d, "rbf_cv_acc": rbf.best_score_}
    HP_PATH.write_text(json.dumps(hp, indent=2))
    print(json.dumps(hp, indent=2))


# ------------------------------------------------------------------ probes
def _fit_one(probe, xtr, ytr, xte, yte, idx, draw, hp):
    y = ytr[idx]
    if probe == "knn":
        k = int(min(C.KNN_MAX_K, np.bincount(y, minlength=C.NUM_CLASSES).min()))
        clf = KNeighborsClassifier(n_neighbors=max(k, 1), metric="cosine", algorithm="brute")
    elif probe == "linear":
        clf = LinearSVC(C=hp["linear_C"], dual="auto", max_iter=5000)
    else:
        idx = _cap(idx, ytr, C.RBF_CAP, draw)
        y = ytr[idx]
        clf = SVC(kernel="rbf", C=hp["rbf_C"], gamma=hp["rbf_gamma_mult"] / xtr.shape[1])
    clf.fit(xtr[idx], y)
    return probe, len(idx), float((clf.predict(xte) == yte).mean())


def run(tag, force=False):
    out = C.RESULTS / "probes" / f"{tag}.csv"
    out.parent.mkdir(exist_ok=True)
    if out.exists() and not force:
        print(f"[probe] {tag} cached"); return
    hp = json.loads(HP_PATH.read_text())
    with open(C.SPLITS / "label_subsets.pkl", "rb") as f:
        subsets = pickle.load(f)
    xtr, ytr, xte, yte = load(tag)
    tasks = []
    for (frac, draw), idx in subsets.items():
        if frac == 1.0 and draw > 0:      # the 100% "draws" are identical
            continue
        for probe in ("knn", "linear", "rbf"):
            tasks.append((frac, draw, probe, idx))
    res = Parallel(n_jobs=C.N_JOBS)(
        delayed(_fit_one)(p, xtr, ytr, xte, yte, idx, d, hp) for fr, d, p, idx in tasks)
    meta = C.parse_tag(tag)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "row", "size", "seed", "data_frac", "label_frac", "draw",
                    "probe", "n_train", "acc"])
        for (fr, d, _, _), (p, n, acc) in zip(tasks, res):
            w.writerow([tag, meta["row"], meta["size"], meta["seed"], meta["data_frac"],
                        fr, d, p, n, acc])
    # unsupervised characterization on the test split
    km = KMeans(n_clusters=C.NUM_CLASSES, n_init=10, random_state=0).fit(xte)
    cl = {"tag": tag, **meta, "feat_dim": int(xte.shape[1]),
          "kmeans_nmi": float(normalized_mutual_info_score(yte, km.labels_)),
          "kmeans_ari": float(adjusted_rand_score(yte, km.labels_))}
    (C.RESULTS / "clusters").mkdir(exist_ok=True)
    (C.RESULTS / "clusters" / f"{tag}.json").write_text(json.dumps(cl, indent=2))
    lin = [r for t, r in zip(tasks, res) if r[0] == "linear"]
    print(f"[probe] {tag}: linear acc by frac "
          + ", ".join(f"{t[0]}:{r[2]:.3f}" for t, r in zip(
              [t for t in tasks if t[2] == 'linear'], lin))
          + f" | NMI {cl['kmeans_nmi']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run"])
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.tag or "B_S_s0_d100")
    elif a.all:
        for p in sorted(C.FEATS.glob("*.npz")):
            run(p.stem, a.force)
    else:
        run(a.tag, a.force)
