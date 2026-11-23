"""Tables and figures from results/ and features/.

    python -m src.report [--proj umap|tsne] [--proj-size S]

Outputs
  results/summary.csv          mean/std over seeds x draws per (row, size, data, label frac, probe)
  results/main_table.md        linear / RBF / kNN accuracy + NMI/ARI + params/latency
  results/gap.csv              paired C-B gap (same seed, same labeled draw) per cell
  results/collapse_checks.csv  final collapse check of every B/C run
  figures/label_efficiency.png figures/gap_vs_params.png figures/ablation_gap.png
  figures/projection_<S>.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

CLASSES = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
           "normal mucosa", "cancer stroma", "adenocarcinoma"]
# categorical slots (validated palette), fixed order; ink colors for text
ROW_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
ROW_MARK = {"A": "o", "B": "s", "C": "^", "D": "D"}
ROW_NAME = {"A": "A random", "B": "B direct (VICReg)", "C": "C distilled", "D": "D teacher"}
FRAC_RAMP = {0.01: "#0d366b", 0.10: "#1c5cab", 0.50: "#2a78d6", 1.00: "#6da7ec"}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                "#6250d6", "#e34948", "#8a8a85"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3de"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def load_probes():
    files = sorted((C.RESULTS / "probes").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["size"] = df["size"].fillna("-")
    return df


def summarize(df):
    g = df.groupby(["row", "size", "data_frac", "label_frac", "probe"])["acc"]
    s = g.agg(["mean", "std", "count"]).reset_index()
    s.to_csv(C.RESULTS / "summary.csv", index=False)
    return s


def paired_gap(df):
    """C - B for the same size, seed, data fraction, label fraction, draw and probe."""
    key = ["size", "seed", "data_frac", "label_frac", "draw", "probe"]
    b = df[df.row == "B"].set_index(key)["acc"]
    c = df[df.row == "C"].set_index(key)["acc"]
    gap = (c - b).dropna().rename("gap").reset_index()
    out = gap.groupby(["size", "data_frac", "label_frac", "probe"])["gap"].agg(
        ["mean", "std", "count"]).reset_index()
    out.to_csv(C.RESULTS / "gap.csv", index=False)
    return out


def clusters():
    rows = [json.loads(p.read_text()) for p in sorted((C.RESULTS / "clusters").glob("*.json"))]
    d = pd.DataFrame(rows)
    d["size"] = d["size"].fillna("-")
    g = d.groupby(["row", "size", "data_frac"])[["kmeans_nmi", "kmeans_ari"]].agg(
        ["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index().round(3)


def main_table(s, cl):
    cost = pd.read_csv(C.RESULTS / "cost_models.csv") if (C.RESULTS / "cost_models.csv").exists() else None
    lines = []
    for probe in ("linear", "rbf", "knn"):
        t = s[(s.probe == probe) & (s.data_frac == 1.0)].copy()
        t["v"] = t.apply(lambda r: f"{100 * r['mean']:.1f} ± {100 * (r['std'] if r['count'] > 1 else 0):.1f}", axis=1)
        p = t.pivot_table(index=["row", "size"], columns="label_frac", values="v", aggfunc="first")
        p.columns = [f"{int(c * 100)}%" for c in p.columns]
        p = p.reset_index()
        lines.append(f"\n### {probe} probe accuracy (%), mean ± std over seeds × draws\n")
        lines.append(p.to_markdown(index=False))
    c1 = cl[cl.data_frac == 1.0]
    lines.append("\n### k-means on test features (k = 9)\n")
    lines.append(c1.to_markdown(index=False))
    if cost is not None:
        lines.append("\n### Cost\n")
        lines.append(cost.round(3).to_markdown(index=False))
    (C.RESULTS / "main_table.md").write_text("\n".join(lines))
    print("\n".join(lines))


def fig_label_efficiency(s):
    sizes = [z for z in C.SIZES if z in set(s["size"])]
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4), sharey=True,
                             squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for row in "ABCD":
            t = s[(s.probe == "linear") & (s.row == row) & (s.data_frac == 1.0)
                  & ((s["size"] == size) if row != "D" else True)].sort_values("label_frac")
            if t.empty:
                continue
            ax.errorbar(t.label_frac, 100 * t["mean"], yerr=100 * t["std"].fillna(0),
                        color=ROW_COLOR[row], marker=ROW_MARK[row], ms=6, lw=2,
                        capsize=3, label=ROW_NAME[row],
                        ls="--" if row == "D" else "-")
        ax.set_xscale("log")
        ax.set_xticks(C.LABEL_FRACS, [f"{int(f * 100)}%" for f in C.LABEL_FRACS])
        ax.set_title(f"student {size}", color=INK)
        ax.set_xlabel("labeled fraction of the training split", color=INK)
        style(ax)
    axes[0][0].set_ylabel("linear SVM test accuracy (%)", color=INK)
    axes[0][-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.FIGS / "label_efficiency.png", dpi=200)
    plt.close(fig)


def fig_gap_vs_params(gap):
    cost = C.RESULTS / "cost_models.csv"
    if not cost.exists():
        print("[report] run `python -m src.cost` first for gap_vs_params"); return
    cm = pd.read_csv(cost).set_index("encoder")["params_M"]
    g = gap[(gap.probe == "linear") & (gap.data_frac == 1.0)].copy()
    g["params"] = g["size"].map(lambda z: cm.get(f"student_{z}", np.nan))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.axhline(0, color=MUTED, lw=1)
    for frac, col in FRAC_RAMP.items():
        t = g[g.label_frac == frac].sort_values("params")
        if t.empty:
            continue
        ax.errorbar(t.params, 100 * t["mean"], yerr=100 * t["std"].fillna(0), color=col,
                    marker="o", ms=6, lw=2, capsize=3, label=f"{int(frac * 100)}% labels")
        ax.annotate(f"{int(frac * 100)}%", (t.params.iloc[-1], 100 * t["mean"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK,
                    fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("student parameters (M)", color=INK)
    ax.set_ylabel("C − B, linear SVM accuracy (pp)", color=INK)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(C.FIGS / "gap_vs_params.png", dpi=200)
    plt.close(fig)


def fig_ablation(gap):
    g = gap[gap.probe == "linear"]
    if g.data_frac.nunique() < 2:
        return
    fracs = [f for f in C.LABEL_FRACS if f in set(g.label_frac)]
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.2 * len(fracs), 3.4), sharey=True,
                             squeeze=False)
    cols = {"S": "#2a78d6", "L": "#eb6834"}
    for ax, fr in zip(axes[0], fracs):
        ax.axhline(0, color=MUTED, lw=1)
        for size in C.SIZES:
            t = g[(g.label_frac == fr) & (g["size"] == size)].sort_values("data_frac")
            if t.empty:
                continue
            ax.errorbar(100 * t.data_frac, 100 * t["mean"], yerr=100 * t["std"].fillna(0),
                        color=cols[size], marker="o" if size == "S" else "s", lw=2,
                        capsize=3, label=f"student {size}")
        ax.set_xticks([50, 100], ["50%", "100%"])
        ax.set_title(f"{int(fr * 100)}% labels", color=INK)
        ax.set_xlabel("unlabeled data used", color=INK)
        style(ax)
    axes[0][0].set_ylabel("C − B (pp)", color=INK)
    axes[0][-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGS / "ablation_gap.png", dpi=200)
    plt.close(fig)


def fig_projection(size="S", seed=0, method="umap"):
    from sklearn.preprocessing import StandardScaler
    from .diag import l2n
    tags = [("A", C.tag("A", size, seed)), ("B", C.tag("B", size, seed)),
            ("C", C.tag("C", size, seed)), ("D", "D")]
    tags = [(r, t) for r, t in tags if (C.FEATS / f"{t}.npz").exists()]
    fig, axes = plt.subplots(1, len(tags), figsize=(4 * len(tags), 4.3), squeeze=False)
    for ax, (row, t) in zip(axes[0], tags):
        z = np.load(C.FEATS / f"{t}.npz")
        sc = StandardScaler().fit(l2n(z["train"]))
        x, y = sc.transform(l2n(z["test"])), z["y_test"]
        if method == "umap":
            import umap
            e = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(x)
        else:
            from sklearn.manifold import TSNE
            e = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(x)
        for k in range(C.NUM_CLASSES):
            m = y == k
            ax.scatter(e[m, 0], e[m, 1], s=3, color=CLASS_COLORS[k], label=CLASSES[k],
                       linewidths=0, alpha=0.8)
        ax.set_title(f"{ROW_NAME[row]}  ({t})", color=INK, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, markerscale=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(C.FIGS / f"projection_{size}.png", dpi=200)
    plt.close(fig)


def collapse_table():
    rows = []
    for r in sorted(C.RUNS.iterdir()):
        mp = r / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            rows.append({"run": m["name"], **m.get("final_collapse_check", {}),
                         "train_gpu_hours": m.get("train_gpu_hours")})
    if rows:
        pd.DataFrame(rows).to_csv(C.RESULTS / "collapse_checks.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--proj-size", default="S")
    a = ap.parse_args()
    df = load_probes()
    s = summarize(df)
    gap = paired_gap(df)
    main_table(s, clusters())
    collapse_table()
    fig_label_efficiency(s)
    fig_gap_vs_params(gap)
    fig_ablation(gap)
    fig_projection(a.proj_size, 0, a.proj)
    print(f"[report] figures in {C.FIGS}")


if __name__ == "__main__":
    main()
