"""Feature extraction (deterministic eval transform) and unsupervised diagnostics
shared by the teacher sanity check and the collapse checks during training."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier

from . import config as C
from .common import DEVICE, autocast, resize_normalize, to_device_uint8


@torch.no_grad()
def encode(model, images, size, mean, std, idx=None, bs=256):
    """images: uint8 NHWC array (memmap ok). Always eval(): BatchNorm uses its running
    statistics (for row A these are the default mean 0 / var 1)."""
    model.eval()
    idx = np.arange(len(images)) if idx is None else np.asarray(idx)
    out = []
    for i in range(0, len(idx), bs):
        x = to_device_uint8(images[idx[i:i + bs]])
        with autocast():
            f = model(resize_normalize(x, size, mean, std))
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def l2n(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def diagnose(x_tr, y_tr, x_te, y_te, k=C.KNN_MAX_K, seed=0):
    """k-NN accuracy (cosine), per-dimension std of L2-normalized embeddings and
    k-means NMI/ARI. Collapse shows up as kNN ~ chance, std ~ 0, NMI ~ 0."""
    a, b = l2n(x_tr), l2n(x_te)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(a)), metric="cosine",
                               algorithm="brute").fit(a, y_tr)
    acc = float((knn.predict(b) == y_te).mean())
    std = float(b.std(0).mean())
    km = KMeans(n_clusters=C.NUM_CLASSES, n_init=10, random_state=seed).fit(b)
    return {"knn_acc": acc, "chance": float(np.bincount(y_te).max() / len(y_te)),
            "emb_std": std, "emb_std_x_sqrt_d": float(std * np.sqrt(b.shape[1])),
            "kmeans_nmi": float(normalized_mutual_info_score(y_te, km.labels_)),
            "kmeans_ari": float(adjusted_rand_score(y_te, km.labels_))}
