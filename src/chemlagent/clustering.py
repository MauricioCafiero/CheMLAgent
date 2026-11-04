"""Unsupervised clustering of featurized molecules + target-space cluster quality.

Shared by the ``cluster_molecules`` agent tool and experiments/cluster_compare.py.

Two methods, both of which pick the number of clusters themselves by default:
  * 'kmeans'  -- KMeans; with n_clusters=0 it sweeps k = 2..max_clusters and
                 keeps the elbow of the inertia curve (the k farthest below
                 the chord joining the first and last points). Silhouette was
                 tried first but picks k=2 on most molecular datasets (4 of 6
                 tested), where the elbow gave 6-11; silhouette per k is still
                 reported in k_scan.
  * 'hdbscan' -- sklearn HDBSCAN (density-based); finds its own number of
                 clusters and labels low-density molecules as noise (-1). Run
                 on the first 10 PCA components with min_samples=5: on 50
                 components it labelled everything noise or fragmented into
                 100+ clusters (density estimates degrade with dimension).

Features are prepared the same way dimension_reduction does: non-finite -> 0,
StandardScaler, then PCA to <= 50 components (k-means) or <= 10 (HDBSCAN)
when wider (fingerprints and Mordred have 1000+ sparse/collinear columns,
where distances behave poorly and run slowly).

Quality is scored in TARGET space, following Toney et al. (Chem. Sci. 2026,
Table 1): the fraction of cluster pairs whose target distributions differ
significantly (two-sided Mann-Whitney U, p < 0.05), plus the Davies-Bouldin
(lower = better) and Calinski-Harabasz (higher = better) indices computed on
the target values. Noise points (HDBSCAN -1) are left out of the scores.
"""
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

METHODS = ("kmeans", "hdbscan")
MAX_PCA = 50        # k-means feature space
HDBSCAN_PCA = 10    # HDBSCAN feature space
HDBSCAN_MIN_SAMPLES = 5
SIL_SAMPLE = 2000   # silhouette on a random subsample: O(n^2) on 6k+ molecules


def cluster_features(X, method="kmeans", n_clusters=0, max_clusters=20,
                     min_cluster_size=0, random_state=42):
  """Cluster a feature matrix. Returns (labels, info).

  n_clusters=0 (kmeans) -> choose k at the inertia elbow over 2..max_clusters.
  min_cluster_size=0 (hdbscan) -> max(10, min(25, n // 100)).
  info: method, n_clusters, n_noise, silhouette (feature space, noise
  excluded; NaN if < 2 clusters), k_scan ({k: {inertia, silhouette}}, kmeans
  auto only), min_cluster_size (hdbscan), n_pca_components (0 = no PCA).
  """
  if method not in METHODS:
    raise ValueError(f"Unknown method {method!r}. Use 'kmeans' or 'hdbscan'.")
  X = np.asarray(X, dtype=np.float64)
  X = StandardScaler().fit_transform(np.where(np.isfinite(X), X, 0.0))
  n, d = X.shape
  n_pca = 0
  max_pca = MAX_PCA if method == "kmeans" else HDBSCAN_PCA
  if d > max_pca:
    n_pca = max(2, min(max_pca, n - 1, d - 1))
    X = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)

  def sil(labels):
    keep = labels >= 0
    if len(set(labels[keep])) < 2:
      return float("nan")
    return float(silhouette_score(X[keep], labels[keep],
                                  sample_size=min(int(keep.sum()), SIL_SAMPLE),
                                  random_state=random_state))

  info = {"method": method, "n_pca_components": n_pca}
  if method == "kmeans":
    if n_clusters:
      labels = KMeans(n_clusters=n_clusters, n_init=10,
                      random_state=random_state).fit_predict(X)
    else:
      ks = list(range(2, max(3, min(max_clusters, n - 1)) + 1))
      fits = [KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
              for k in ks]
      inertia = np.array([f.inertia_ for f in fits])
      # elbow: normalise k and inertia to [0, 1]; pick the point farthest
      # below the straight line from (first k, max inertia) to (last k, min).
      xs = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
      ys = (inertia - inertia.min()) / max(inertia.max() - inertia.min(), 1e-12)
      best = int(np.argmax((1 - xs) - ys))
      labels = fits[best].labels_
      info["k_scan"] = {k: {"inertia": round(float(i), 2),
                            "silhouette": round(sil(f.labels_), 4)}
                        for k, i, f in zip(ks, inertia, fits)}
  else:
    mcs = min_cluster_size or max(10, min(25, n // 100))
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=HDBSCAN_MIN_SAMPLES,
                     copy=True).fit_predict(X)
    info["min_cluster_size"] = int(mcs)

  info["n_clusters"] = int(len(set(labels[labels >= 0])))
  info["n_noise"] = int((labels < 0).sum())
  info["silhouette"] = sil(labels)
  return labels, info


def cluster_quality(y, labels, alpha=0.05):
  """Target-space quality of a clustering (noise label -1 excluded).

  Returns: u_test_frac (fraction of cluster pairs with Mann-Whitney p < alpha),
  dbi, chi (on the target values; NaN if < 2 clusters), and per-cluster
  summaries sorted by size: [{cluster, size, target_mean, target_std}].
  """
  y = np.asarray(y, dtype=float)
  labels = np.asarray(labels)
  keep = labels >= 0
  y, labels = y[keep], labels[keep]
  ids = sorted(set(labels))
  groups = [y[labels == c] for c in ids]
  summary = sorted(
      ({"cluster": int(c), "size": int(len(g)), "target_mean": float(g.mean()),
        "target_std": float(g.std())} for c, g in zip(ids, groups)),
      key=lambda r: -r["size"])
  out = {"u_test_frac": float("nan"), "dbi": float("nan"),
         "chi": float("nan"), "clusters": summary}
  if len(ids) < 2:
    return out
  pvals = [mannwhitneyu(groups[a], groups[b]).pvalue
           for a in range(len(ids)) for b in range(a + 1, len(ids))]
  out["u_test_frac"] = float(np.mean(np.array(pvals) < alpha))
  out["dbi"] = float(davies_bouldin_score(y[:, None], labels))
  out["chi"] = float(calinski_harabasz_score(y[:, None], labels))
  return out
