"""Cluster quality by featurizer -- a Table-1-style comparison (Toney et al.,
Chem. Sci. 2026) using chemlagent.clustering (the cluster_molecules tool core).

For each dataset x fp_type, featurize with the agent's own get_fingerprints
(train + test stacked; the split is irrelevant here), cluster with k-means
(k auto by silhouette, 2..20) and HDBSCAN (auto), and score the clusters in
target space: fraction of cluster pairs with significantly different target
distributions (Mann-Whitney U p < 0.05), Davies-Bouldin (lower better),
Calinski-Harabasz (higher better). A Murcko-scaffold grouping (the paper's
"Scaffold" baseline; scaffolds with < 5 members treated as noise) is scored
the same way.

Features are cached in experiments/results/cluster_cache/ (3D conformer
generation on CHEMBL220 takes ~12 min); results go to
experiments/results/cluster_compare_<dataset>.csv.

Usage: .venv/bin/python experiments/cluster_compare.py [--features ...] [--datasets ...]
"""
import argparse
import os
import time

import torch  # noqa: F401  (import before lightgbm/skfp: libomp double-load)
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemlagent.clustering import cluster_features, cluster_quality
from chemlagent.fingerprints import clean_smiles, get_fingerprints

DATASETS = {
    "CHEMBL220": ("recent_work/CHEMBL220/data.csv", "target"),
    "azo": ("621-azo.csv", "Lmax"),
}
CACHE = "experiments/results/cluster_cache"
OUT = "experiments/results/cluster_compare_{}.csv"  # one file per dataset

ap = argparse.ArgumentParser()
ap.add_argument("--features", nargs="+",
                default=["ECFP", "RDKit_2D", "Spectral_2D", "Spectral"])
ap.add_argument("--datasets", nargs="+", default=["azo", "CHEMBL220"],
                choices=list(DATASETS))
args = ap.parse_args()
os.makedirs(CACHE, exist_ok=True)

rows = []
for dname in args.datasets:
    path, ycol = DATASETS[dname]
    df = pd.read_csv(path).dropna(subset=["SMILES", ycol])
    smiles, y_all = df["SMILES"].tolist(), df[ycol].to_numpy(float)

    # Scaffold baseline: group by Murcko scaffold of the cleaned SMILES.
    scaf = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) for s in clean_smiles(smiles)]
    codes, inv, counts = np.unique(scaf, return_inverse=True, return_counts=True)
    labels = np.where(counts[inv] >= 5, inv, -1)
    q = cluster_quality(y_all, labels)
    rows.append(dict(dataset=dname, features="Murcko scaffold", method="scaffold",
                     n=len(y_all), n_clusters=len(set(labels[labels >= 0])),
                     n_noise=int((labels < 0).sum()), silhouette=np.nan,
                     u_test_frac=q["u_test_frac"], dbi=q["dbi"], chi=q["chi"]))
    print(rows[-1], flush=True)

    for fp in args.features:
        cache = os.path.join(CACHE, f"{dname}_{fp}.npz")
        if os.path.exists(cache):
            c = np.load(cache)
            X, y, t_feat = c["X"], c["y"], float(c["seconds"])
        else:
            t0 = time.time()
            gfp = get_fingerprints(smiles, y_all, transform_flag=False)
            Xtr, Xte, ytr, yte = gfp.create(fp_type=fp)
            X, y = np.vstack([Xtr, Xte]), np.concatenate([ytr, yte])
            t_feat = time.time() - t0
            np.savez(cache, X=X, y=y, seconds=t_feat)
        for method in ("kmeans", "hdbscan"):
            t0 = time.time()
            labels, info = cluster_features(X, method=method)
            q = cluster_quality(y, labels)
            rows.append(dict(dataset=dname, features=fp, method=method, n=len(y),
                             n_clusters=info["n_clusters"], n_noise=info["n_noise"],
                             silhouette=info["silhouette"],
                             u_test_frac=q["u_test_frac"], dbi=q["dbi"], chi=q["chi"],
                             featurize_seconds=t_feat, cluster_seconds=time.time() - t0))
            print(rows[-1], flush=True)
        pd.DataFrame([r for r in rows if r["dataset"] == dname]).to_csv(OUT.format(dname), index=False)

r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.drop(columns=["featurize_seconds", "cluster_seconds"], errors="ignore")
      .round(3).to_string(index=False))
