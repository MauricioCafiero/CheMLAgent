"""Hunt for pockets of ill-behaved molecules that poison RF training.

Features for the RF:  RDKit 2D descriptors (RDKit2DDescriptors).
Features for k-means: ECFP fingerprints. Using two different featurizations
keeps the clustering from leaking the regression feature space.

Procedure:
  level 0: RF on the full set (baseline).
  each level: k-means(k=2) on ECFP of the current pocket, RF on each half,
  descend into the half with the lower test R2; stop when a half is < MIN_SIZE.
Each RF score is the mean over SEEDS scaffold splits *within* that subset.

Poisoning test (per level): fix the global scaffold split, retrain RF on the
global train set with the pocket removed, and score on the global test
molecules outside the pocket -- compared with the full-train model on the same
test molecules. Improvement => the pocket is hurting the rest of the model.

--metric picks the worse cluster by lower test R2 or higher test RMSE (R2 is
confounded by a cluster's target spread). --k sets the k-means cluster count per
level; clusters < MIN_SIZE are skipped (not scored), and the run stops when
fewer than two clusters are scorable.

Usage: .venv/bin/python experiments/poison_pockets.py [--data CSV] [--out DIR]
                                                      [--metric r2|rmse] [--k K]
"""
import argparse
import json
import os

import torch  # noqa: F401  (import before lightgbm/skfp: libomp double-load)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from skfp.fingerprints import ECFPFingerprint
from skfp.preprocessing import MolFromSmilesTransformer

from chemlagent.fingerprints import (RDKit2DDescriptors, clean_smiles,
                                     scaffold_train_test_split)
from chemlagent.models import random_forest_regression

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="recent_work/CHEMBL220/data.csv")
ap.add_argument("--out", default="experiments/results/chembl220_pockets")
ap.add_argument("--metric", choices=["r2", "rmse"], default="r2")
ap.add_argument("--k", type=int, default=2)
args = ap.parse_args()
DATA, OUT = args.data, args.out
MIN_SIZE = 150
SEEDS = (132, 7, 2024)
N_EST = 200

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(DATA)
smiles = clean_smiles(df["SMILES"].tolist())
y = df["target"].to_numpy(float)
mols = MolFromSmilesTransformer().transform(smiles)
print(f"{len(smiles)} molecules; featurizing...", flush=True)
X = RDKit2DDescriptors().transform(mols)
X[~np.isfinite(X)] = np.nan          # sklearn RF handles NaN natively
FP = ECFPFingerprint(fp_size=2048, radius=2, n_jobs=-1).transform(mols).astype(np.float32)
pos = {s: i for i, s in enumerate(smiles)}


def split_idx(idx, seed):
    tr, te, _, _ = scaffold_train_test_split([smiles[i] for i in idx], y[idx],
                                             test_size=0.2, random_state=seed)
    return np.array([pos[s] for s in tr]), np.array([pos[s] for s in te])


def rf_score(idx):
    r2s, rmses = [], []
    for seed in SEEDS:
        tr, te = split_idx(idx, seed)
        r2, _, rf = random_forest_regression(X[tr], y[tr], X[te], y[te],
                                             n_estimators=N_EST, n_jobs=-1)
        r2s.append(r2)
        rmses.append(mean_squared_error(y[te], rf.predict(X[te])) ** 0.5)
    return dict(n=len(idx), r2=float(np.mean(r2s)), r2_sd=float(np.std(r2s)),
                rmse=float(np.mean(rmses)), y_sd=float(np.std(y[idx])))


# Fixed global split + full-train model for the poisoning test.
g_tr, g_te = split_idx(np.arange(len(y)), SEEDS[0])
_, _, rf_full = random_forest_regression(X[g_tr], y[g_tr], X[g_te], y[g_te],
                                         n_estimators=N_EST, n_jobs=-1)
pred_full = rf_full.predict(X[g_te])


def poison_test(pocket):
    pocket = set(pocket.tolist())
    tr = np.array([i for i in g_tr if i not in pocket])
    keep = np.array([i not in pocket for i in g_te])
    te = g_te[keep]
    _, _, rf = random_forest_regression(X[tr], y[tr], X[te], y[te],
                                        n_estimators=N_EST, n_jobs=-1)
    p = rf.predict(X[te])
    return dict(n_train_removed=len(g_tr) - len(tr), n_test_rest=len(te),
                r2_rest_full=float(r2_score(y[te], pred_full[keep])),
                r2_rest_without=float(r2_score(y[te], p)),
                rmse_rest_full=float(mean_squared_error(y[te], pred_full[keep]) ** 0.5),
                rmse_rest_without=float(mean_squared_error(y[te], p) ** 0.5))


def fmt(s):
    return (f"n={s['n']:5d}  R2={s['r2']:.3f}±{s['r2_sd']:.3f}  "
            f"RMSE={s['rmse']:.3f}  y_sd={s['y_sd']:.3f}")


node = np.arange(len(y))
base = rf_score(node)
print(f"\nLEVEL 0 baseline: {fmt(base)}", flush=True)
history = [dict(level=0, node=base)]
labels = np.full(len(y), -1)          # deepest level at which a mol was in the bad pocket
labels[node] = 0
level = 0
while True:
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(FP[node])
    parts = [node[km.labels_ == k] for k in range(args.k)]
    ok = [k for k in range(args.k) if len(parts[k]) >= MIN_SIZE]
    if len(ok) < 2:
        print(f"\nStop: fewer than two clusters with >= {MIN_SIZE} molecules "
              f"({sorted(len(p) for p in parts)}).")
        break
    level += 1
    scores = {k: rf_score(parts[k]) for k in ok}
    pick = min if args.metric == "r2" else max
    worse = pick(ok, key=lambda k: scores[k][args.metric])
    # Poison-test every scored cluster when k > 2 (only the worse one for k=2).
    tested = ok if args.k > 2 else [worse]
    pts = {k: poison_test(parts[k]) for k in tested}
    print(f"\nLEVEL {level}  (worse by {args.metric}; skipped small clusters: "
          f"{[len(parts[k]) for k in range(args.k) if k not in ok]})")
    for k in ok:
        line = f"  cluster {k}{' (worse)' if k == worse else '        '}: {fmt(scores[k])}"
        if k in pts:
            pt = pts[k]
            line += (f"\n      poison test: rest R2 {pt['r2_rest_full']:.3f} -> "
                     f"{pt['r2_rest_without']:.3f}  RMSE {pt['rmse_rest_full']:.3f} -> "
                     f"{pt['rmse_rest_without']:.3f}  (-{pt['n_train_removed']} train, "
                     f"{pt['n_test_rest']} test)")
        print(line, flush=True)
    history.append(dict(level=level, clusters=scores, worse=worse, poison=pts))
    node = parts[worse]
    labels[node] = level

df.assign(pocket_level=labels).to_csv(os.path.join(OUT, "pocket_levels.csv"), index=False)
with open(os.path.join(OUT, "history.json"), "w") as fh:
    json.dump(history, fh, indent=2)
print(f"\nWrote {OUT}/pocket_levels.csv and history.json")
