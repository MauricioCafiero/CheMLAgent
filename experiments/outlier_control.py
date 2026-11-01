"""Control for poison_pockets.py: is dropping the k-means pocket better than
dropping the same number of training molecules by generic outlier detection
(features + target) or at random?

Same global scaffold split (seed 132) and RF as poison_pockets.py. Outlier
detectors are fit on the TRAIN split only, on median-imputed, standardized
RDKit 2D descriptors with the standardized target appended as one more column.
Each removal drops the same number of train molecules as the pocket did.

Scored on:  rest = global test minus the pocket (same set the pocket result used)
            full = whole global test (outlier/random removals only)

Usage: .venv/bin/python experiments/outlier_control.py [--pocket CSV] [--rf-seeds N]
"""
import argparse

import torch  # noqa: F401  (import before lightgbm/skfp: libomp double-load)
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from skfp.preprocessing import MolFromSmilesTransformer

from chemlagent.fingerprints import (RDKit2DDescriptors, clean_smiles,
                                     scaffold_train_test_split)
from chemlagent.descriptor_cleaning import impute_only_cleaner
from chemlagent.models import random_forest_regression

ap = argparse.ArgumentParser()
ap.add_argument("--pocket", default="experiments/results/chembl220_pockets_rmse_k2/pocket_levels.csv")
ap.add_argument("--rf-seeds", type=int, default=3)
args = ap.parse_args()

df = pd.read_csv(args.pocket)
smiles = clean_smiles(df["SMILES"].tolist())
y = df["target"].to_numpy(float)
in_pocket = df["pocket_level"].to_numpy() >= 1
X = RDKit2DDescriptors().transform(MolFromSmilesTransformer().transform(smiles))
X[~np.isfinite(X)] = np.nan
pos = {s: i for i, s in enumerate(smiles)}
tr, te, _, _ = scaffold_train_test_split(smiles, y, test_size=0.2, random_state=132)
g_tr = np.array([pos[s] for s in tr])
g_te = np.array([pos[s] for s in te])
rest = g_te[~in_pocket[g_te]]
n_drop = int(in_pocket[g_tr].sum())
print(f"train {len(g_tr)}, test {len(g_te)} (rest {len(rest)}); "
      f"pocket removes {n_drop} train mols", flush=True)

# Outlier scores on train: imputed + standardized descriptors, plus target.
Xc = impute_only_cleaner().fit_transform(X[g_tr])
Z = np.hstack([StandardScaler().fit_transform(Xc),
               StandardScaler().fit_transform(y[g_tr, None])])
iso = -IsolationForest(n_estimators=500, random_state=0).fit(Z).score_samples(Z)
lof = -LocalOutlierFactor(n_neighbors=20).fit(Z).negative_outlier_factor_
rng = np.random.default_rng(0)

removals = {
    "none (full train)": np.zeros(len(g_tr), bool),
    "k-means pocket": in_pocket[g_tr],
    "IsolationForest": np.isin(np.arange(len(g_tr)), np.argsort(iso)[-n_drop:]),
    "LOF": np.isin(np.arange(len(g_tr)), np.argsort(lof)[-n_drop:]),
}
for s in range(5):
    removals[f"random #{s}"] = np.isin(np.arange(len(g_tr)),
                                       rng.choice(len(g_tr), n_drop, replace=False))


def score(idx, pred):
    return r2_score(y[idx], pred), mean_squared_error(y[idx], pred) ** 0.5


rows = []
for name, drop in removals.items():
    keep = g_tr[~drop]
    r = []
    for seed in range(args.rf_seeds):
        _, _, rf = random_forest_regression(X[keep], y[keep], X[g_te], y[g_te],
                                            n_estimators=200, n_jobs=-1,
                                            random_state=seed)
        p = dict(zip(g_te, rf.predict(X[g_te])))
        r.append(score(rest, [p[i] for i in rest]) + score(g_te, [p[i] for i in g_te]))
    r = np.array(r)
    rows.append(dict(removal=name, n_removed=int(drop.sum()),
                     pocket_overlap=int((drop & in_pocket[g_tr]).sum()),
                     dropped_y_mean=float(y[g_tr][drop].mean()) if drop.any() else np.nan,
                     rest_r2=r[:, 0].mean(), rest_r2_sd=r[:, 0].std(),
                     rest_rmse=r[:, 1].mean(), full_r2=r[:, 2].mean(),
                     full_rmse=r[:, 3].mean()))
    print(f"{name:20s} done", flush=True)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n" + out.round(3).to_string(index=False))
out.to_csv("experiments/results/chembl220_outlier_control.csv", index=False)
print("\nWrote experiments/results/chembl220_outlier_control.csv")
