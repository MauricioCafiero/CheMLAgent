"""Rigorous RF vs GP comparison, separating the model from the featurization.

2x2 design per split: {RF, GP} x {RDKit_2D descriptors, ECFP fingerprints},
using the agent's own featurizer (get_fingerprints.create) and model functions
(random_forest_regression, gaussian_process_regression with kernel='auto' ->
Tanimoto on ECFP, RBF on standardized RDKit_2D).

Datasets: CHEMBL220 (log10 IC50, ~6k mols) and 621-azo (Lmax in nm, 621 mols).
Splits:   SEEDS scaffold splits (80/20 by Murcko scaffold) per dataset.
Metrics:  test R2 / RMSE / MAE; for the GP also 95%-interval coverage and the
          Spearman correlation between predicted std and |error|.

Results append to experiments/results/gp_vs_rf.csv; finished
(dataset, seed, features, model) rows are skipped on re-run, so adding a new
featurizer only computes the new rows and reuses the existing baselines.

Usage: .venv/bin/python experiments/gp_vs_rf.py
       .venv/bin/python experiments/gp_vs_rf.py --features Spectral Spectral_2D
           [--models RF GP] [--datasets azo CHEMBL220]
"""
import argparse
import os
import time

import torch  # noqa: F401  (import before lightgbm/skfp: libomp double-load)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from chemlagent.fingerprints import get_fingerprints
from chemlagent.models import gaussian_process_regression, random_forest_regression

DATASETS = {
    "CHEMBL220": ("recent_work/CHEMBL220/data.csv", "target"),
    "azo": ("621-azo.csv", "Lmax"),
}
SEEDS = (132, 7, 2024, 11, 99)
OUT = "experiments/results/gp_vs_rf.csv"
KEY = ["dataset", "seed", "features", "model"]

ap = argparse.ArgumentParser()
ap.add_argument("--features", nargs="+", default=["RDKit_2D", "ECFP"],
                help="get_fingerprints fp_type values to compare")
ap.add_argument("--models", nargs="+", default=["RF", "GP"], choices=["RF", "GP"])
ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
args = ap.parse_args()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = (set(map(tuple, pd.read_csv(OUT)[KEY].values)) if os.path.exists(OUT) else set())

for dname in args.datasets:
    path, ycol = DATASETS[dname]
    df = pd.read_csv(path).dropna(subset=["SMILES", ycol])
    # One featurizer object per dataset: 3D conformers are generated once and
    # reused for every seed and every 3D fp_type (get_fingerprints caches them).
    gfp = get_fingerprints(df["SMILES"].tolist(), df[ycol].to_numpy(float),
                           transform_flag=False)
    for seed in SEEDS:
        rows = []
        for fp in args.features:
            todo = [m for m in args.models if (dname, seed, fp, m) not in done]
            if not todo:
                print(f"skip {dname} seed {seed} {fp} (done)")
                continue
            t_feat = time.time()
            Xtr, Xte, ytr, yte = gfp.create(fp_type=fp, split_seed=seed)
            Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
            Xtr[~np.isfinite(Xtr)] = np.nan
            Xte[~np.isfinite(Xte)] = np.nan
            t_feat = time.time() - t_feat
            for model in todo:
                t0 = time.time()
                if model == "RF":
                    _, _, m = random_forest_regression(Xtr, ytr, Xte, yte,
                                                       n_estimators=200, n_jobs=-1)
                    p, sd = m.predict(Xte), None
                else:
                    _, _, m = gaussian_process_regression(Xtr, ytr, Xte, yte)
                    p, sd = m.predict(Xte, return_std=True)
                err = np.abs(yte - p)
                rows.append(dict(
                    dataset=dname, seed=seed, features=fp, model=model,
                    n_train=len(ytr), n_test=len(yte),
                    r2=r2_score(yte, p), rmse=mean_squared_error(yte, p) ** 0.5,
                    mae=mean_absolute_error(yte, p),
                    cov95=np.mean(err <= 1.96 * sd) if sd is not None else np.nan,
                    std_err_spearman=spearmanr(sd, err)[0] if sd is not None else np.nan,
                    kernel=str(m[-1].kernel_) if model == "GP" else "",
                    seconds=time.time() - t0, featurize_seconds=t_feat))
                print(f"{dname} s{seed} {fp:8s} {model}: R2={rows[-1]['r2']:.3f} "
                      f"RMSE={rows[-1]['rmse']:.3f} ({rows[-1]['seconds']:.0f}s)", flush=True)
        if rows:
            new = pd.DataFrame(rows)
            if os.path.exists(OUT):  # align columns (featurize_seconds added later)
                new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
            new.to_csv(OUT, index=False)

r = pd.read_csv(OUT)
pd.set_option("display.width", 200)
print("\nMean test R2 over splits (features x model):")
print(r.pivot_table(index=["dataset", "features"], columns="model", values="r2",
                    aggfunc=["mean", "std"]).round(3).to_string())
print("\nMean (sd) over splits:")
print(r.groupby(["dataset", "features", "model"])[["r2", "rmse", "mae", "cov95",
                                                   "std_err_spearman"]]
      .agg(["mean", "std"]).round(3).to_string())
print("\nPaired per-split differences (GP - RF), R2:")
w = r.pivot_table(index=["dataset", "seed", "features"], columns="model", values="r2")
w["GP-RF"] = w["GP"] - w["RF"]
print(w.round(3).to_string())
print("\nPaired: ECFP GP minus best of the two RFs, R2:")
b = r.pivot_table(index=["dataset", "seed"], columns=["features", "model"], values="r2")
best_rf = b[[("RDKit_2D", "RF"), ("ECFP", "RF")]].max(axis=1)
print((b[("ECFP", "GP")] - best_rf).round(3).to_string())
