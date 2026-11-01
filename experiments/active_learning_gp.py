"""Active learning with a Tanimoto-kernel Gaussian process on CHEMBL220.

Pool = train side of the global scaffold split (seed 132, same split as
poison_pockets.py / outlier_control.py); scored on the held-out test side.

Start set:   MaxMin diversity pick (RDKit MaxMinPicker, Tanimoto) or random.
Acquisition: highest GP predictive std ("variance") or random (control).
Every (start, acquisition, seed) run starts from N_START molecules and adds
BATCH per round up to N_MAX, refitting the GP each round.

GP: ConstantKernel * Tanimoto + WhiteKernel on Morgan bits (r=2, 2048),
normalize_y=True; amplitude and noise are re-optimized every round.

Per run it writes experiments/results/chembl220_al/<name>_curve.csv (test R2/RMSE vs n) and
<name>_acq.csv (each acquired molecule: pre-acquisition prediction, std, true
value, z = |resid|/std, and k-means pocket membership). Existing runs are
skipped, so the script resumes after an interruption.

Usage: .venv/bin/python experiments/active_learning_gp.py [--seeds 3] [--n-max 1500]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import MaxMinPicker
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score

from chemlagent.fingerprints import clean_smiles, scaffold_train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="experiments/results/chembl220_pockets_rmse_k2/pocket_levels.csv")
ap.add_argument("--out", default="experiments/results/chembl220_al")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n-start", type=int, default=100)
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--n-max", type=int, default=1500)
args = ap.parse_args()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.makedirs(args.out, exist_ok=True)


class Tanimoto(Kernel):
    """Tanimoto (Jaccard) similarity kernel on binary/count fingerprints.
    Parameter-free; scale it with a ConstantKernel."""

    def __init__(self):  # sklearn requires an explicit (here empty) signature
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        Y = X if Y is None else Y
        xy = X @ Y.T
        K = xy / ((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - xy)
        K = K.astype(np.float64)
        return (K, np.empty((X.shape[0], X.shape[0], 0))) if eval_gradient else K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False


df = pd.read_csv(args.data)
smiles = clean_smiles(df["SMILES"].tolist())
y = df["target"].to_numpy(float)
in_pocket = (df["pocket_level"].to_numpy() >= 1) if "pocket_level" in df else np.zeros(len(y), bool)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
bvs = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
FP = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], np.float32)
pos = {s: i for i, s in enumerate(smiles)}
tr, te, _, _ = scaffold_train_test_split(smiles, y, test_size=0.2, random_state=132)
pool_all = np.array([pos[s] for s in tr])
test = np.array([pos[s] for s in te])
print(f"pool {len(pool_all)}, test {len(test)}", flush=True)


def fit_gp(idx):
    k = ConstantKernel(1.0, (1e-4, 1e2)) * Tanimoto() + WhiteKernel(0.3, (1e-5, 1e1))
    return GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(FP[idx], y[idx])


def start_set(kind, seed):
    if kind == "maxmin":
        picks = MaxMinPicker().LazyBitVectorPick([bvs[i] for i in pool_all], len(pool_all),
                                                 args.n_start, seed=seed)
        return pool_all[list(picks)]
    return np.random.default_rng(seed).choice(pool_all, args.n_start, replace=False)


for seed in range(args.seeds):
    for start in ("maxmin", "random"):
        for acq in ("variance", "random"):
            name = f"{start}-start_{acq}-acq_s{seed}"
            if os.path.exists(os.path.join(args.out, f"{name}_curve.csv")):
                print(f"skip {name} (done)")
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 + seed)
            train = start_set(start, seed)
            curve, acq_log, rnd = [], [], 0
            while True:
                gp = fit_gp(train)
                p = gp.predict(FP[test])
                curve.append(dict(n=len(train), r2=r2_score(y[test], p),
                                  rmse=mean_squared_error(y[test], p) ** 0.5,
                                  pocket_frac=in_pocket[train].mean(),
                                  kernel=str(gp.kernel_)))
                if len(train) >= args.n_max:
                    break
                pool = np.setdiff1d(pool_all, train)
                mu, sd = gp.predict(FP[pool], return_std=True)
                pick = (np.argsort(sd)[-args.batch:] if acq == "variance"
                        else rng.choice(len(pool), args.batch, replace=False))
                rnd += 1
                for j in pick:
                    i = pool[j]
                    acq_log.append(dict(round=rnd, idx=int(i), smiles=smiles[i], y=y[i],
                                        pred=mu[j], std=sd[j], z=abs(y[i] - mu[j]) / sd[j],
                                        in_pocket=bool(in_pocket[i])))
                train = np.concatenate([train, pool[pick]])
            pd.DataFrame(curve).to_csv(os.path.join(args.out, f"{name}_curve.csv"), index=False)
            pd.DataFrame(acq_log).to_csv(os.path.join(args.out, f"{name}_acq.csv"), index=False)
            print(f"{name}: final n={curve[-1]['n']} R2={curve[-1]['r2']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

# Reference: GP on the whole pool.
ref = os.path.join(args.out, "full_pool_gp.csv")
if not os.path.exists(ref):
    gp = fit_gp(pool_all)
    p = gp.predict(FP[test])
    pd.DataFrame([dict(n=len(pool_all), r2=r2_score(y[test], p),
                       rmse=mean_squared_error(y[test], p) ** 0.5,
                       kernel=str(gp.kernel_))]).to_csv(ref, index=False)
print(pd.read_csv(ref).to_string(index=False))
