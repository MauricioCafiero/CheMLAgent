"""Does the spectral fingerprint add a perspective the existing featurizers lack?

Four tests, all on the same molecules per dataset (those that embed in 3D):
  1. Redundancy     -- can existing featurizers reconstruct it? 3-fold CV R2 of a
                       random forest predicting the target featurizer's (standardized)
                       columns from a source featurizer. Low R2 = new information.
                       Spectral is compared with other 3D types as calibration.
  2. Neighbourhoods -- overlap (Jaccard) of each molecule's 10 nearest neighbours
                       between every pair of featurizers. Low = groups molecules
                       differently.
  3. Complementarity (5 scaffold splits, RF):
       a. concatenation: base (ECFP / RDKit_2D) vs base + Spectral(_2D), with
          base + MORSE / RDF / another 2D type as calibration;
       b. residuals: can spectral features predict a base model's test errors
          (base trained on train; residual model on out-of-fold train residuals);
       c. error correlation between the base model and the spectral model.
  4. Unique capability -- isomer pairs (E/Z, cis/trans, R/S, positional): the
     distance between the pair, relative to the median nearest-neighbour
     distance in azo (0 = cannot tell the pair apart).

Features are built with the agent's own get_fingerprints (conformers once per
dataset, shared by all 3D types) and cached in
experiments/results/complementarity_cache/. Results:
experiments/results/complementarity_*.csv.

Usage: .venv/bin/python experiments/spectral_complementarity.py [--n 300] [--datasets ...]
"""
import argparse
import os
import time

import torch  # noqa: F401  (import before lightgbm/skfp: libomp double-load)
import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from chemlagent.fingerprints import (SmilesFeaturizer, get_fingerprints,
                                     scaffold_train_test_split)

RDLogger.DisableLog("rdApp.*")
DATASETS = {
    "azo": ("621-azo.csv", "Lmax"),
    "CHEMBL220": ("recent_work/CHEMBL220/data.csv", "target"),
}
FP_2D = ["ECFP", "MACCS", "RDKit_2D", "Mordred", "Spectral_2D"]
FP_3D = ["Spectral", "MORSE", "RDF", "Autocorr", "E3FP"]
BINARY = {"ECFP", "MACCS", "E3FP"}  # neighbours by Jaccard; others standardized Euclidean
SEEDS = (132, 7, 2024, 11, 99)
ISOMERS = [
    ("E/Z azobenzene", "c1ccc(cc1)/N=N/c1ccccc1", "c1ccc(cc1)/N=N\\c1ccccc1"),
    ("E/Z stilbene", "c1ccc(cc1)/C=C/c1ccccc1", "c1ccc(cc1)/C=C\\c1ccccc1"),
    ("cis/trans 1,2-dimethylcyclohexane", "C[C@H]1CCCC[C@@H]1C", "C[C@H]1CCCC[C@H]1C"),
    ("R/S ibuprofen", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"),
    ("o/p cresol", "Cc1ccccc1O", "Cc1ccc(O)cc1"),
]
RES = "experiments/results"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="random subset size per dataset (0 = all)")
ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
args = ap.parse_args()
tag = f"_n{args.n}" if args.n else ""
CACHE = os.path.join(RES, f"complementarity_cache{tag}")
os.makedirs(CACHE, exist_ok=True)


def clean(X):
    X = np.asarray(X, dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(med), 0.0, med), X)
    return X[:, X.std(axis=0) > 0]  # drop constant columns


def featurize(dname):
    """Return (feats dict, y, mols) on the molecules that embed in 3D."""
    path, ycol = DATASETS[dname]
    df = pd.read_csv(path).dropna(subset=["SMILES", ycol])
    if args.n:
        df = df.sample(min(args.n, len(df)), random_state=0)
    f_npz = os.path.join(CACHE, f"{dname}.npz")
    gfp = get_fingerprints(df["SMILES"].tolist(), df[ycol].to_numpy(float), transform_flag=False)
    gfp.many_conf, gfp.num_confs = True, 5
    t0 = time.time()
    mols3d, y = gfp._embed_training_mols(np.asarray(gfp.y))
    ok = gfp._conformer_cache[(True, 5)][1]
    mols2d = [m for m, k in zip(gfp.mols_raw, ok) if k]
    print(f"{dname}: {len(y)}/{len(ok)} molecules embedded in {time.time()-t0:.0f}s", flush=True)
    feats = dict(np.load(f_npz)) if os.path.exists(f_npz) else {}
    for fp in FP_2D + FP_3D:
        if fp in feats:
            continue
        t0 = time.time()
        feats[fp] = clean(gfp.fp_hash[fp][0].transform(mols3d if fp in FP_3D else mols2d))
        print(f"  {fp:12s} {feats[fp].shape[1]:5d} cols  {time.time()-t0:.0f}s", flush=True)
        np.savez(f_npz, **feats)
    return feats, y, mols2d


def rf(n=200):
    return RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=0)


def redundancy(feats):
    """3-fold CV R2 (variance-weighted over target columns) predicting target from source."""
    rows = []
    kf = KFold(3, shuffle=True, random_state=0)
    for tgt in ["Spectral", "Spectral_2D", "MORSE", "RDF"]:
        T = StandardScaler().fit_transform(feats[tgt])
        for src in ["ECFP", "MACCS", "RDKit_2D", "Mordred"] + (["Spectral_2D"] if tgt == "Spectral" else []):
            P = cross_val_predict(RandomForestRegressor(100, max_features="sqrt", n_jobs=-1,
                                                        random_state=0), feats[src], T, cv=kf)
            rows.append(dict(target=tgt, source=src,
                             r2=r2_score(T, P, multioutput="variance_weighted")))
    return rows


def neighbours(feats, k=10):
    nn = {}
    for fp, X in feats.items():
        if fp in BINARY:
            m = NearestNeighbors(n_neighbors=k + 1, metric="jaccard").fit(X > 0)
            idx = m.kneighbors(X > 0, return_distance=False)
        else:
            Z = StandardScaler().fit_transform(X)
            idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)
        nn[fp] = [set(r[1:]) for r in idx]
    rows, names = [], list(nn)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = np.mean([len(x & y) / len(x | y) for x, y in zip(nn[a], nn[b])])
            rows.append(dict(a=a, b=b, jaccard=j))
    return rows


def complementarity(feats, y, mols):
    rows = []
    ids = {id(m): i for i, m in enumerate(mols)}
    for seed in SEEDS:
        mtr, mte, _, _ = scaffold_train_test_split(list(mols), y, test_size=0.2, random_state=seed)
        tr, te = np.array([ids[id(m)] for m in mtr]), np.array([ids[id(m)] for m in mte])
        pred = {}

        def fit(name, X):
            pred[name] = rf().fit(X[tr], y[tr]).predict(X[te])
            rows.append(dict(seed=seed, test="concat", model=name, r2=r2_score(y[te], pred[name])))

        for fp in ["ECFP", "RDKit_2D", "Spectral", "Spectral_2D", "MORSE"]:
            fit(fp, feats[fp])
        for base, other in [("ECFP", "Spectral"), ("ECFP", "Spectral_2D"), ("ECFP", "MORSE"),
                            ("ECFP", "RDF"), ("ECFP", "RDKit_2D"),
                            ("RDKit_2D", "Spectral"), ("RDKit_2D", "Spectral_2D"),
                            ("RDKit_2D", "MORSE"), ("RDKit_2D", "Mordred")]:
            fit(f"{base}+{other}", np.hstack([feats[base], feats[other]]))
        for base in ["ECFP", "RDKit_2D"]:
            oof = cross_val_predict(rf(100), feats[base][tr], y[tr],
                                    cv=KFold(3, shuffle=True, random_state=0))
            res_tr, res_te = y[tr] - oof, y[te] - pred[base]
            for extra in ["Spectral", "Spectral_2D", "MORSE"]:
                p = rf(100).fit(feats[extra][tr], res_tr).predict(feats[extra][te])
                rows.append(dict(seed=seed, test="residual", model=f"{extra}->{base} residuals",
                                 r2=r2_score(res_te, p), corr=np.corrcoef(res_te, p)[0, 1]))
                rows.append(dict(seed=seed, test="error_corr", model=f"{base} vs {extra}",
                                 corr=np.corrcoef(res_te, y[te] - pred[extra])[0, 1]))
    return rows


def isomers():
    """Featurize azo + the isomer molecules together (shared conformer run and
    column scaling); distance of each pair relative to the median azo
    nearest-neighbour distance in the same space."""
    path, ycol = DATASETS["azo"]
    azo = pd.read_csv(path).dropna(subset=["SMILES", ycol])["SMILES"].tolist()
    if args.n:
        azo = azo[:args.n]
    iso = [s for _, a, b in ISOMERS for s in (a, b)]
    g = get_fingerprints(azo + iso, list(range(len(azo) + len(iso))), transform_flag=False)
    g.many_conf, g.num_confs = True, 5
    m3, _ = g._embed_training_mols(np.asarray(g.y))
    ok = g._conformer_cache[(True, 5)][1]
    assert ok[len(azo):].all(), "an isomer failed to embed"
    m2 = [m for m, k in zip(g.mols_raw, ok) if k]
    n_iso = len(iso)
    rows = []
    for fp in FP_2D + FP_3D:
        X = np.asarray(g.fp_hash[fp][0].transform(m3 if fp in FP_3D else m2), dtype=float)
        if fp in BINARY:
            B = X > 0
            D = lambda u, v: 1 - (u & v).sum() / max((u | v).sum(), 1)
            nnd = NearestNeighbors(n_neighbors=2, metric="jaccard").fit(B[:-n_iso]).kneighbors(B[:-n_iso])[0][:, 1]
            P = B[-n_iso:]
        else:
            X = clean(X)
            Z = StandardScaler().fit(X[:-n_iso]).transform(X)
            D = lambda u, v: float(np.linalg.norm(u - v))
            nnd = NearestNeighbors(n_neighbors=2).fit(Z[:-n_iso]).kneighbors(Z[:-n_iso])[0][:, 1]
            P = Z[-n_iso:]
        scale = np.median(nnd)
        for k, (name, _, _) in enumerate(ISOMERS):
            d = D(P[2 * k], P[2 * k + 1])
            rows.append(dict(featurizer=fp, pair=name, distance=d, rel_to_median_nn=d / scale))
    return rows


all_rows = {"redundancy": [], "neighbours": [], "complementarity": [], "isomers": []}
for dname in args.datasets:
    t0 = time.time()
    feats, y, mols = featurize(dname)
    for r in redundancy(feats):
        all_rows["redundancy"].append(dict(dataset=dname, **r))
    print(f"[{dname}] redundancy done {time.time()-t0:.0f}s", flush=True)
    for r in neighbours(feats):
        all_rows["neighbours"].append(dict(dataset=dname, **r))
    print(f"[{dname}] neighbours done {time.time()-t0:.0f}s", flush=True)
    for r in complementarity(feats, y, mols):
        all_rows["complementarity"].append(dict(dataset=dname, **r))
    print(f"[{dname}] complementarity done {time.time()-t0:.0f}s", flush=True)
    for k, v in all_rows.items():
        if v:
            pd.DataFrame(v).to_csv(os.path.join(RES, f"complementarity_{k}{tag}.csv"), index=False)

all_rows["isomers"] = isomers()
pd.DataFrame(all_rows["isomers"]).to_csv(os.path.join(RES, f"complementarity_isomers{tag}.csv"), index=False)

pd.set_option("display.width", 220)
R = pd.DataFrame(all_rows["redundancy"])
print("\n1. Redundancy: CV R2 reconstructing target from source (low = new information)")
print(R.pivot_table(index=["dataset", "target"], columns="source", values="r2").round(2).to_string())
N = pd.DataFrame(all_rows["neighbours"])
print("\n2. Neighbour overlap (Jaccard of 10-NN sets) involving Spectral(_2D)")
print(N[N.a.str.startswith("Spectral") | N.b.str.startswith("Spectral")].pivot_table(
    index=["dataset", "a"], columns="b", values="jaccard").round(2).to_string())
C = pd.DataFrame(all_rows["complementarity"])
print("\n3a. Concatenation, mean test R2 over splits")
print(C[C.test == "concat"].pivot_table(index="model", columns="dataset", values="r2").round(3).to_string())
print("\n3b/c. Residual prediction (r2, corr) and error correlation")
print(C[C.test != "concat"].pivot_table(index=["test", "model"], columns="dataset",
      values=["r2", "corr"]).round(3).to_string())
I = pd.DataFrame(all_rows["isomers"])
print("\n4. Isomer pair distance / median azo nearest-neighbour distance (0 = indistinguishable)")
print(I.pivot_table(index="pair", columns="featurizer", values="rel_to_median_nn").round(2).to_string())
