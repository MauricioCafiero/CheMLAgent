#!/usr/bin/env python
"""Baseline-test a local (SMILES, target) CSV with RF+ECFP and
MLP+Mordred+PCA, via direct chemlagent function calls (no Ollama).

Consolidated from three near-duplicate copies written during the CheMLAgent
big_datasets investigation (run_ld50_test.py, run_deltah_clean.py,
run_watersol_clean.py) -- each ran the same prepare -> featurize -> train ->
evaluate pipeline on a different CSV. See ../../INVESTIGATION.md for what
those runs found.

Trains two models for a quick before/after or dataset-quality read:
  - RandomForestRegressor on ECFP fingerprints (fast, robust baseline)
  - the tuned-default wide-and-deep MLP on Mordred descriptors, cleaned +
    standardized + PCA-reduced to 95% variance (chemlagent.pytorch_mlp's
    prep_data pipeline) -- this combination outperformed ECFP+MLP in every
    comparison run during the investigation.

For large CSVs, Mordred featurization is slow (single-threaded, ~0.1-0.15s/
molecule) -- use --mlp-max-rows to cap the MLP+Mordred path to a random
subset while RF+ECFP still runs on the full set (this is what was done for
water_sol_data_clean.csv, ~18k rows).

Usage:
    uv run python big_datasets/scripts/test_dataset.py \\
        --input-csv big_datasets/deltah_fusion_data_clean.csv \\
        --run-id deltah_clean_full --limit 0
"""
import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from chemlagent import tools
from chemlagent.models import random_forest_regression
from chemlagent.pytorch_mlp import MLP_Model, prep_data, train, evaluate_regression


def show(label, d):
    print(f"\n--- {label} ---")
    print(json.dumps(d, indent=2, default=float))


def run_rf_ecfp(run_id, input_csv, limit, n_estimators, random_state):
    prep = tools.prepare_chembl_csv(input_csv=input_csv, run_id=run_id, limit=limit)
    show("prepare_chembl_csv", prep)
    feat = tools.featurize_fingerprints(run_id=run_id, fp_type="ECFP")
    show("featurize_fingerprints (ECFP)", feat)

    feats = np.load(os.path.join("runs", run_id, "features.npz"))
    X_train, X_test = feats["X_train"], feats["X_test"]
    y_train, y_test = feats["y_train"], feats["y_test"]
    r2, r2_train, model = random_forest_regression(
        X_train, y_train, X_test, y_test,
        n_estimators=n_estimators, random_state=random_state)
    mae = float(np.mean(np.abs(model.predict(X_test) - y_test)))
    result = {"n_train": int(feat["n_train"]), "n_test": int(feat["n_test"]),
              "train_r2": float(r2_train), "test_r2": float(r2), "test_mae": mae}
    show("RF + ECFP", result)
    return prep, result


def run_mlp_mordred_pca(run_id, input_csv, epochs, patience, min_delta,
                        neurons, num_hidden_layers, lr, weight_decay, pca_var,
                        batch_size):
    prep = tools.prepare_chembl_csv(input_csv=input_csv, run_id=run_id, limit=0)
    show("prepare_chembl_csv", prep)
    feat = tools.featurize_fingerprints(run_id=run_id, fp_type="Mordred")
    show("featurize_fingerprints (Mordred)", feat)

    feats = np.load(os.path.join("runs", run_id, "features.npz"))
    X_train, X_test = feats["X_train"], feats["X_test"]
    y_train, y_test = feats["y_train"], feats["y_test"]

    rd = os.path.abspath(os.path.join("runs", run_id))
    cwd = os.getcwd()
    os.chdir(rd)
    input_dims = None
    epochs_run = 0
    diverged = False
    try:
        prep_d = prep_data(batch_size=batch_size, shuffle=True, reduce_dim="pca",
                           pca_var=pca_var)
        _, _, train_loader, _ = prep_d.create_data_loader(X_train, y_train, X_test, y_test)
        input_dims = int(prep_d.X_train.shape[1])
        mlp = MLP_Model(neurons=neurons, input_dims=input_dims,
                        num_hidden_layers=num_hidden_layers, skip_connection=True)
        loss_fn = nn.MSELoss()
        opt = torch.optim.SGD(mlp.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        stale = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            epoch_loss = train(train_loader, mlp, loss_fn, opt)
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        train_r2, test_r2 = evaluate_regression(
            prep_d.X_train, np.asarray(y_train), prep_d.X_test, np.asarray(y_test), mlp)
        plt.close("all")
    except Exception as e:
        # SGD can blow weights up to NaN (most often at a high lr) -- record
        # the divergence instead of crashing the whole script.
        train_r2 = float("nan")
        test_r2 = float("nan")
        diverged = True
        print(f"  [diverged] {e}")
    finally:
        os.chdir(cwd)

    result = {"n_train": int(feat["n_train"]), "n_test": int(feat["n_test"]),
              "n_components": input_dims, "epochs_run": epochs_run,
              "train_r2": float(train_r2), "test_r2": float(test_r2),
              "diverged": diverged}
    show("MLP + Mordred + PCA", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", required=True,
                    help="Local CSV with a SMILES column and a target/activity column.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="prepare_chembl_csv row cap, 0 = keep all rows. Default 0 "
                         "(NOT the tool's own default of 2000) -- an alphabetically- "
                         "or otherwise-sorted input CSV silently biases a 2000-row "
                         "truncation; shuffle the CSV first if you do cap it.")
    ap.add_argument("--mlp-max-rows", type=int, default=0,
                    help="If >0, run RF+ECFP on the full --input-csv but MLP+Mordred "
                         "on a random subset of this many rows (Mordred is slow, "
                         "~0.1-0.15s/molecule single-threaded). The input CSV should "
                         "already be row-shuffled for this subset to be representative.")
    ap.add_argument("--skip-rf", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--rf-n-estimators", type=int, default=100)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--mlp-epochs", type=int, default=500)
    ap.add_argument("--mlp-patience", type=int, default=75)
    ap.add_argument("--mlp-min-delta", type=float, default=1e-5)
    ap.add_argument("--mlp-neurons", type=int, default=250)
    ap.add_argument("--mlp-layers", type=int, default=1, help="num_hidden_layers")
    ap.add_argument("--mlp-lr", type=float, default=0.002)
    ap.add_argument("--mlp-weight-decay", type=float, default=0.2)
    ap.add_argument("--mlp-pca-var", type=float, default=0.95)
    ap.add_argument("--mlp-batch-size", type=int, default=32)
    args = ap.parse_args()

    rf_result = None
    mlp_result = None

    if not args.skip_rf:
        _, rf_result = run_rf_ecfp(args.run_id, args.input_csv, args.limit,
                                   args.rf_n_estimators, args.random_state)

    if not args.skip_mlp:
        mlp_csv = args.input_csv
        mlp_run_id = args.run_id
        if args.mlp_max_rows > 0:
            mlp_csv = os.path.join(os.path.dirname(os.path.abspath(args.input_csv)),
                                   f"_{args.run_id}_mlp_subset.csv")
            pd.read_csv(args.input_csv).head(args.mlp_max_rows).to_csv(mlp_csv, index=False)
            mlp_run_id = f"{args.run_id}_mlp_subset"
            print(f"\nMLP path: using a {args.mlp_max_rows}-row subset -> {mlp_csv} "
                  f"(run_id={mlp_run_id})")
        mlp_result = run_mlp_mordred_pca(
            mlp_run_id, mlp_csv, args.mlp_epochs, args.mlp_patience, args.mlp_min_delta,
            args.mlp_neurons, args.mlp_layers, args.mlp_lr, args.mlp_weight_decay,
            args.mlp_pca_var, args.mlp_batch_size)

    print("\n========== SUMMARY ==========")
    if rf_result:
        print(f"RF + ECFP:          train_r2={rf_result['train_r2']:.3f}  "
              f"test_r2={rf_result['test_r2']:.3f}")
    if mlp_result:
        print(f"MLP + Mordred+PCA:  train_r2={mlp_result['train_r2']:.3f}  "
              f"test_r2={mlp_result['test_r2']:.3f}")


if __name__ == "__main__":
    main()
