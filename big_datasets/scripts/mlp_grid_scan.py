#!/usr/bin/env python
"""Manual MLP hyperparameter grid scan over a run's saved raw features.npz.

Direct calls into chemlagent.pytorch_mlp (MLP_Model / prep_data / train /
evaluate_regression), bypassing the train_mlp tool -- which only exposes
epochs/patience/min_delta, not architecture or optimizer hyperparameters.
prep_data's clean (aggressive_cleaner) -> standardize -> PCA(pca_var)
pipeline runs on whatever raw feature matrix the run was featurized with
(ECFP, Mordred, etc.), so this works on any run_id that already has a
runs/<run_id>/features.npz.

Consolidated from three near-duplicate copies written during the CheMLAgent
big_datasets investigation (ld50+ECFP, ld50+Mordred, deltaH+Mordred) --
each only differed in --run-id/--out, so this is the one generalized
version. See ../../INVESTIGATION.md for what those three scans found.

Usage:
    uv run python big_datasets/scripts/mlp_grid_scan.py \\
        --run-id deltah_clean_full \\
        --out big_datasets/scripts/results/deltah_mlp_grid.csv

Every combination is trained with SGD + MSE loss, early stopping (patience/
min_delta), batch_size fixed (excluded from the sweep -- vary it by hand if
needed). A combination that diverges (SGD blows weights up to NaN, most
often at high lr) is caught and recorded with diverged=True / test_r2=nan
rather than crashing the scan. Results are appended to --out incrementally;
rerunning with the same --out resumes after the last completed row (rows
are produced in the same deterministic itertools.product order every time).
"""
import argparse
import csv
import itertools
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")  # headless, no display available

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from chemlagent.pytorch_mlp import MLP_Model, prep_data, train, evaluate_regression


def parse_floats(s):
    return [float(x) for x in s.split(",")]


def parse_ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True,
                    help="runs/<run-id>/features.npz must already exist "
                         "(i.e. featurize_fingerprints has been called for this run).")
    ap.add_argument("--out", required=True, help="Output CSV path (resumable).")
    ap.add_argument("--neurons", type=parse_ints, default=[64, 250, 500])
    ap.add_argument("--layers", type=parse_ints, default=[0, 1, 2],
                    help="num_hidden_layers -- additional sigmoid hidden layers "
                         "beyond the input layer (0 = input layer only).")
    ap.add_argument("--lr", type=parse_floats, default=[5e-4, 2e-3, 8e-3])
    ap.add_argument("--weight-decay", type=parse_floats, default=[0.05, 0.2, 0.5])
    ap.add_argument("--pca-var", type=parse_floats, default=[0.90, 0.95, 0.99])
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Fixed for the whole scan -- not swept.")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=75)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--skip-connection", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = os.path.abspath(os.path.join("runs", args.run_id))
    feats = np.load(os.path.join(run_dir, "features.npz"))
    X_train_raw, X_test_raw = feats["X_train"], feats["X_test"]
    y_train_raw, y_test_raw = feats["y_train"], feats["y_test"]

    combos = list(itertools.product(args.neurons, args.layers, args.lr,
                                    args.weight_decay, args.pca_var))

    fieldnames = ["idx", "neurons", "num_hidden_layers", "lr", "weight_decay",
                  "pca_var", "epochs_run", "n_features_after_pca", "train_r2",
                  "test_r2", "diverged", "seconds"]

    done = 0
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = max(0, sum(1 for _ in fh) - 1)
    print(f"Total combinations: {len(combos)}; resuming after {done} already done.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        t0_all = time.time()
        for idx, (neurons, n_hidden, lr, wd, pca_var) in enumerate(combos, start=1):
            if idx <= done:
                continue
            t0 = time.time()

            cwd = os.getcwd()
            os.chdir(run_dir)  # MLP_Model() writes MLP_model_params.txt to CWD
            input_dims = None
            epochs_run = 0
            diverged = False
            try:
                prep = prep_data(batch_size=args.batch_size, shuffle=True,
                                 reduce_dim="pca", pca_var=pca_var)
                _, _, train_loader, _ = prep.create_data_loader(
                    X_train_raw, y_train_raw, X_test_raw, y_test_raw)
                input_dims = int(prep.X_train.shape[1])
                model = MLP_Model(neurons=neurons, input_dims=input_dims,
                                  num_hidden_layers=n_hidden,
                                  skip_connection=args.skip_connection)
                loss_fn = nn.MSELoss()
                opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

                best_loss = float("inf")
                stale = 0
                for epoch in range(args.epochs):
                    epochs_run = epoch + 1
                    epoch_loss = train(train_loader, model, loss_fn, opt)
                    if epoch_loss < best_loss - args.min_delta:
                        best_loss = epoch_loss
                        stale = 0
                    else:
                        stale += 1
                        if stale >= args.patience:
                            break

                train_r2, test_r2 = evaluate_regression(
                    prep.X_train, np.asarray(y_train_raw), prep.X_test,
                    np.asarray(y_test_raw), model)
                plt.close("all")
            except Exception as e:
                train_r2 = float("nan")
                test_r2 = float("nan")
                diverged = True
                print(f"  [diverged] idx={idx}: {e}")
            finally:
                os.chdir(cwd)

            elapsed = time.time() - t0
            row = {
                "idx": idx, "neurons": neurons, "num_hidden_layers": n_hidden,
                "lr": lr, "weight_decay": wd, "pca_var": pca_var,
                "epochs_run": epochs_run, "n_features_after_pca": input_dims,
                "train_r2": train_r2, "test_r2": test_r2, "diverged": diverged,
                "seconds": round(elapsed, 2),
            }
            writer.writerow(row)
            fh.flush()

            total_elapsed = time.time() - t0_all
            n_left = len(combos) - idx
            eta = (total_elapsed / max(1, idx - done)) * n_left
            print(f"[{idx}/{len(combos)}] neurons={neurons} layers={n_hidden} "
                  f"lr={lr} wd={wd} pca_var={pca_var} -> test_r2={test_r2:.4f} "
                  f"train_r2={train_r2:.4f} ({elapsed:.1f}s, ETA {eta/60:.1f} min)")

    print("DONE")


if __name__ == "__main__":
    main()
