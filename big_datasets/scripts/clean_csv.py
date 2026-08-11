#!/usr/bin/env python
"""General-purpose cleaner for this data source's bulk CSV exports.

All five big_datasets/ files (and likely future pulls from the same source)
share one format: SMILES, CASRN, EXTERNALID, N, NAME, ARTICLEID, PAGE, TABLE,
<property>, UNIT{<property>}, <property> {measured, converted}, UNIT{...},
plus assay-specific context columns. Every file audited so far had at least
one of these problems -- see INVESTIGATION.md for the specifics:

  - the target column never matches prepare_chembl_csv's autodetect hints
  - malformed rows (unescaped embedded quotes) that crash pandas' C parser
  - a "measured, converted" column that's a naive/wrong unit conversion
    (pka_mc.csv's is off by a sign flip: 10^(+pKa) instead of 10^(-pKa))
  - heavy duplicate-SMILES repetition (usually legitimate repeat
    measurements, best median-aggregated rather than arbitrarily
    keep-last'd)
  - occasional physically-impossible outliers that would otherwise trip
    prepare_chembl_csv's auto-log-transform (range >= 1000)
  - alphabetically-sorted-by-default row order, which biases
    prepare_chembl_csv's default limit=2000 truncation if not shuffled away

This script runs the common steps (NOT the conversion-correctness check --
always spot-check a "measured, converted" column by hand first, see
INVESTIGATION.md's audit table for how that was done for each file here):
tolerant parsing, target selection + outlier filtering, median/mean/
keep-last deduplication by SMILES, optional Murcko-scaffold-diverse
downsampling to a target size, and a final shuffle.

NOT SAFE, without a manual check first, for biological/bioactivity data with
multiple assay conditions (organism, administration route, tissue, cell
line, assay type, ...). Deduplication here groups by SMILES alone -- it has
no notion of a condition column, so pointing this at an unchecked file
silently averages together measurements that may not be the same endpoint.
This is exactly what's wrong with ld50_data.csv (mixes ~20 species and 11
administration routes under one nominal unit) -- cross-tabulate duplicate
SMILES against any species/route/solvent/tissue/condition columns present
BEFORE running this script, and either filter to one consistent condition
first or don't use this script at all if no single condition dominates.
Safe for physicochemical or single-well-defined-endpoint properties where
that check has been done and duplicate-SMILES agreement is tight (as it was
for deltaH/water_sol/bbb_perm/pKa in this investigation).

Usage:
    uv run python big_datasets/scripts/clean_csv.py \\
        --input big_datasets/newfile.csv \\
        --target-col "Some Raw Property" \\
        --output big_datasets/newfile_clean.csv \\
        --min -10 --max 30 \\
        --scaffold-sample 2000
"""
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(mol))
    except Exception:
        return None


def scaffold_diverse_sample(df, n, seed):
    """Sample ~n rows spreading across as many distinct Murcko scaffolds as
    possible: one representative per scaffold first (full diversity
    coverage), then extra rounds pulling a 2nd/3rd/... representative from
    the scaffolds with the most remaining members, until n rows are hit
    (only relevant when n > the number of unique scaffolds) or every row
    is used. If there are more unique scaffolds than n, a random n of them
    are kept (one each) rather than trying to cover every scaffold.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["_scaffold"] = df["SMILES"].apply(murcko_scaffold)
    n_failed = df["_scaffold"].isna().sum()
    df = df.dropna(subset=["_scaffold"])

    groups = {s: g.sample(frac=1.0, random_state=seed).to_dict("records")
              for s, g in df.groupby("_scaffold")}
    scaffolds = list(groups.keys())
    rng.shuffle(scaffolds)
    n_scaffolds = len(scaffolds)

    picked = []
    if n_scaffolds >= n:
        for s in scaffolds[:n]:
            picked.append(groups[s][0])
    else:
        # one representative per scaffold first (full coverage)...
        for s in scaffolds:
            picked.append(groups[s].pop(0))
        # ...then round-robin extra representatives from scaffolds that
        # still have members, until n is hit or everything is used.
        remaining = [s for s in scaffolds if groups[s]]
        while len(picked) < n and remaining:
            for s in list(remaining):
                if len(picked) >= n:
                    break
                picked.append(groups[s].pop(0))
                if not groups[s]:
                    remaining.remove(s)

    out = pd.DataFrame(picked).drop(columns=["_scaffold"])
    return out, n_failed, n_scaffolds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--target-col", required=True,
                    help="Raw (not '{measured, converted}') column name to use as "
                         "the target -- verify its unit/conversion is trustworthy "
                         "by hand first (see module docstring).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--min", type=float, default=None,
                    help="Drop rows with target below this (physically-impossible "
                         "outlier filter).")
    ap.add_argument("--max", type=float, default=None)
    ap.add_argument("--dedup", choices=["median", "mean", "keep_last"], default="median",
                    help="How to collapse duplicate SMILES. median (default) is the "
                         "principled choice for legitimate repeat measurements of a "
                         "physical/biological property; keep_last matches "
                         "prepare_chembl_csv's own (arbitrary) behavior.")
    ap.add_argument("--scaffold-sample", type=int, default=0,
                    help="If >0, downsample to ~this many rows spread across as many "
                         "distinct Murcko scaffolds as possible (see module "
                         "docstring) -- use when the deduplicated set is too large "
                         "for a full grid scan in reasonable time.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = pd.read_csv(args.input, on_bad_lines="skip", engine="python")
    n_raw = len(raw)

    df = raw[[args.smiles_col, args.target_col]].dropna()
    df = df.rename(columns={args.smiles_col: "SMILES", args.target_col: "standard_value"})
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    n_numeric = len(df)

    n_outliers = 0
    if args.min is not None or args.max is not None:
        lo = args.min if args.min is not None else -np.inf
        hi = args.max if args.max is not None else np.inf
        before = len(df)
        df = df[(df["standard_value"] >= lo) & (df["standard_value"] <= hi)]
        n_outliers = before - len(df)

    n_before_dedup = len(df)
    n_unique_smiles = df["SMILES"].nunique()
    if args.dedup == "keep_last":
        df = df.drop_duplicates(subset=["SMILES"], keep="last")
    else:
        agg = "mean" if args.dedup == "mean" else "median"
        df = df.groupby("SMILES", as_index=False)["standard_value"].agg(agg)

    n_failed_scaffold = n_scaffolds = None
    if args.scaffold_sample > 0:
        df, n_failed_scaffold, n_scaffolds = scaffold_diverse_sample(
            df, args.scaffold_sample, args.seed)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"{args.input}:")
    print(f"  {n_raw} rows read (malformed rows silently skipped by the tolerant parser)")
    print(f"  {n_numeric} rows with a valid numeric {args.target_col!r}")
    if n_outliers:
        print(f"  {n_outliers} outliers dropped outside [{args.min}, {args.max}]")
    print(f"  {n_before_dedup} rows / {n_unique_smiles} unique SMILES before dedup "
          f"({args.dedup})")
    if args.scaffold_sample > 0:
        print(f"  {n_failed_scaffold} rows dropped (unparseable scaffold); "
              f"{n_scaffolds} unique Murcko scaffolds found")
    print(f"  -> {len(df)} clean rows -> {args.output}")


if __name__ == "__main__":
    main()
