#!/usr/bin/env python
"""Scan a bulk CSV from this data source for the gotchas that make
clean_csv.py unsafe to run blindly (see its docstring and the checklist in
INVESTIGATION.md) -- prints a decision-support report instead of cleaning
anything, so a human/agent can decide whether the file is a single
consistent endpoint (safe for clean_csv.py's blind SMILES dedup) or a
mixed-condition bioactivity file like ld50_data.csv (needs filtering to one
condition first, or isn't safe to dedup this way at all).

Reports, for a given target column:
  1. row counts (including how many the tolerant parser had to skip)
  2. duplicate-SMILES rate and how much the target value disagrees across
     duplicates (log10 scale, since these properties often span orders of
     magnitude) -- with the worst-disagreement groups printed in full,
     across every condition-like column, so the cause is visible at a
     glance (this is exactly how ld50_data.csv's species/route mixing was
     first spotted)
  3. every "condition-like" column found (organism, route, tissue, solvent,
     assay, ... -- auto-detected by keyword, or pass --condition-cols to
     override) with its value counts, so you can see what's actually being
     mixed together before deciding whether to filter
  4. the raw column's UNIT{...} value counts, if present, to catch a
     mixed-unit column like water_sol_data.csv's raw column (21 units)

Usage:
    uv run python big_datasets/scripts/audit_csv.py \\
        --input big_datasets/newfile.csv --target-col "Some Raw Property"
"""
import argparse

import numpy as np
import pandas as pd

CONDITION_KEYWORDS = {
    "organism", "species", "route", "administration", "tissue", "cell",
    "solvent", "assay", "method", "strain", "model", "phase", "temperature",
    "exposure", "duration", "ionic", "quality", "anion", "binding",
    "dataset", "clinical", "ph", "voc", "class",
}


def tokenize(col):
    return set(w.lower() for w in
              col.replace("{", " ").replace("}", " ").replace("_", " ")
              .replace("(", " ").replace(")", " ").split())


def find_condition_cols(columns, exclude):
    found = []
    for c in columns:
        if c in exclude:
            continue
        if tokenize(c) & CONDITION_KEYWORDS:
            found.append(c)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--target-col", required=True,
                    help="Raw column name (not '{measured, converted}').")
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--condition-cols", default=None,
                    help="Comma-separated column names to treat as condition "
                         "columns, overriding auto-detection.")
    ap.add_argument("--top-n-worst", type=int, default=5,
                    help="How many worst-disagreement duplicate-SMILES groups "
                         "to print in full detail.")
    ap.add_argument("--disagreement-threshold", type=float, default=1.0,
                    help="log10-unit threshold (default 1.0 = 10x) for "
                         "flagging a duplicate group as disagreeing.")
    args = ap.parse_args()

    raw = pd.read_csv(args.input, on_bad_lines="skip", engine="python")
    n_raw = len(raw)
    print(f"=== {args.input} ===")
    print(f"{n_raw} rows read (malformed rows silently skipped)")
    print(f"{len(raw.columns)} columns: {list(raw.columns)}")
    print()

    # --- target column + unit sanity -----------------------------------
    unit_col = f"UNIT {{{args.target_col}}}"
    converted_col = f"{args.target_col} {{measured, converted}}"
    print(f"--- target column: {args.target_col!r} ---")
    if unit_col in raw.columns:
        print(f"raw unit ({unit_col}) value counts:")
        print(raw[unit_col].value_counts(dropna=False).to_string())
        n_units = raw[unit_col].nunique(dropna=True)
        if n_units > 1:
            print(f"⚠️  {n_units} distinct raw units -- mixing them without "
                  f"conversion would combine incompatible values.")
    else:
        print(f"(no {unit_col!r} column found)")
    if converted_col in raw.columns:
        print(f"\n'{converted_col}' exists -- SPOT-CHECK it by hand against a "
              f"known reference value before trusting it (see checklist #2; "
              f"pka_mc.csv's was silently wrong).")
    print()

    # --- duplicate-SMILES analysis --------------------------------------
    sub = raw[[args.smiles_col, args.target_col]].dropna()
    sub[args.target_col] = pd.to_numeric(sub[args.target_col], errors="coerce")
    sub = sub.dropna(subset=[args.target_col])
    n_numeric = len(sub)
    n_unique = sub[args.smiles_col].nunique()
    dup = sub.groupby(args.smiles_col).filter(lambda g: len(g) > 1)
    print(f"--- duplicate-{args.smiles_col} analysis ---")
    print(f"{n_numeric} rows with a valid numeric target, {n_unique} unique "
          f"{args.smiles_col}")
    print(f"{len(dup)} rows ({100*len(dup)/max(1,n_numeric):.1f}%) share a "
          f"{args.smiles_col} with >=1 other row")

    span_col = None
    if len(dup):
        all_positive = (dup[args.target_col] > 0).all()
        if all_positive:
            vals = np.log10(dup[args.target_col])
            span_col = "log10 target"
        else:
            vals = dup[args.target_col]
            span_col = "target (linear -- has non-positive values, can't log)"
        dup = dup.assign(_val=vals)
        spread = dup.groupby(args.smiles_col)["_val"].agg(lambda s: s.max() - s.min())
        print(f"\n{span_col} spread across duplicate groups (describe):")
        print(spread.describe().to_string())
        t = args.disagreement_threshold
        n_over = (spread > t).sum()
        print(f">{t} unit disagreement: {n_over} / {len(spread)} groups "
              f"({100*n_over/len(spread):.1f}%)")

        cond_cols = (args.condition_cols.split(",") if args.condition_cols
                    else find_condition_cols(raw.columns, {args.smiles_col, args.target_col}))
        worst = spread.sort_values(ascending=False).head(args.top_n_worst)
        if len(worst) and worst.iloc[0] > 0:
            print(f"\n--- worst {len(worst)} disagreeing groups "
                  f"(full rows, all condition-like columns) ---")
            show_cols = [args.smiles_col, args.target_col] + \
                [c for c in cond_cols if c in raw.columns]
            for smi, s in worst.items():
                print(f"\n{args.smiles_col}={smi}  (spread={s:.2f})")
                rows = raw[raw[args.smiles_col] == smi][show_cols]
                print(rows.to_string(index=False))
    print()

    # --- condition-like columns ------------------------------------------
    cond_cols = (args.condition_cols.split(",") if args.condition_cols
                else find_condition_cols(raw.columns, {args.smiles_col, args.target_col}))
    print(f"--- condition-like columns ({'user-specified' if args.condition_cols else 'auto-detected by keyword'}) ---")
    if not cond_cols:
        print("none found")
    for c in cond_cols:
        if c not in raw.columns:
            print(f"{c!r}: not in file, skipping")
            continue
        non_null = raw[c].notna().sum()
        print(f"\n{c!r} ({non_null}/{n_raw} non-null):")
        print(raw[c].value_counts(dropna=False).head(15).to_string())

    print("\n=== summary ===")
    if cond_cols and len(dup) and span_col and "log10" in span_col:
        print(f"{len(cond_cols)} condition-like column(s) found: {cond_cols}. "
              f"If duplicate disagreement above is wide, check whether it "
              f"correlates with these before assuming a single endpoint --"
              f"see the worst-groups printout above.")
    print("Decide: single consistent endpoint (safe for clean_csv.py) vs. "
          "needs filtering to one condition first (like ld50_data.csv) vs. "
          "not safe to blindly SMILES-dedup at all.")


if __name__ == "__main__":
    main()
