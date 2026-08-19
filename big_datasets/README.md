# big_datasets/

Five open-chemistry datasets pulled from a single source, audited for data
quality, cleaned where safe, and modeled with the **same CheMLAgent tools**
used on ChEMBL/local CSVs on `main` — `prepare_chembl_csv`,
`featurize_fingerprints` (ECFP / Mordred+PCA), `train_model` (RF), and the
PyTorch MLP (`chemlagent.pytorch_mlp`). This folder is the data + the
reusable scripts that came out of that investigation; the **full writeup —
the audit, per-dataset results, grid-scan findings, and cross-dataset
takeaways — lives in [`../INVESTIGATION.md`](../INVESTIGATION.md)** at the
repo root. Read that for the narrative; this file is the map.

This work lives on the `linux-bigdata` branch, which is meant to stay
diverged from `main` rather than merge back into it.

## Datasets

Every file shares one export format, and every one had at least one of the
same handful of problems (target column won't autodetect; mixed/signed
units; unescaped quotes crashing `pandas.read_csv`; duplicate SMILES; a
wide outlier tripping the auto-log-transform). See the
[cleaning checklist](../INVESTIGATION.md#cleaning-checklist-for-future-data-pulls)
before starting on a new pull from this source.

| file | property | raw rows | verdict |
|---|---|---|---|
| `ld50_data.csv` | acute toxicity (LD50) | 2199 | 🔴 **broken as a single endpoint** — mixes ~20 species, 11 administration routes, 9 raw units into one "mg/kg" column. No clean version produced; see the [ld50 deep dive](../INVESTIGATION.md#ld50_datacsv-deep-dive). |
| `pka_mc.csv` | acidity (pKa) | 14248 | 🟡 usable once cleaned — the `pKa {measured, converted}` column is wrong by construction (missing sign flip); **use the raw `pKa` column**. Filter to `[-10, 30]` to drop 5 mis-entered outliers. |
| `pka_mc_clean.csv` | pKa, deduplicated | 10862 | clean (median-aggregated by SMILES, shuffled). |
| `pka_mc_clean_scaffold2k.csv` | pKa, scaffold-diverse ~2K sample | 2015 | one representative per Murcko scaffold — tractable size for a 243-combo grid scan. |
| `water_sol_data.csv` | water solubility | 53488 | 🔴 **malformed file, good data once fixed** — 8 rows have unescaped embedded quotes that crash `pandas.read_csv`; raw column mixes 21 units. Use the `{measured, converted}` g/L column. |
| `water_sol_data_clean.csv` | water solubility, deduplicated | 18484 | clean (log10-space median-aggregated, shuffled). |
| `water_sol_data_scaffold2k.csv` | water solubility, scaffold-diverse ~2K | 2000 | one representative per scaffold (2857 scaffolds found → 2000 random distinct). |
| `bbb_perm_data.csv` | blood-brain barrier permeability (logBB) | 6118 | 🟢 cleanest dataset — correct sign convention, tight duplicate agreement; 87% duplication → 1626 unique compounds. |
| `bbb_perm_data_clean.csv` | logBB, deduplicated | 1626 | clean (median-aggregated, shuffled). |
| `deltah_fusion_data.csv` | enthalpy of fusion (kJ/mol) | 2379 | 🟢 clean — single consistent unit, no species/route confound (pure physical property). |
| `deltah_fusion_data_clean.csv` | ΔHfus, deduplicated | 2199 | clean (median-aggregated, shuffled). |

Best MLP grid-scan test R² per dataset (full numbers + per-hyperparameter
analysis in `../INVESTIGATION.md`):

| dataset | best MLP test R² | notes |
|---|---|---|
| bbb_perm | **0.526** | every one of 243 combos positive (zero divergence); the most *positive* grid |
| water_sol (scaffold-2K) | **0.564** | the most *consistent* grid (std 0.012 across all 243 combos) |
| deltah_fusion | **0.572** | highest peak, but noisy / outlier-prone (one combo at −47) |
| pka (scaffold-2K) | **0.325** | RF still wins overall (0.467); pKa is inherently multi-valued per molecule |
| ld50 | — | never above "can't beat the mean" — label isn't a well-defined function of structure |

## Scripts

`scripts/` holds the four parameterized, argparse-driven scripts consolidated
from the one-off scratch work. Recommended order for a new file from this data
source: **`audit_csv.py`** first (decide whether it's safe to clean), then
**`clean_csv.py`**, then **`test_dataset.py`**, then optionally
**`mlp_grid_scan.py`**.

- **`scripts/audit_csv.py`** — decision-support scan that **prints a report,
  doesn't clean**: duplicate-SMILES rate and disagreement (log10 scale, worst
  groups printed in full across every detected condition column — reproduces
  ld50's species/route mixing automatically), every condition-like column
  (organism, route, tissue, solvent, assay, …) with value counts, and the raw
  column's `UNIT{...}` breakdown. Run this **before** `clean_csv.py` on any new
  file.
  ```bash
  uv run python big_datasets/scripts/audit_csv.py \
      --input big_datasets/newfile.csv --target-col "Some Raw Property"
  ```

- **`scripts/clean_csv.py`** — general-purpose cleaner for this source's
  format: tolerant parsing, target-column selection + outlier filtering,
  SMILES deduplication (median by default), optional Murcko-scaffold-diverse
  downsampling (`--scaffold-sample`), and a final shuffle.
  ```bash
  uv run python big_datasets/scripts/clean_csv.py \
      --input big_datasets/newfile.csv --target-col "Some Raw Property" \
      --output big_datasets/newfile_clean.csv --min -10 --max 30 --scaffold-sample 2000
  ```
  > **⚠️ Not safe for biological/bioactivity data with multiple assay
  > conditions** (organism, route, tissue, cell line, assay type, …) without
  > checking first. Its dedup groups by SMILES alone with no notion of a
  > condition column, so pointing it at a file like `ld50_data.csv` would
  > silently average together measurements that are not the same endpoint.
  > Safe for `deltaH`/`water_sol`/`bbb_perm`/`pKa`-style physicochemical /
  > single-endpoint properties; **deliberately not run on `ld50_data.csv`**.
  > For any bioactivity-style file, do the condition-column cross-tab
  > (checklist item #6) by hand first.

- **`scripts/test_dataset.py`** — RF+ECFP and MLP+Mordred+PCA baseline on a
  local (SMILES, target) CSV, via direct `chemlagent.tools` /
  `chemlagent.pytorch_mlp` calls (no Ollama). `--mlp-max-rows` caps the slow
  Mordred path (~0.1–0.15 s/molecule, single-threaded) to a subset while RF
  still trains on the full set.
  ```bash
  uv run python big_datasets/scripts/test_dataset.py \
      --input-csv big_datasets/deltah_fusion_data_clean.csv \
      --run-id deltah_clean_full --limit 0
  ```

- **`scripts/mlp_grid_scan.py`** — sweeps
  `neurons × num_hidden_layers × lr × weight_decay × pca_var` (3 values each
  = 243 combos by default) over a run's saved `runs/<run-id>/features.npz`
  via direct calls into `chemlagent.pytorch_mlp` (the `train_mlp` tool only
  exposes `epochs`/`patience`/`min_delta`, not architecture or optimizer
  hyperparameters). Resumable (reruns with the same `--out` skip completed
  rows); catches divergent combos (SGD → NaN) and records `diverged=True`
  instead of crashing. Results land in `scripts/results/`.
  ```bash
  uv run python big_datasets/scripts/mlp_grid_scan.py \
      --run-id deltah_clean_full \
      --out big_datasets/scripts/results/deltah_mlp_grid.csv
  ```

  > Note: `main` now ships an Ollama-callable **`mlp_scan`** tool that does a
  > bounded 5-fold-CV version of this same scan (≤3 values/param, ≤6 combos,
  > refits + saves the best like `train_mlp`). This script is the unbounded,
  > single-split, resumable-CSV variant used for the 243-combo sweeps recorded
  > here.

## Results

`scripts/results/` holds the grid-scan CSVs produced by `mlp_grid_scan.py`:

- `bbb_mlp_grid.csv` — bbb_perm (clean), 243 combos, best 0.526, zero divergence.
- `watersol_scaffold2k_mlp_grid.csv` — water_sol (scaffold-2K), best 0.564, std 0.012.
- `pka_scaffold2k_mlp_grid.csv` — pKa (scaffold-2K), best 0.325.

(deltah_fusion's scan was run but its results CSV is not committed; see
`../INVESTIGATION.md` for the numbers.)