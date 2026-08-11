# big_datasets Investigation

This document records a data-quality audit and modeling investigation of the
five CSVs in [`big_datasets/`](big_datasets/), plus a Python 3.14
compatibility fix found along the way. It lives on the `linux-bigdata`
branch, which is meant to stay diverged from `main` (the state on the
author's other, Mac, machine) rather than merge back into it.

Reusable code produced during this work lives under
**[`big_datasets/scripts/`](big_datasets/scripts/)** (not a top-level
`scripts/` dir) — see [Reusable scripts](#reusable-scripts) below. All five
files in `big_datasets/` (and any future pull from the same source) share
one export format, and every one had at least one of the same handful of
problems — see the [cleaning checklist](#cleaning-checklist-for-future-data-pulls)
before starting on a new file.

## Contents

- [Environment fix: `n_jobs` pickling crash on Python 3.14](#environment-fix-n_jobs-pickling-crash-on-python-314)
- [Reusable scripts](#reusable-scripts)
- [Cleaning checklist for future data pulls](#cleaning-checklist-for-future-data-pulls)
- [Dataset quality audit](#dataset-quality-audit)
- [ld50_data.csv deep dive](#ld50_datacsv-deep-dive)
- [deltaH of fusion: clean + grid scan](#deltah-of-fusion-clean--grid-scan)
- [Water solubility: clean + test + grid scan](#water-solubility-clean--test--grid-scan)
- [Blood-brain barrier permeability (logBB): clean + grid scan](#blood-brain-barrier-permeability-logbb-clean--grid-scan)
- [pKa: clean + scaffold-diverse sample + grid scan](#pka-clean--scaffold-diverse-sample--grid-scan)
- [Cross-dataset takeaways](#cross-dataset-takeaways)
- [Status / next steps (pick up here)](#status--next-steps-pick-up-here)

## Environment fix: `n_jobs` pickling crash on Python 3.14

Fresh `uv venv --python 3.14 && uv sync --all-extras` install, `uv run
pytest` passed 24/25; `test_pipeline_round_trip` failed in
`featurize_fingerprints` with:

```
_pickle.PicklingError: Could not pickle the task to send it to the workers.
...
TypeError: cannot pickle '_abc._abc_data' object
```

Root cause: every `skfp` fingerprint/descriptor estimator in
`fingerprints.py` defaulted to `n_jobs=-1`, which forks joblib/loky worker
processes. joblib's bundled `cloudpickle` cannot serialize objects
containing Python 3.14's new `_abc_data` C struct, so any multiprocess
fingerprint call crashes on this Python version. Fixed by pinning
`n_jobs=1` everywhere in `fingerprints.py` and the two call sites in
`tools.py` (`featurize_fingerprints`'s `get_fingerprints` and
`SmilesFeaturizer` construction) — mirrors the `n_jobs=1` pattern
`train_model` already used for lightgbm/RF to dodge the Apple-Silicon
libomp double-load segfault. All 25 tests pass after the fix.

## Reusable scripts

Six near-duplicate one-off scratch scripts (three copies of a 243-combo MLP
hyperparameter grid scan, three copies of a prepare→featurize→train→evaluate
baseline test, each differing only in which run/CSV they pointed at) were
consolidated into parameterized, argparse-driven scripts under
`big_datasets/scripts/`. Recommended order for a new file from this data
source: **`audit_csv.py`** first (decide whether it's safe to clean), then
**`clean_csv.py`**, then **`test_dataset.py`**, then optionally
**`mlp_grid_scan.py`**.

- **`big_datasets/scripts/audit_csv.py`** — decision-support scan, prints a
  report rather than cleaning anything: duplicate-SMILES rate and
  disagreement (log10 scale, with the worst-disagreement groups printed in
  full across every detected condition column — this is how ld50's
  species/route mixing was first found, and the script reproduces that
  exact finding automatically), every "condition-like" column found by
  keyword (organism, route, tissue, solvent, assay, ...) with its value
  counts, and the raw column's `UNIT{...}` breakdown. Run this **before**
  `clean_csv.py` on any new file — it's what tells you whether the file is
  a single consistent endpoint or needs the `ld50_data.csv` treatment (see
  the warning on `clean_csv.py` below).

  ```bash
  uv run python big_datasets/scripts/audit_csv.py \
      --input big_datasets/newfile.csv --target-col "Some Raw Property"
  ```

- **`big_datasets/scripts/mlp_grid_scan.py`** — sweeps
  `neurons × num_hidden_layers × lr × weight_decay × pca_var` (3 values each
  by default = 243 combos) over any run's saved `runs/<run-id>/features.npz`
  via direct calls into `chemlagent.pytorch_mlp` (`train_mlp`'s tool
  interface only exposes `epochs`/`patience`/`min_delta`, not architecture
  or optimizer hyperparameters). Resumable (reruns with the same `--out`
  skip already-completed rows); catches divergent combinations (SGD blows
  weights to NaN) and records `diverged=True` instead of crashing.

  ```bash
  uv run python big_datasets/scripts/mlp_grid_scan.py \
      --run-id deltah_clean_full \
      --out big_datasets/scripts/results/deltah_mlp_grid.csv
  ```

- **`big_datasets/scripts/test_dataset.py`** — RF+ECFP and
  MLP+Mordred+PCA baseline on a local (SMILES, target) CSV, via direct
  `chemlagent.tools`/`chemlagent.pytorch_mlp` calls (no Ollama). Supports
  `--mlp-max-rows` to cap the slow Mordred path (~0.1-0.15s/molecule,
  single-threaded per the `n_jobs=1` fix above) to a subset while RF still
  trains on the full set — used for `water_sol_data_clean.csv` below.

  ```bash
  uv run python big_datasets/scripts/test_dataset.py \
      --input-csv big_datasets/deltah_fusion_data_clean.csv \
      --run-id deltah_clean_full --limit 0
  ```

- **`big_datasets/scripts/clean_csv.py`** — general-purpose cleaner for
  this data source's format (see the checklist below): tolerant parsing,
  target-column selection + outlier filtering, SMILES deduplication
  (median by default), optional Murcko-scaffold-diverse downsampling to a
  target size (`--scaffold-sample`, the technique used for pKa below), and
  a final shuffle. Consolidates the four near-identical one-off cleaning
  scripts written for deltaH/water_sol/bbb_perm/pKa in this investigation.

  ```bash
  uv run python big_datasets/scripts/clean_csv.py \
      --input big_datasets/newfile.csv --target-col "Some Raw Property" \
      --output big_datasets/newfile_clean.csv \
      --min -10 --max 30 --scaffold-sample 2000
  ```

  **⚠️ Not safe for biological/bioactivity data with multiple assay
  conditions (organism, administration route, tissue, cell line, assay
  type, ...) without checking first.** `clean_csv.py`'s dedup blindly
  groups by SMILES alone — it has no notion of a condition column, so
  pointing it at a file like `ld50_data.csv` (mixed species/routes/units)
  would silently reproduce that exact problem: averaging together
  measurements that are not the same endpoint, with no warning. It's safe
  for `deltaH`/`water_sol`/`bbb_perm`/`pKa`-style physicochemical or
  single-well-defined-endpoint properties (where duplicate-SMILES
  agreement was checked and found tight — see the quality audit) but was
  deliberately **not** run on `ld50_data.csv` for this reason. For any new
  bioactivity-style file, do the cross-tabulation-against-condition-columns
  check (checklist item #6) by hand *before* running `clean_csv.py`, and
  either filter to one consistent condition first or don't use this script
  at all if no single condition dominates the file.

All four were smoke-tested against a real run before being committed —
`clean_csv.py` against every already-known cleaning result in this document
(both the plain-dedup and the scaffold-sampling code paths), and
`audit_csv.py` against `ld50_data.csv` (correctly reproduced the
species/route mixing finding, worst-groups printout included) and
`deltah_fusion_data.csv` (correctly showed a single raw unit, zero
condition-like columns, and tight duplicate agreement).

## Cleaning checklist for future data pulls

Every one of the five `big_datasets/` files hit at least one of these.
Check for all of them before trusting a new pull from the same source:

1. **The target column won't autodetect.** `prepare_chembl_csv`'s
   `_TARGET_HINTS` never matched any of the five files' raw or converted
   column names — plan to pass `--target-col` (to `clean_csv.py`) or
   rename the column before calling `prepare_chembl_csv`.
2. **Spot-check the `{measured, converted}` column against a known
   reference value by hand before trusting it** — `audit_csv.py` flags
   when one exists but can't verify it's *correct*, only that it's
   present; the arithmetic check is still manual. It was silently wrong
   for pKa (missing sign flip: `10^(+pKa)` instead of `10^(-pKa)`), and
   the *raw* column was the safer choice for both pKa and logBB — but was
   *correct* and the better choice for water_sol and deltaH. There's no
   rule that always picks the right column; verify a handful of rows
   arithmetically (e.g. does `10^raw` or `10^-raw` reproduce the converted
   value? does multiplying by the known unit-conversion factor work out?).
3. **Try reading the raw file with pandas' default C parser first, and
   expect it to maybe fail outright** (`ParserError`, not just a warning)
   from unescaped embedded quotes in NAME-type fields — `water_sol_data.csv`
   did. `clean_csv.py`/`audit_csv.py` always read tolerantly
   (`on_bad_lines="skip", engine="python"`) so this is handled by default,
   but note how many rows got silently dropped.
4. **Check the raw (non-"converted") column's `UNIT{...}` value counts**
   before using it directly — it's often not one unit. `audit_csv.py`
   prints this automatically. `water_sol_data.csv`'s raw column mixed 21
   different units, some sign-flipped (`log(mol/L)` vs `-log(M)`). The
   `{measured, converted}` column is usually (not always, see #2) the
   safer bet specifically because it's unit-normalized.
5. **Filter physically-impossible outliers before handing the column to
   `prepare_chembl_csv`.** A wide-enough outlier (pKa's mis-entered
   perfluorocarboxylic acids at 2575-3128) can push the column range past
   `prepare_chembl_csv`'s auto-log-transform threshold (max-min ≥ 1000)
   and cause it to silently drop every negative-valued row when it
   shouldn't be log-transforming at all.
6. **Check duplicate-SMILES agreement before assuming the dataset is one
   consistent endpoint — do this manually, `clean_csv.py` does not check
   for you.** Run `audit_csv.py` first — it cross-tabulates disagreement
   against any species/route/solvent/tissue/assay-type/condition columns
   present automatically, and prints the worst-disagreeing groups in full
   so the cause is visible at a glance. `clean_csv.py`'s dedup blindly
   groups by SMILES alone with no awareness of such columns, so pointing
   it at an unaudited file risks silently reproducing ld50's exact problem
   (see the `clean_csv.py` warning above). ld50 turned
   out to mix ~20 species and 11 administration routes under one nominal
   unit; some files (logBB, deltaH, water_sol once cleaned) had tight
   duplicate agreement and no
   such confound. Median-aggregate duplicates (`--dedup median`, the
   default) rather than an arbitrary `keep="last"` when they do agree.
7. **Shuffle the cleaned CSV before saving** (`clean_csv.py` does this by
   default). `groupby`-based dedup leaves rows in alphabetical-SMILES
   order, and `prepare_chembl_csv`'s default `limit=2000` just takes
   `df.iloc[:limit]` for a local CSV with no shuffle — an unshuffled file
   hands it a structurally biased slice. This measurably hurt deltaH's
   results before it was fixed (see below).
8. **If the deduplicated file is too large for a full hyperparameter grid
   scan in reasonable time** (rule of thumb from this investigation: a
   243-combo scan on ~2000 rows takes low hours; pKa's 10862-row set
   would've taken ~11h), use `--scaffold-sample` to downsample to ~2000
   rows spread across as many distinct Murcko scaffolds as possible,
   rather than a plain random subsample (which risks skewing toward the
   most common structural motifs) — then confirm the sample is
   representative with a quick baseline test before committing to the
   full scan.
9. **Set a sleep inhibitor before launching anything that'll run
   unattended for more than ~1 hour** — see `CLAUDE.md`.

## Dataset quality audit

`prepare_chembl_csv`'s target-column autodetection (`_TARGET_HINTS` in
`data.py`) looks for column names containing `ic50`/`pic50`/`ec50`/`ki`/`kd`/
`potency`/`activity`/`standard_value`/`value`/`lmax`/... — **none of the
five `big_datasets/` CSVs match any of these**, so every one needs its
target column renamed (or `prepare_chembl_csv` needs a `target_col`
override, not currently implemented) before use.

| file | verdict | issue |
|---|---|---|
| `ld50_data.csv` | 🔴 broken as a single endpoint | mixes ~20 species, 11 administration routes, 9 raw units into one "mg/kg" column; see deep dive below |
| `pka_mc.csv` | 🟡 usable once cleaned | `pKa {measured, converted}` = `10^(+pKa)` instead of the chemically correct `10^(-pKa)` = Ka; e.g. pKa 9.67 → reported as 4.68×10⁹ instead of ~2×10⁻¹⁰. **Use the raw `pKa` column, never the converted one.** Also inherently multi-valued per molecule (polyprotic acids/bases) with the disambiguating group-ID column stripped down to 5 non-null rows out of 14248. Raw column also has 5 physically-impossible outliers (three perfluorocarboxylic acids mis-entered as pKa 2575-3128) that would otherwise trip `prepare_chembl_csv`'s auto-log-transform; see below. |
| `water_sol_data.csv` | 🔴 malformed file, but good data once fixed | 8 rows have unescaped embedded quotes in the compound-name field, which crashes `pandas.read_csv` outright (`ParserError`), not just a warning — needs `on_bad_lines='skip', engine='python'`. Raw `Water solubility` column also mixes 21 different units including sign-flipped ones (`log(mol/L)` vs `-log(M)`); use the `{measured, converted}` g/L column instead. Once fixed: duplicate-SMILES agreement is good (median spread ≈0 in log10(g/L), only 0.7% disagree by >1000x). |
| `bbb_perm_data.csv` | 🟢 clean, best-behaved dataset found | 6118 rows but only 1626 unique SMILES (87% duplication). `logBB`'s converted-column sign convention is correct (unlike pKa). Species/tissue unlabeled for ~90%+ of rows, but duplicate agreement is good, suggesting one consistent in-vivo convention throughout. Once cleaned, the *only* dataset in this investigation where every hyperparameter combo in a 243-combo grid scored a positive test R² — see below. |
| `deltah_fusion_data.csv` | 🟢 clean | single raw unit (kJ/mol, 100% consistent) verified against a valid kJ/mol→kcal/mol conversion; no species/route confound (pure physical property); low duplication (2199/2379 rows unique) with plausible measurement-precision-level disagreement (median 0.79 kJ/mol, max 8.3). |

## `ld50_data.csv` deep dive

The file's actual content, discovered by cross-tabulating `Organism` ×
`Administration route` × `UNIT {LC 50}`:

- **42% of all rows are earthworm/soil ecotoxicity** (`Eisenia
  foetida`/`fetida`/`andrei`, `earthworms`, `Lumbricus terrestris`, ...),
  not mammalian toxicology.
- **All 976 rat rows are inhalation-route** — there is no oral rat LD50 in
  this file at all, despite the filename.
- Raw units before the blanket "converted to mg/kg" step: `mg/L` (1007,
  aquatic concentration), `mg/kg` (987, **99% of which is earthworm data**,
  not oral dosing), `mcg/m3`/`ppm`/`g/m3` (inhalation air concentration),
  `-log(mmol/L)` (molar potency), `pph`, `ppb`.
- Of 608 compounds appearing more than once, 23% disagree by >10x, 17% by
  >100x, 5% by >1000x — purely from which species/route was tested, not
  structure. Worst case: 7.7 orders of magnitude for the identical SMILES
  (0.0000174 mg/kg rat-inhalation vs. 806 mg/kg earthworm).
- `prepare_chembl_csv`'s dedup (`drop_duplicates(keep="last")`) has no
  endpoint preference, so the 1090-row set every model in this
  investigation trained on turned out to be **45% earthworm data** by
  chance of file order.

Three RF+ECFP / MLP+Mordred+PCA comparisons were run to test whether
restricting to one consistent endpoint helps:

| dataset | rows (train/test) | RF test R² | MLP test R² |
|---|---|---|---|
| mixed (all species/routes) | 1090 (941/144) | −0.055 | −0.123 |
| earthworm-only (single endpoint) | 576 (410/164) | −0.072 | −0.433 |
| rat-inhalation-only (single endpoint, mg/L) | 236 (173/62) | −0.225 | −0.621 |

Both models get monotonically **worse** as the dataset shrinks — the
noisy-but-large mixed dataset beats both "cleaner" subsets outright. Every
way of getting a consistent endpoint here also means throwing away
50-80% of the rows, and that swamps any signal-quality gain at this size. A
243-combo MLP hyperparameter grid on the mixed ld50 data (ECFP: best test
R² −0.188; Mordred+PCA: best test R² −0.004) never got a mixed-endpoint
model above "can't beat the mean" — this dataset's label just isn't a
well-defined function of chemical structure, and no amount of tuning fixes
that.

## deltaH of fusion: clean + grid scan

`deltah_fusion_data.csv` (2379 rows, single unit kJ/mol) was
median-aggregated by SMILES (576→2199 unique compounds → 2199 clean rows;
no arbitrary `keep="last"` — legitimate repeated measurements of the same
physical property, so averaging is the principled choice) and shuffled
(fixed seed 42) before saving, to avoid `prepare_chembl_csv`'s default
`limit=2000` truncation silently biasing the row order (see below). Output:
`big_datasets/deltah_fusion_data_clean.csv`.

**The default `limit=2000` truncation mattered a lot** before the CSV was
shuffled: `groupby` output is alphabetically SMILES-sorted, so the first
2000 rows were a structurally skewed slice. Using all 2199 rows instead of
the first 2000:

| model | truncated (2000 rows) | full (2199 rows) |
|---|---|---|
| RF + ECFP | test R² = −0.579 | test R² = −0.159 |
| MLP + Mordred+PCA | test R² = −2.019 | **test R² = +0.180** |

+0.18 was the first positive test R² anywhere in this investigation. A
243-combo MLP grid scan (`big_datasets/scripts/mlp_grid_scan.py --run-id
deltah_clean_full`) then found:

- **Best: test R² = 0.572** (neurons=64, 1 hidden layer, lr=5e-4,
  weight_decay=0.05, pca_var=0.90 → 103 components; train R²=0.946).
- All 81 `lr=0.008` combos diverged, same failure mode as every ld50 scan.
- One non-diverged combo (500n/1L/lr=0.002/wd=0.05/pca=0.99) blew up to
  test R²=−47 — weights exploded to huge-but-finite values without hitting
  literal NaN, so it slipped past the divergence catch. This one outlier
  badly distorts *mean* test R² per hyperparameter (e.g. neurons=500 shows
  mean −0.60 despite a median of +0.29) — **medians are the reliable read
  here**, clustered around 0.24–0.37 across almost the entire non-lr=0.008
  grid.
- Unlike ld50 (where more `weight_decay` always helped), **less
  regularization did slightly better here** (wd=0.05 median 0.369 vs
  wd=0.50's 0.289) — consistent with there being real signal worth fitting
  rather than pure noise to guard against.

## Water solubility: clean + test + grid scan

`water_sol_data.csv` (53488 raw rows) was read past its 8 malformed rows
with `on_bad_lines="skip", engine="python"`, the `{measured, converted}`
g/L column used (not the 21-unit-mixed raw column), duplicate SMILES
median-aggregated in log10(g/L) space (appropriate given solubility spans
orders of magnitude) then converted back to g/L, and shuffled before
saving → 18484 unique-compound rows at
`big_datasets/water_sol_data_clean.csv`.

Tested via `big_datasets/scripts/test_dataset.py`. Full-set Mordred
featurization would take ~44 minutes single-threaded (`n_jobs=1`, see
above), so RF+ECFP ran on the full clean set while MLP+Mordred+PCA ran on a
5000-row subset (`--mlp-max-rows 5000`):

| model | test R² | notes |
|---|---|---|
| RF + ECFP (full 18484 rows) | **0.371** | train R²=0.965, MAE=1.41 (log10 g/L) |
| MLP + Mordred+PCA (5000-row subset) | **0.592** | train R²=0.737, 204 components |

The MLP result needed one hyperparameter change: the tuned default
`lr=0.002` diverged 3/3 attempts on this dataset/feature combination (SGD
blew weights to NaN even though this exact lr is safe on every other
dataset tried in this investigation); dropping to `lr=0.0005` — one of the
values the deltaH grid scan already flagged as generally safer — trained
cleanly. Water solubility is one of the best-studied, most learnable QSPR
endpoints in the cheminformatics literature, and this file turned out to be
the cleanest one once its structural (not label-consistency) problems were
fixed.

### Scaffold-diverse ~2K sample + grid scan

water_sol's 18484 clean rows is the largest file in `big_datasets/` and too
big for a 243-combo grid in reasonable time, same problem as pKa. Applying
`clean_csv.py --scaffold-sample 2000` directly to the raw file found
**2857 unique Murcko scaffolds** among 18455 usable compounds — *more*
scaffolds than the 2000 target this time (unlike pKa's near-exact 2015),
so the sample is 2000 randomly-chosen distinct scaffolds, one representative
each → `big_datasets/water_sol_data_scaffold2k.csv`.

Baseline on this sample: RF test R²=0.426, MLP test R²=0.532 (both stable
at `lr=0.0005`, no divergence) — comparable to the full-set baseline above,
confirming it's representative. A 243-combo grid scan (with the sleep
inhibitor active throughout this time — see `CLAUDE.md`) then found:

- **Best: test R² = 0.564** (64 neurons, 1 extra hidden layer, lr=2e-5,
  weight_decay=0.5, pca_var=0.99 → 291 components, train R²=0.656 — a
  **0.09 train/test gap**, the tightest of any best-combo in this
  investigation). Zero divergence across all 243 combos.
- **By far the most consistent result found**: min 0.493, max 0.564, mean
  0.536, median 0.537, **std 0.012** — every single combo landed in a
  narrow 0.49-0.56 band. Compare to bbb_perm's std of 0.046 or deltaH's
  wild, outlier-driven swings. Every hyperparameter effect is mild (lr
  matters most, and even that's only a 0.018 mean gap between 2e-5 and
  5e-4) — this property is robustly learnable close to regardless of how
  the network is configured.
- **Honest caveat: this grid's best (0.564) is slightly below the original
  single-run baseline on the larger, randomly-sampled 5000-row subset
  (0.592).** Two likely reasons, probably both at once: less training data
  here (1600 vs ~4000 rows), and the scaffold-diverse split deliberately
  maximizes train/test structural dissimilarity — a harder, arguably more
  meaningful generalization test than a random split, not a weaker
  featurization. The grid's real contribution isn't a new peak number; it's
  confirming the earlier 0.592 wasn't a lucky single run — the whole
  hyperparameter space around it works reliably.

## Blood-brain barrier permeability (logBB): clean + grid scan

`bbb_perm_data.csv` (6118 rows) was cleaned the same way as deltaH: the raw
`logBB` column used directly (correct sign convention, already log-scale,
range -6.0 to 2.03 — no auto-log-transform risk), median-aggregated by
SMILES (6118→1626 unique compounds), and shuffled before saving →
`big_datasets/bbb_perm_data_clean.csv`.

Baseline (`test_dataset.py`): **RF+ECFP test R² = 0.463** (train R²=0.960).
MLP+Mordred+PCA diverged at every learning rate tried down to `lr=5e-4`
(train_r2 and test_r2 both landing on astronomical finite values, e.g.
−3×10³⁶ — a more severe, harder-to-catch failure than the ld50/deltaH NaN
divergences, since these never hit the `try/except` NaN check). Stable
training only started at `lr≤1e-4`; `lr=2e-5` gave test R²=0.511 without
even triggering early stopping in 500 epochs.

Given that promise, a full 243-combo grid was run with a lr range shifted
down accordingly (`--lr "0.00002,0.00005,0.0001"` instead of the standard
`5e-4,2e-3,8e-3`):

- **Every one of the 243 combinations scored a positive test R²** — no
  divergence at all, min 0.120, mean 0.469, median 0.478, max **0.526**
  (250 neurons, 0 extra hidden layers, lr=2e-5, weight_decay=0.5,
  pca_var=0.95, train R²=0.676). This is the only dataset in the whole
  investigation where the entire hyperparameter grid worked.
- The train/test gap at the best combo (0.676 vs 0.526) is much tighter
  than deltaH's best (0.946 vs 0.572) — this model generalizes rather than
  just memorizing.
- `weight_decay=0.5` (mean 0.487) clearly beat `0.05` (mean 0.457), and
  `pca_var=0.95` (mean 0.486) beat both `0.90` and `0.99` — the worst 5
  combos are almost all at `pca_var=0.99` (dragging its mean down to 0.448
  despite a median of 0.471, the same mean-vs-median outlier pattern seen
  in the deltaH scan).
- `~99s → ~35s`-per-combo runtime: this scan took **~5.1 hours** wall-clock
  (started 19:07, finished ~00:1x), 3-4x longer than deltaH's, both because
  low-lr training rarely triggers early stopping (many combos ran the full
  500 epochs) and because of real CPU contention from concurrently running
  the pKa baseline test and diagnostics — worth remembering before sizing
  the next scan's ETA.

## pKa: clean + scaffold-diverse sample + grid scan

`pka_mc.csv` (14248 rows) used the raw `pKa` column (the converted column is
broken, see the quality-audit table), filtered to `[-10, 30]` to drop 5
physically-impossible outliers (three perfluorocarboxylic acids mis-entered
around pKa 2575-3128 — off by ~1000x from anything chemically real, which
would otherwise push the column range past `prepare_chembl_csv`'s
1000-unit auto-log-transform threshold and silently drop every negative-pKa
compound), median-aggregated by SMILES (14217→10862 unique compounds), and
shuffled before saving → `big_datasets/pka_mc_clean.csv`.

Baseline (`test_dataset.py --mlp-max-rows 5000 --mlp-lr 0.0001`, RF on the
full 10862 rows):

| model | test R² | notes |
|---|---|---|
| RF + ECFP (full 10862 rows) | **0.467** | train R²=0.946 |
| MLP + Mordred+PCA (5000-row subset, lr=0.0001) | 0.226 | train R²=0.419, 159 components, no divergence |

### Scaffold-diverse ~2K sample, for a grid scan at a tractable size

10862 unique compounds was too large for a 243-combo grid at the tuned
lr range in reasonable time (a timing probe on the 5000-row MLP subset
came back at ~166s/combo, ~11h uncontended estimate for the full set). A
Murcko scaffold breakdown found **2015 unique scaffolds** among the 10862
compounds (9 SMILES failed to parse/scaffold) — heavily skewed (benzene
alone: 2281 compounds, 21%; acyclic/no-ring: 1554, 14%; 63% of scaffolds
are singletons). Taking exactly one random representative per scaffold
gives full scaffold-diversity coverage at almost exactly the ~2K target,
with no extra quota-sampling logic needed → shuffled →
`big_datasets/pka_mc_clean_scaffold2k.csv` (2015 rows).

Baseline on this sample confirmed it's representative (RF test R²=0.397,
MLP test R²=0.240, both close to the full-set baseline above). A
243-combo grid scan (`--lr "0.00002,0.00005,0.0001"`, the same safe range
bbb_perm needed) then found:

- **Best: test R² = 0.325** (500 neurons, 0 extra hidden layers, lr=1e-4,
  weight_decay=0.2, pca_var=0.99 → 265 components, train R²=0.487). Zero
  divergence across all 243 combos; a tight, well-behaved distribution
  (min 0.155, mean 0.234, median 0.237, std 0.058) — no catastrophic
  outliers like deltaH's, but a clearly lower ceiling than bbb_perm's or
  deltaH's.
- **`pca_var` direction flips here** — `0.99` (mean 0.303) clearly beats
  `0.90` (mean 0.164), the opposite of bbb_perm (`0.95` best), ld50
  (`0.90` best), and deltaH (`0.90` among the better values). Consistent
  with pKa's chemistry: subtle, site-specific electronic effects appear to
  need more retained variance to capture, so compressing to 90% variance
  throws away real pKa-relevant signal.
- `neurons`/`num_hidden_layers` are flat again (mean ~0.23-0.24
  regardless) — capacity was never the bottleneck on any dataset in this
  investigation.
- The grid meaningfully beat the naive-default-pca_var baseline (0.325 vs
  0.240, ~35% relative gain, mostly from finding pca_var=0.99 over the
  default 0.95) — but **RF still wins overall on pKa** (0.467 baseline
  > any MLP grid result), unlike bbb_perm/water_sol where the tuned MLP
  eventually overtook RF. The lower ceiling versus bbb_perm/deltaH tracks
  with the quality audit's finding that pKa is inherently multi-valued per
  molecule (site-specific ionization ambiguity) — a cap no amount of
  tuning removes.
- This scan got killed twice mid-run by the machine's 1-hour idle
  auto-suspend (active even on AC power) before that was diagnosed and
  fixed with a `systemd-inhibit` lock — see `CLAUDE.md` (repo) and the
  global `~/.claude/CLAUDE.md` for the pattern. No progress was lost either
  time (`mlp_grid_scan.py` writes incrementally and resumes automatically),
  just wall-clock time to notice and relaunch.

## Cross-dataset takeaways

1. **File format bugs hide silently.** `water_sol_data.csv` crashes
   `pandas.read_csv` outright; `pka_mc.csv`'s "converted" column is
   physically wrong by construction (missing the sign flip in `Ka =
   10^-pKa`). Neither shows up unless you actually try to read/use the
   column — always spot-check a "measured, converted" column against a
   known reference value before trusting it.
2. **Sample size dominates label noise at this data scale.** Restricting
   ld50 to a single consistent endpoint (species/route/unit) never helped,
   because doing so also cut the dataset by 50-80%. The fair test — noisy
   mixed data vs. cleaned data *at matched size* — was never run; that
   would be the next experiment if ld50 is worth revisiting.
3. **Default MLP hyperparameters aren't universally safe, and the safe
   range varies a lot by dataset.** `lr=0.002` (the `train_mlp` tuned
   default) is fine on ld50 and deltaH but diverges reliably on
   water_sol+Mordred+PCA (`lr=0.0005` fixed it) *and* on bbb_perm+
   Mordred+PCA, which needed to go all the way down to `lr≤1e-4` — `5e-4`
   still diverged there. There's no single safe fallback lr; check the
   specific dataset/feature combination before committing to a grid scan's
   lr range, the way the bbb_perm section above did.
4. **`n_jobs=-1` and default `limit=2000` are both landmines worth
   knowing about**: the former crashes outright on Python 3.14 (fixed in
   `fingerprints.py`/`tools.py`), the latter silently biases which rows
   survive if the input CSV isn't randomly ordered (fixed by shuffling
   both clean CSVs at save time).
5. **Once cleaned, 4 of 5 files show real, positive, learnable signal**
   (bbb_perm, deltaH, water_sol, pKa) — the initial run of negative-R²
   results across this investigation was mostly a data-quality artifact
   (ld50's mixed endpoints, or a diverging default lr), not a sign that
   these properties are inherently hard to model from structure.
6. **`pca_var`'s optimal value is dataset-specific, not just its
   *safe* lr range.** More retained variance (0.99, more PCA components)
   was clearly *best* for pKa, but clearly *worst* (or a source of
   instability) for bbb_perm, ld50, and deltaH. Don't assume the direction
   of any hyperparameter effect transfers across datasets — rerun the
   analysis (or at least eyeball the per-hyperparameter means/medians) for
   each new one.
7. **One representative-per-Murcko-scaffold is a good way to shrink an
   oversized dataset for a grid scan** without a naive random subsample's
   risk of skewing toward the most common structural motifs. Worked both
   ways round: pKa's 10862 compounds collapsed to almost exactly 2015
   unique scaffolds (no extra quota logic needed), while water_sol's 18455
   collapsed to *more* scaffolds than the 2000 target (2857), exercising
   the random-subset-of-scaffolds path instead. Confirm the sample is
   representative with a quick baseline test before committing to a full
   grid scan on it either way.
8. **This machine auto-suspends after 1 hour idle, even on AC power**,
   and will silently kill unattended background jobs — see `CLAUDE.md`
   for the `systemd-inhibit` fix. Set the inhibitor *before* launching
   anything expected to run unattended for more than ~1 hour.
9. **A scaffold-diverse split can score *lower* than a random split on the
   same underlying data, without that meaning anything is wrong** —
   water_sol's scaffold-2K grid scan (best 0.564) never quite reached the
   earlier random-5000-subset single run (0.592), most likely because
   forcing train/test structural dissimilarity is a genuinely harder
   generalization test, not a worse model. Don't compare R² across
   different sampling schemes as if they were the same benchmark.
10. **Grid-scan result *consistency* (std across all 243 combos) is as
    informative as the best single number** — water_sol's scan (std=0.012,
    every combo landed 0.49-0.56) says this property is robustly learnable
    almost regardless of hyperparameters; deltaH's scan (wild swings, one
    combo at −47) says the opposite about that model/feature combination,
    even though its peak (0.572) was nominally higher than water_sol's
    (0.564). The peak number alone would have hidden that difference.

## Status / next steps (pick up here)

Where this session left off:

- **Done and pushed**: `n_jobs` fix, all four reusable scripts (including
  `audit_csv.py`, the decision-support scan), the sleep-inhibitor fix
  (`CLAUDE.md`, repo and global), and clean+tested results for all five
  files: ld50 (deep dive + mixed/earthworm/rat-inhalation comparison, no
  clean version — see takeaway #2), deltaH (clean + full grid scan, best
  0.572 but noisy/outlier-prone), bbb_perm (clean + full grid scan, best
  0.526, zero divergence, the most *positive* grid), pKa (clean +
  scaffold-diverse 2K sample + full grid scan, best 0.325 but RF wins
  overall at 0.467), and water_sol (clean + baseline + scaffold-diverse 2K
  grid scan, best 0.564 but the *most consistent* grid — std 0.012 across
  all 243 combos, see takeaway #10).
- **Not started**: no further cleaning was attempted on `ld50_data.csv`
  itself. The ld50 deep dive's own suggestion — subsample the mixed dataset
  down to match the earthworm/rat-inhalation subset sizes, to isolate
  whether label noise or sample size is really the bottleneck — was never
  run. That's the natural next experiment if ld50 is worth revisiting.
- **Every dataset in `big_datasets/` has now been through at least a
  baseline test, and 4 of 5 (all but ld50) have a full grid scan.** If
  there's a next dataset to investigate, it'd be a genuinely new pull from
  this data source rather than more work on these five — see the cleaning
  checklist and `audit_csv.py` above for where to start.
- **Branch hygiene**: `src/agent_template.py` and
  `src/chemlagent/modrag_protein_functions.py` still show as locally
  modified (pure CRLF/whitespace noise, verified byte-identical to HEAD
  after stripping `\r` — see the git history for this branch). They were
  deliberately left out of every commit; still true as of this writeup.
