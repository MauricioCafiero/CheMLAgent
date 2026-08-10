# big_datasets Investigation

This document records a data-quality audit and modeling investigation of the
five CSVs in [`big_datasets/`](big_datasets/), plus a Python 3.14
compatibility fix found along the way. It lives on the `linux-bigdata`
branch, which is meant to stay diverged from `main` (the state on the
author's other, Mac, machine) rather than merge back into it.

Reusable code produced during this work lives under
**[`big_datasets/scripts/`](big_datasets/scripts/)** (not a top-level
`scripts/` dir) — see [Reusable scripts](#reusable-scripts) below.

## Contents

- [Environment fix: `n_jobs` pickling crash on Python 3.14](#environment-fix-n_jobs-pickling-crash-on-python-314)
- [Reusable scripts](#reusable-scripts)
- [Dataset quality audit](#dataset-quality-audit)
- [ld50_data.csv deep dive](#ld50_datacsv-deep-dive)
- [deltaH of fusion: clean + grid scan](#deltah-of-fusion-clean--grid-scan)
- [Water solubility: clean + test](#water-solubility-clean--test)
- [Cross-dataset takeaways](#cross-dataset-takeaways)

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
consolidated into two parameterized, argparse-driven scripts under
`big_datasets/scripts/`:

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

Both were smoke-tested against a real run before being committed.

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
| `pka_mc.csv` | 🔴 broken conversion | `pKa {measured, converted}` = `10^(+pKa)` instead of the chemically correct `10^(-pKa)` = Ka; e.g. pKa 9.67 → reported as 4.68×10⁹ instead of ~2×10⁻¹⁰. **Use the raw `pKa` column, never the converted one.** Also inherently multi-valued per molecule (polyprotic acids/bases) with the disambiguating group-ID column stripped down to 5 non-null rows out of 14248. |
| `water_sol_data.csv` | 🔴 malformed file, but good data once fixed | 8 rows have unescaped embedded quotes in the compound-name field, which crashes `pandas.read_csv` outright (`ParserError`), not just a warning — needs `on_bad_lines='skip', engine='python'`. Raw `Water solubility` column also mixes 21 different units including sign-flipped ones (`log(mol/L)` vs `-log(M)`); use the `{measured, converted}` g/L column instead. Once fixed: duplicate-SMILES agreement is good (median spread ≈0 in log10(g/L), only 0.7% disagree by >1000x). |
| `bbb_perm_data.csv` | 🟡 usable, smaller than it looks | 6118 rows but only 1626 unique SMILES (87% duplication). `logBB`'s converted-column sign convention is correct (unlike pKa). Species/tissue unlabeled for ~90%+ of rows, but duplicate agreement is good, suggesting one consistent in-vivo convention throughout. |
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

## Water solubility: clean + test

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
cleanly. This is the best result of the whole investigation, and makes
sense: water solubility is one of the best-studied, most learnable QSPR
endpoints in the cheminformatics literature, and this file turned out to be
the cleanest one once its structural (not label-consistency) problems were
fixed.

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
3. **Default MLP hyperparameters aren't universally safe.** `lr=0.002`
   (the `train_mlp` tuned default) is fine on ld50 and deltaH but diverges
   reliably on water_sol+Mordred+PCA. `lr=0.0005` is the safer fallback
   across every dataset tried here.
4. **`n_jobs=-1` and default `limit=2000` are both landmines worth
   knowing about**: the former crashes outright on Python 3.14 (fixed in
   `fingerprints.py`/`tools.py`), the latter silently biases which rows
   survive if the input CSV isn't randomly ordered (fixed by shuffling
   both clean CSVs at save time).
