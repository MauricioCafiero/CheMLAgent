# Experiments

Exploratory studies on CHEMBL220 (AChE, log10 IC50), run 2026-11-01. They use
the package's own featurizers and model functions but are **not** part of the
agent. Outputs go to `experiments/results/` (git-ignored). Run all of them from
the repo root with `.venv/bin/python experiments/<script>.py`.

## 1. Poisoning-pocket hunt — `poison_pockets.py` (negative)

Idea: find a cluster of "ill-behaved" molecules that drags down training.
Recursively split the set in 2 (or k) with k-means on ECFP, train an RF on
RDKit_2D descriptors per cluster, and descend into the worse one. The two
featurizations keep the clustering from leaking the regression feature space.
Each level also runs a direct **poisoning test**: drop the cluster from the
global train split and score the rest of the global test set.

- Ranking clusters by R² (`--metric r2`) is misleading: R² tracks a cluster's
  target spread, so it "finds" narrow-range series, not bad molecules. Removing
  them never helped.
- Ranking by RMSE (`--metric rmse`, `--k 2` or `--k 6`) found the same pocket
  both times: ~900 tacrine/huprine hybrids (very potent, wide range, a few
  implausible values). Dropping it from training raised the rest's R² by
  ~+0.012–0.017 on one split, but loses the ability to predict that series.

## 2. Outlier-removal control — `outlier_control.py` (negative)

Drop the same number of train molecules (765) by IsolationForest / LOF on
descriptors + target, or at random. Generic outlier removal was **worse than
random** (−0.08 to −0.12 R²): it removes the informative edges of descriptor
space, and it barely overlaps the k-means pocket. The pocket beat random by
~0.03 R², but the effect was too small to pursue.

## 3. GP active learning — `active_learning_gp.py` (negative for AL)

Tanimoto-kernel GP on Morgan bits, growing the training set 100 → 3000 from a
MaxMin or random start, by max-variance or random acquisition (5 seeds).
Max-variance acquisition was worse than random until ~2300 molecules (it
chases isolated singletons); MaxMin seeds were too spread out for the GP to fit
at n = 100. The useful finding: the GP itself, on the full pool, scored R² 0.615
vs 0.485 for the RF baseline on the same split.

## 4. RF vs GP, model vs features — `gp_vs_rf.py` (positive → shipped)

2×2 {RF, GP} × {RDKit_2D, ECFP} on CHEMBL220 and 621-azo, 5 scaffold splits,
using the agent's own `get_fingerprints` and model functions.

| dataset   | RF + RDKit_2D | RF + ECFP   | GP + RDKit_2D | GP + ECFP    |
|-----------|---------------|-------------|---------------|--------------|
| CHEMBL220 | 0.500 ± 0.02  | 0.546 ± 0.05| 0.501 ± 0.04  | 0.612 ± 0.06 |
| 621-azo   | 0.710 ± 0.13  | 0.616 ± 0.14| 0.700 ± 0.12  | 0.719 ± 0.10 |

On CHEMBL220 features and model each contribute about half of the gain; the
GP only helps with the Tanimoto kernel on fingerprints.

Numbers above are after the salt fix (`clean_smiles` now keeps the largest
fragment, and training + inference featurize the cleaned SMILES -- before,
`get_fingerprints` cleaned the list but featurized the raw one, so ~17% of
CHEMBL220, mostly HCl/HBr/iodide salts, was featurized with counter-ions).
The fix changed R2 by -0.015..+0.004 (within split noise); the pre-fix run is
kept as `results/gp_vs_rf_pre_saltfix.csv`. Counter-ions likely tracked
chemical series, so removing them took away a small shortcut rather than
noise. Added to the agent as
`train_model(model_type="gp")` — see the README's Gaussian process section.

## 5. Spectral fingerprints, standalone — `gp_vs_rf.py --features Spectral_2D Spectral` (weaker alone; see §7)

Spectral fingerprints (Toney et al., Chem. Sci. 2026; `src/chemlagent/spectral_fp.py`,
64 graph-Laplacian eigenvalues) run through the same 5 scaffold splits, RF and
GP (RBF kernel on standardized features), reusing section 4's baselines.
Run 2026-11-04; log in `results/gp_vs_rf_spectral.log`.

| dataset | features    | RF R²        | GP R²        |
|---------|-------------|--------------|--------------|
| azo     | Spectral_2D | 0.593 ± 0.07 | 0.525 ± 0.10 |
| azo     | Spectral    | 0.574 ± 0.09 | 0.519 ± 0.09 |
| azo     | RDKit_2D    | 0.710 ± 0.13 | 0.700 ± 0.12 |
| azo     | ECFP        | 0.616 ± 0.14 | 0.719 ± 0.10 |

On azo both spectral variants trail every baseline, and the 3D conformer
version is no better than the bond-path (2D) one. CHEMBL220 (only these splits
ran; the run was stopped once the conformer cost was found — see below):

| split | Spectral_2D RF | Spectral_2D GP | Spectral RF | Spectral GP |
|-------|----------------|----------------|-------------|-------------|
| 132   | 0.326          | 0.370          | 0.295       | 0.324       |
| 7     | 0.364          | 0.415          | —           | —           |

vs 5-split baselines RDKit_2D 0.50 / 0.50 and ECFP 0.55 / 0.61 (RF / GP).
Standalone R² is not the whole question, though — §7 tests whether spectral
adds a different perspective.

Conformer cost (fixed). The first run crashed on a CHEMBL220 cage molecule
RDKit cannot embed (skfp `errors="raise"`); the second took ~45–60 min per
split. Profiling showed the time was almost all molecules that cannot be
embedded (up to 22 min each before failing; the other 999 of a 1,000 sample
took ~2 min on 6 cores). Now: 10 s embedding limit, unembeddable molecules
dropped without retry, conformers cached across splits / 3D types →
full CHEMBL220 3D featurization 837 s once, ~10 s per further split.

## 6. Cluster quality by featurizer — `cluster_compare.py` (k-means reliable; HDBSCAN data-dependent)

Table-1-style comparison (Toney et al.) of `chemlagent.clustering` (the
`cluster_molecules` core) on azo and CHEMBL220, scored in target space against
a Murcko-scaffold grouping (scaffolds with ≥ 5 members; the rest = noise).
U-test = fraction of cluster pairs with significantly different target
distributions (Mann–Whitney p < 0.05); DBI / CHI on the target. Run 2026-11-04
on cached features (CHEMBL220 3D Spectral not run — it needs a 14-min
conformer build); clustering itself takes < 4 s. Results:
`results/cluster_compare_{azo,CHEMBL220}.csv`.

| dataset   | features        | k-means: k / U-test / DBI | HDBSCAN: clusters / noise / U-test / DBI |
|-----------|-----------------|---------------------------|------------------------------------------|
| azo       | Murcko scaffold | 18 groups, 16% noise / 0.73 / 9.5 | |
| azo       | ECFP            | 6 / 0.47 / 6.4            | 5 / 9% / 0.90 / 6.5                      |
| azo       | RDKit_2D        | 11 / 0.66 / 16.8          | 10 / 44% / 0.89 / 8.3                    |
| azo       | Spectral_2D     | 8 / 0.86 / 5.2            | 9 / 46% / 0.78 / 27.8                    |
| azo       | Spectral        | 6 / 0.93 / 4.4            | 4 / 9% / 0.83 / 2.8                      |
| CHEMBL220 | Murcko scaffold | 244 groups, 52% noise / 0.71 / 840 | |
| CHEMBL220 | ECFP            | 7 / 0.91 / 9.4            | 26 / 31% / 0.88 / 26.7                   |
| CHEMBL220 | RDKit_2D        | 7 / 0.76 / 31.4           | 2 / 0% / 0.00 / 14.8                     |
| CHEMBL220 | Spectral_2D     | 6 / 0.87 / 10.6           | 2 / 0% / 0.00 / 12.6                     |

k-means (k at the inertia elbow) gives 6–11 clusters that separate the target
better than scaffolds in 6 of 7 cases. HDBSCAN works on azo and on CHEMBL220
with ECFP, but with continuous descriptors on CHEMBL220 it finds one dominant
cluster plus a small one that do not separate the target — the descriptor
space has no density gaps to split on (not fixed by min_cluster_size 10/25 or
min_samples default/5 in the tuning scan). Use k-means for such data.

How the defaults were chosen: a first run with silhouette-chosen k and
HDBSCAN (min_cluster_size = n/50, 50 PCA dims) picked k = 2 almost everywhere
(a one-pair U-test, uninformative) and labelled all of CHEMBL220 noise. A scan
on the cached features (k = 2..20; HDBSCAN at 50 vs 10 PCA dims,
min_cluster_size 10/25, min_samples default/5) led to the current defaults:
k at the inertia elbow, and HDBSCAN on 10 PCA dims with min_samples = 5 and
min_cluster_size = max(10, min(25, n/100)).

## 7. Does spectral add a different perspective? — `spectral_complementarity.py` (yes, but not a unique one)

Run 2026-11-04 on a 300-molecule random subset of each dataset (`--n 300`,
4.3 min; the full-size run was judged unnecessary). Same molecules for every
featurizer (those that embed in 3D). Results:
`results/complementarity_*_n300.csv`.

1. **Redundancy** — 3-fold CV R² of an RF reconstructing the target featurizer
   from a source featurizer (low = new information):

   | target (CHEMBL220) | from ECFP | from Mordred | from RDKit_2D |
   |--------------------|-----------|--------------|---------------|
   | Spectral           | 0.22      | 0.60         | 0.56          |
   | MORSE              | 0.35      | 0.51         | 0.51          |
   | RDF                | 0.35      | 0.64         | 0.64          |

   ~40% of spectral's variance is not recoverable from 2D descriptors — about
   the same as the existing 3D types (azo: 0.62–0.78, similar ordering).
2. **Neighbourhoods** — overlap (Jaccard) of each molecule's 10 nearest
   neighbours: spectral vs every other featurizer 0.08–0.19 (CHEMBL220) /
   0.19–0.30 (azo), the lowest of all; for calibration ECFP–RDKit_2D 0.27 /
   0.41, ECFP–MORSE 0.13 / 0.32, MORSE–RDF 0.15 / 0.30.
3. **Complementarity** (5 scaffold splits, RF): concatenating spectral onto
   ECFP or RDKit_2D did not improve R² (nor did MORSE / RDF; only
   ECFP+RDKit_2D helped slightly); spectral features cannot predict the base
   models' residuals (R² ≤ 0, as for MORSE); spectral-model errors correlate
   less with ECFP's (0.72–0.78) than MORSE's do (0.79–0.82).
4. **Unique capability** — E/Z and cis/trans pairs: separated by 3D Spectral,
   MORSE, RDF, E3FP; identical under every 2D type including Spectral_2D. R/S:
   only E3FP. Spectral has no capability the existing set lacks.

Verdict: a distinct (least-overlapping neighbourhoods, less-correlated errors)
but not unique view — the same kind of difference the existing 3D types give,
in a compact 64-dim form. Kept as an option in the agent.

