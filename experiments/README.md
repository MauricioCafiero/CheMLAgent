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
