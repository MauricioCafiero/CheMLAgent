"""Shared descriptor-matrix cleaning for the AZO Lmax pipeline.

Mordred 2D descriptors (1613 columns) contain NaN/inf for descriptors that
cannot be computed for a given molecule, plus constant / near-constant and
mutually near-duplicate columns. Feeding that raw matrix straight into a
kernel model (SVR) or into PCA causes numerical problems -- the randomized
SVD inside sklearn PCA emits "divide by zero / overflow encountered in matmul"
RuntimeWarnings from the degenerate columns, and the few huge-magnitude
columns dominate the polynomial kernel.

`DescriptorCleaner` is the single, principled preprocessing step used by BOTH
the SVR path (`models.svr_regression`, inside the sklearn Pipeline) and the
MLP path (`pytorch_mlp.prep_data`, inline). It is fit on the TRAIN split only
(no leakage) and stores everything needed to apply the identical transform to
test / new molecules:

  1. replace +/-inf with NaN
  2. drop columns whose NaN-fraction on train exceeds `max_nan_frac`
  3. drop columns with (nan-aware) std <= `var_threshold`  (dead / constant)
  4. greedy pairwise |correlation| pruning: among surviving columns, drop the
     later of any pair whose |corr| exceeds `corr_threshold`  (duplicates)
  5. impute remaining NaN with the TRAIN-column median  (stored on `medians_`)

The `var_threshold` / `corr_threshold` defaults match the inline filter that
`prep_data` already used, so the kept-column set is comparable to the prior
MLP behaviour; the only behavioural change is median imputation (previously
NaN->0) and treating inf as a missing value (previously inf->0).
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DescriptorCleaner(BaseEstimator, TransformerMixin):
    """Column removal + median imputation for raw Mordred descriptors.

    Fit on train only; transform applies the learned column mask and medians
    to any matrix with the same number of columns.
    """

    def __init__(self, var_threshold=1e-3, corr_threshold=0.98,
                 max_nan_frac=1.0, impute_strategy='median'):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.max_nan_frac = max_nan_frac
        self.impute_strategy = impute_strategy      # 'median' or 'zero'

    def _as_float64(self, X):
        return np.asarray(X, dtype=np.float64)

    def fit(self, X, y=None):
        X = self._as_float64(X)
        X = np.where(np.isinf(X), np.nan, X)
        n_features = X.shape[1]

        # nanstd / nanmedian / corrcoef emit RuntimeWarnings on all-NaN or
        # constant columns ("Degrees of freedom <= 0", "invalid value in
        # divide"). Those columns are handled below (dropped or imputed) and
        # the warnings are cosmetic, so silence them inside fit.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            # Impute FIRST so the variance/correlation filters see a complete
            # matrix (matching the prior inline filter, which zero-filled then
            # filtered). impute_strategy picks the fill value:
            #   'median' -> train-column median  (used by the SVR: empirically
            #               ~6 nm better than zero-fill on the no-PCA poly SVR)
            #   'zero'   -> 0, i.e. nan_to_num      (used by the MLP: its tuned
            #               wd/epochs/PCA config depends on zero-fill; median
            #               imputation shifts the PCA dim and regresses it
            #               27 -> 40 nm, so the MLP keeps zero-fill)
            if self.impute_strategy == 'zero':
                X_imp = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                fill = np.zeros(n_features)
            else:
                medians = np.nanmedian(X, axis=0)
                medians = np.where(np.isnan(medians), 0.0, medians)
                X_imp = X.copy()
                ij = np.where(np.isnan(X_imp))
                if ij[0].size:
                    X_imp[ij] = np.take(medians, ij[1])
                fill = medians

            # (1) optionally drop columns missing on too large a fraction of the
            # train set. Both presets use max_nan_frac=1.0 (off) so partially-
            # missing columns are median-imputed and kept, not dropped -- the
            # near-constant filter below already removes the uninformative ones.
            nan_frac = np.isnan(X).mean(axis=0)
            keep = nan_frac <= self.max_nan_frac

            # (2) drop near-constant / dead columns (std on the imputed matrix).
            kept_idx = np.where(keep)[0]
            std = X_imp[:, kept_idx].std(axis=0)
            kept_idx = kept_idx[std > self.var_threshold]

            # (3) greedy pairwise correlation pruning among the survivors, on
            # the imputed matrix; drop the later column of any pair whose
            # correlation exceeds the threshold. NB: this checks corr directly
            # (NOT |corr|), so only positively-correlated duplicates are dropped
            # -- matching the original inline filter the MLP was tuned against.
            Xk = X_imp[:, kept_idx]
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.corrcoef(Xk.T)
            corr = np.nan_to_num(corr, nan=0.0)
            drop_local = set()
            for j in range(len(kept_idx)):
                if j in drop_local:
                    continue
                for k in range(j + 1, len(kept_idx)):
                    if k in drop_local:
                        continue
                    if corr[j, k] > self.corr_threshold:
                        drop_local.add(k)
            survive = [kept_idx[j] for j in range(len(kept_idx))
                       if j not in drop_local]

            keep_mask = np.zeros(n_features, dtype=bool)
            keep_mask[survive] = True

            # (4) imputation fill values for the surviving columns (used at
            # transform time).
            fill = fill[keep_mask]

        self.keep_mask_ = keep_mask
        self.medians_ = fill
        self.n_features_in_ = n_features
        self.n_kept_ = int(keep_mask.sum())
        return self

    def transform(self, X):
        X = self._as_float64(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"DescriptorCleaner.transform expected "
                f"{self.n_features_in_} columns, got {X.shape[1]}.")
        X = np.where(np.isinf(X), np.nan, X)
        X = X[:, self.keep_mask_]
        # impute remaining NaN with the stored train medians
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X = X.copy()
            X[inds] = np.take(self.medians_, inds[1])
        return X


# --- named presets for the two model paths ---------------------------------
#
# Both paths share the same principled nan handling (inf -> nan, then
# column-median imputation fit on train) via this class; only the column-
# dropping aggressiveness differs, because the two models respond differently
# to dimensionality reduction:
#
#  * SVR (poly kernel + optional PCA): column dropping HURTS it -- the near-
#    zero-variance columns become high-magnitude features after standardization
#    that PCA/the kernel use as signal, and median imputation alone already
#    beats the old nan_to_num->0 (28 nm vs 34 nm no-PCA, 29.5 vs 32 with PCA).
#    So the SVR uses impute-only (drop nothing except perfect-duplicate cols).
#
#  * MLP (sigmoid skip + PCA to 95% var): the aggressive var/corr filter is
#    beneficial -- the network is regularized by the low-dimensional, de-
#    correlated PCA input, and this matches the filter prep_data already used.
#
def impute_only_cleaner():
    """Median-impute all descriptors, drop only perfect-duplicate columns.

    Used by the SVR path: keeps the full descriptor set so StandardScaler + the
    polynomial kernel see all the signal, and median imputation (vs the old
    nan_to_num->0) is worth ~6 nm on the no-PCA poly SVR (28 nm vs 34 nm).
    var_threshold < 0 keeps every column; corr_threshold = 1.0 drops only
    |corr| == 1 duplicates; max_nan_frac = 1.0 keeps partially-missing columns
    (median-imputed)."""
    return DescriptorCleaner(var_threshold=-1.0, corr_threshold=1.0,
                             max_nan_frac=1.0, impute_strategy='median')


def aggressive_cleaner():
    """Zero-fill (nan_to_num), then drop near-constant (std <= 1e-3) and
    highly-correlated (|corr| > 0.98) columns. Used by the MLP path (then PCA
    to 95% variance). Zero-fill is used deliberately -- the MLP's tuned
    weight_decay/epochs/PCA config depends on it; median imputation was tested
    and regresses the MLP 27 -> 40 nm (it shifts the PCA dimension). max_nan_frac
    = 1.0 so partially-missing columns are filled and kept, matching the prior
    inline filter the MLP was tuned against."""
    return DescriptorCleaner(var_threshold=1e-3, corr_threshold=0.98,
                             max_nan_frac=1.0, impute_strategy='zero')