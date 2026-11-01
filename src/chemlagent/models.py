from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Kernel, ConstantKernel, RBF,
                                              WhiteKernel)
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np

from chemlagent.descriptor_cleaning import impute_only_cleaner

# Broadcast print flag: the agent sets chemlagent.models.print_flag = <args.print>
# at startup (see agent.py, mirroring src/agent_template.py). The per-model
# "Testing with ..." banners always print; the evaluate_model result table is
# debug output and is gated here. Default False = debug-silent unless --print.
print_flag = False

def random_forest_regression(X_train, y_train, X_test, y_test, n_estimators=100,
                             random_state=42, n_jobs=None):
    """
    Train and evaluate a Random Forest Regressor.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like
        Testing features
    y_test : array-like
        Testing target values
    
    Returns:
    --------
    r2 : float
        R^2 score on the test set
    r2_train : float
        R^2 score on the training set
    """
    print(f'Testing with Random Forest Regressor '
          f'(n_estimators={n_estimators}, random_state={random_state}).')
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               random_state=random_state, n_jobs=n_jobs)
    
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)
    
    return r2, r2_train, rf

def lightgbm_regression(X_train, y_train, X_test, y_test, n_estimators=100,
                        random_state=42, n_jobs=1):
    """
    Train and evaluate a LightGBM Regressor.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like
        Testing features
    y_test : array-like
        Testing target values
    n_jobs : int, default 1
        Number of OpenMP threads for LightGBM. Defaults to 1 because with
        n_jobs=-1 LightGBM segfaults (exit 139) when run in-process alongside
        torch/MPS on this platform; the OMP_NUM_THREADS env var does not
        override LightGBM's own thread count.

    Returns:
    --------
    r2 : float
        R^2 score on the test set
    r2_train : float
        R^2 score on the training set
    """
    print(f'Testing with LightGBM Regressor '
          f'(n_estimators={n_estimators}, random_state={random_state}, '
          f'n_jobs={n_jobs}).')
    lgbm = LGBMRegressor(n_estimators=n_estimators, random_state=random_state,
                        verbose=-1, n_jobs=n_jobs)
    
    lgbm.fit(X_train, y_train)
    
    y_train_pred = lgbm.predict(X_train)
    y_pred = lgbm.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)

    return r2, r2_train, lgbm

def svr_regression(X_train, y_train, X_test, y_test, pca_n_components=None,
                   C=150, epsilon=0.1, coef0=7, degree=2, gamma='scale'):
    """
    Train and evaluate a Support Vector Regressor with a second-degree
    polynomial kernel, matching the literature hyperparameters: C=150
    (regularization / margin), epsilon=0.1 (the epsilon-tube, the "support
    vector coefficient e"), coef0=7 (the high-degree-term coefficient), and
    degree=2. gamma is left at its default 'scale'.

    SVR is a kernel / distance-based model, so -- unlike the tree-based
    RandomForest and LightGBM above, which run on the raw Mordred descriptors
    -- the features must be cleaned (Mordred emits NaN/inf for descriptors it
    cannot compute) and standardized before the kernel is computed, otherwise
    the few huge-magnitude descriptor columns dominate the polynomial kernel.

    Cleaning uses `descriptor_cleaning.impute_only_cleaner` (a DescriptorCleaner
    fit on the TRAIN split only): inf -> nan, then column-median imputation,
    dropping only perfect-duplicate (|corr| == 1) columns. Median imputation
    is both more principled and empirically better than the old nan_to_num -> 0
    (no-PCA SVR: 28 nm imputed vs 34 nm zero-filled). Column dropping beyond
    perfect duplicates is deliberately NOT applied here: the near-zero-variance
    Mordred columns become high-magnitude features after standardization that
    the polynomial kernel uses as signal, and dropping them regresses the SVR
    to ~40-50 nm (verified). The MLP path uses the aggressive cleaner instead --
    see descriptor_cleaning.aggressive_cleaner -- because the MLP benefits from
    the de-correlated low-dimensional PCA input whereas the SVR does not.

    To keep the same call/return contract as the other two regressors (so
    evaluate_model's model.predict(X_new) works unchanged on the raw Mordred
    features of new molecules), the returned model is a sklearn Pipeline that
    applies the identical clean -> standardize -> (PCA) -> SVR transform at
    predict time, fit on the TRAIN split only (no leakage).

    Parameters:
    -----------
    X_train : array-like
        Training features (raw Mordred descriptors)
    y_train : array-like
        Training target values
    X_test : array-like
        Testing features (raw Mordred descriptors)
    y_test : array-like
        Testing target values
    pca_n_components : int, float or None, default None
        If None (the production choice), run on the full imputed+standardized
        descriptor set with NO PCA -- this is the best config on the scaffold
        split (~28 nm, better than any PCA count tested). If set, apply PCA
        (int = fixed component count, float = variance fraction) after
        standardization; retained only for experimentation.

    Returns:
    --------
    r2 : float
        R^2 score on the test set
    r2_train : float
        R^2 score on the training set
    svr : sklearn.pipeline.Pipeline
        Fitted (clean -> StandardScaler -> [PCA] -> SVR) pipeline; .predict
        applies the same transform to new molecules.
    """
    tag = (f'PCA->{pca_n_components}, ' if pca_n_components is not None else 'no PCA, ')
    print(f'Testing with SVR ({tag}impute-only clean, '
          f'poly kernel, degree={degree}, C={C}, epsilon={epsilon}, '
          f'coef0={coef0}, gamma={gamma}).')

    steps = [
        ('clean', impute_only_cleaner()),
        ('scale', StandardScaler()),
    ]
    if pca_n_components is not None:
        # PCA is sensitive to feature scale, so it goes AFTER StandardScaler.
        from sklearn.decomposition import PCA
        steps.append(('pca', PCA(n_components=pca_n_components,
                                 random_state=132)))
    steps.append(('svr', SVR(kernel='poly', degree=degree, C=C,
                             epsilon=epsilon, coef0=coef0, gamma=gamma)))

    svr = Pipeline(steps)

    svr.fit(X_train, y_train)

    y_train_pred = svr.predict(X_train)
    y_pred = svr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)

    return r2, r2_train, svr

class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) similarity kernel for fingerprint features:
    k(x, y) = x.y / (|x|^2 + |y|^2 - x.y). Exact Tanimoto on binary bits, the
    dot-product generalization on counts. Parameter-free and positive
    semi-definite; scale it with a ConstantKernel."""

    def __init__(self):  # sklearn kernels need an explicit (here empty) signature
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=np.float64)
        Y = X if Y is None else np.asarray(Y, dtype=np.float64)
        xy = X @ Y.T
        denom = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - xy
        K = np.divide(xy, denom, out=np.zeros_like(xy), where=denom > 0)
        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        return np.ones(np.asarray(X).shape[0])

    def is_stationary(self):
        return False


def _looks_like_fingerprint(X):
    """True if X is all finite, non-negative integers (fingerprint bits/counts)."""
    X = np.asarray(X, dtype=np.float64)
    return bool(np.all(np.isfinite(X)) and np.all(X >= 0)
                and np.array_equal(X, np.round(X)))


def gaussian_process_regression(X_train, y_train, X_test, y_test, kernel='auto',
                                random_state=42):
    """
    Train and evaluate a Gaussian process regressor, which also gives a
    per-molecule predictive uncertainty (std).

    Two kernels, chosen to match the featurization:
      * 'tanimoto' -- ConstantKernel * Tanimoto + WhiteKernel on fingerprint
        bits/counts (ECFP, Atom_Pair, MACCS, PubChem, ...). No scaling.
      * 'rbf' -- ConstantKernel * RBF + WhiteKernel on continuous descriptors
        (RDKit_2D, Mordred, EState, ...), after the same impute-only clean +
        StandardScaler step the SVR uses (fit on TRAIN only).
      * 'auto' (default) -- 'tanimoto' if the features are all non-negative
        integers, else 'rbf'.
    Amplitude, noise (and the RBF length scale) are fit by maximizing the
    marginal likelihood; targets are normalized internally (normalize_y).

    An exact GP costs O(n^3) in the training-set size: fine up to several
    thousand molecules (CHEMBL220's ~4800 train fit in seconds to a minute),
    slow beyond ~10-15k.

    Parameters:
    -----------
    X_train, y_train, X_test, y_test : array-like
        Train/test features and targets.
    kernel : {'auto', 'tanimoto', 'rbf'}, default 'auto'
    random_state : int, default 42

    Returns:
    --------
    r2 : float
        R^2 score on the test set
    r2_train : float
        R^2 score on the training set
    gp : sklearn.pipeline.Pipeline
        Fitted ([clean -> scale ->] GP) pipeline. .predict(X) gives the mean;
        .predict(X, return_std=True) also returns the predictive std.
    """
    if kernel == 'auto':
        kernel = 'tanimoto' if _looks_like_fingerprint(X_train) else 'rbf'
    print(f'Testing with Gaussian Process Regressor (kernel={kernel}, '
          f'random_state={random_state}).')
    if kernel == 'tanimoto':
        k = (ConstantKernel(1.0, (1e-4, 1e2)) * TanimotoKernel()
             + WhiteKernel(0.3, (1e-5, 1e1)))
        steps = []
    elif kernel == 'rbf':
        n_features = np.asarray(X_train).shape[1]
        k = (ConstantKernel(1.0, (1e-4, 1e2))
             * RBF(np.sqrt(n_features), (1e-2, 1e4))
             + WhiteKernel(0.3, (1e-5, 1e1)))
        steps = [('clean', impute_only_cleaner()), ('scale', StandardScaler())]
    else:
        raise ValueError(f"kernel must be 'auto', 'tanimoto' or 'rbf', got {kernel!r}")
    steps.append(('gp', GaussianProcessRegressor(k, normalize_y=True,
                                                 random_state=random_state)))
    gp = Pipeline(steps)
    with warnings.catch_warnings():
        # Hyperparameters hitting a bound is informative, not an error.
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp.fit(X_train, y_train)

    r2 = r2_score(y_test, gp.predict(X_test))
    r2_train = r2_score(y_train, gp.predict(X_train))
    return r2, r2_train, gp


def random_forest_classifier(X_train, y_train, X_test, y_test, n_estimators=100,
                             random_state=42):
    """
    Train and evaluate a Random Forest Classifier (classifier counterpart of
    random_forest_regression; same signature/return contract, accuracy instead
    of R^2). y values are int class indices.

    Returns:
      accuracy, accuracy_train (floats), fitted model
    """
    print(f'Testing with Random Forest Classifier '
          f'(n_estimators={n_estimators}, random_state={random_state}).')
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                random_state=random_state)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    return (accuracy_score(y_test, y_pred),
            accuracy_score(y_train, y_train_pred), rf)

def lightgbm_classifier(X_train, y_train, X_test, y_test, n_estimators=100,
                        random_state=42, n_jobs=1):
    """
    Train and evaluate a LightGBM Classifier. n_jobs defaults to 1 for the
    same libomp/MPS segfault reason as lightgbm_regression (see its docstring).

    Returns:
      accuracy, accuracy_train (floats), fitted model
    """
    print(f'Testing with LightGBM Classifier '
          f'(n_estimators={n_estimators}, random_state={random_state}, '
          f'n_jobs={n_jobs}).')
    lgbm = LGBMClassifier(n_estimators=n_estimators, random_state=random_state,
                          verbose=-1, n_jobs=n_jobs)
    lgbm.fit(X_train, y_train)
    y_train_pred = lgbm.predict(X_train)
    y_pred = lgbm.predict(X_test)
    return (accuracy_score(y_test, y_pred),
            accuracy_score(y_train, y_train_pred), lgbm)

def svc_classifier(X_train, y_train, X_test, y_test, pca_n_components=None,
                   C=2.0, degree=3, coef0=2.0, gamma='scale'):
    """
    Train and evaluate a Support Vector Classifier with a third-degree
    polynomial kernel -- the classifier counterpart of svr_regression. The
    kernel defaults (C=2, degree=3, coef0=2) follow the poly-kernel SVC in
    CafChemClassifiers.ridge_svm_methods. Like svr_regression, the returned
    model is a sklearn Pipeline (impute-only clean -> StandardScaler -> [PCA]
    -> SVC) fit on the TRAIN split only, so .predict works on raw Mordred
    features of new molecules.

    Returns:
      accuracy, accuracy_train (floats), fitted Pipeline
    """
    tag = (f'PCA->{pca_n_components}, ' if pca_n_components is not None
           else 'no PCA, ')
    print(f'Testing with SVC ({tag}impute-only clean, poly kernel, '
          f'degree={degree}, C={C}, coef0={coef0}, gamma={gamma}).')
    steps = [
        ('clean', impute_only_cleaner()),
        ('scale', StandardScaler()),
    ]
    if pca_n_components is not None:
        from sklearn.decomposition import PCA
        steps.append(('pca', PCA(n_components=pca_n_components,
                                 random_state=132)))
    steps.append(('svc', SVC(kernel='poly', degree=degree, C=C, coef0=coef0,
                             gamma=gamma)))
    svc = Pipeline(steps)
    svc.fit(X_train, y_train)
    y_train_pred = svc.predict(X_train)
    y_pred = svc.predict(X_test)
    return (accuracy_score(y_test, y_pred),
            accuracy_score(y_train, y_train_pred), svc)

def evaluate_model(model, fps, new_smiles_list, new_targets):
    '''
    Evaluate a trained model on a new (external) dataset, print the per-
    molecule comparison table, and RETURN the predictions + metrics so the
    caller (azo_model.run_utils.save_run) can persist them.

    Parameters:
    -----------
    model : object
        Trained model with .predict(X) -- for SVR this is the fitted sklearn
        Pipeline, so raw Mordred features of new molecules are cleaned/scaled
        inside .predict exactly as at train time.
    fps : object
        Fingerprint generator (exposes .transform(smiles_list))
    new_smiles_list : list
        List of SMILES strings for new molecules
    new_targets : array-like
        True target values for new molecules

    Returns:
    --------
    result : dict with keys
        r2 (float), mae (float), predictions (np.ndarray), truths (np.ndarray)
    '''
    X_new = fps.transform(new_smiles_list)

    y_new_pred = model.predict(X_new)
    truths = np.asarray(new_targets, dtype=float).reshape(-1)
    r2_new = r2_score(truths, y_new_pred)
    mae_new = mean_absolute_error(truths, y_new_pred)
    if print_flag:
        print(f'R^2 score (new dataset): {r2_new}')
        print(f'MAE (new dataset): {mae_new:.2f} nm')

        # print predictions for new molecules with a header and columns of values
        print(f'Predictions for new molecules:')
        smiles_width = max(len(smi) for smi in new_smiles_list) + 2
        print(f'{"SMILES":<{smiles_width}} {"Predicted Lmax":<20} {"Actual Lmax":<20} {"Difference":<15}')
        differences = []
        for i, smiles in enumerate(new_smiles_list):
            diff = abs(y_new_pred[i] - truths[i])
            differences.append(diff)
            print(f'{smiles:<{smiles_width}} {y_new_pred[i]:<20.1f} {truths[i]:<20.1f} {diff:<15.1f}')
        avg_diff = sum(differences) / len(differences)
        print(f'{"Average":<{smiles_width}} {"":20} {"":20} {avg_diff:<15.1f}')

    return {'r2': r2_new, 'mae': mae_new,
            'predictions': np.asarray(y_new_pred).reshape(-1), 'truths': truths}
