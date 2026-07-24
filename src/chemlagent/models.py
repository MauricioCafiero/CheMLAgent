from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

from chemlagent.descriptor_cleaning import impute_only_cleaner

# Broadcast print flag: the agent sets chemlagent.models.print_flag = <args.print>
# at startup (see agent.py, mirroring src/agent_template.py). The per-model
# "Testing with ..." banners always print; the evaluate_model result table is
# debug output and is gated here. Default False = debug-silent unless --print.
print_flag = False

def random_forest_regression(X_train, y_train, X_test, y_test, n_estimators=100,
                             random_state=42):
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
                               random_state=random_state)
    
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
