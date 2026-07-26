"""Tests for the grid_search tool: validator caps/whitelist + end-to-end save.

Covers the cost caps the tool enforces for the LLM-driven agent (max 3 values
per hyperparameter, max 6 combinations total), the per-model whitelist, alias
normalization, and the save/manifest contract that makes grid_search drop-in
compatible with train_model -> evaluate_model/run_inference.

Uses a synthetic features.npz (no RDKit/featurize step) so the search path is
fast and focused. Run from a clean CWD under tmp_path so runs/ is isolated.
"""

import os

import numpy as np
import pytest

from chemlagent import tools as T


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def seeded_run(tmp_path, monkeypatch):
    """Create a run_id with a synthetic features.npz + seeded manifest.

    Target is a linear function of the features so SVR/RF/LightGBM can all fit
    it well enough for the asserts (r2 > 0, finite) to be meaningful.
    """
    monkeypatch.chdir(tmp_path)
    run_id = "gridtest"
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, size=(n, 20))
    w = rng.normal(0, 1, size=20)
    y = X @ w + 0.05 * rng.normal(size=n)
    idx = rng.permutation(n)
    n_tr = 160
    Xtr, Xte = X[idx[:n_tr]], X[idx[n_tr:]]
    ytr, yte = y[idx[:n_tr]], y[idx[n_tr:]]

    d = T.run_dir(run_id)  # creates runs/gridtest
    np.savez(os.path.join(d, T._FEATURES),
             X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte)
    T._write_manifest(run_id, {"run_id": run_id, "fp_type": "synthetic"})
    return run_id


# --- validator: caps & whitelist -------------------------------------------

def test_validate_accepts_valid_grids():
    g, n = T._validate_param_grid("random_forest",
                                  {"n_estimators": [100, 300],
                                   "max_depth": [None, 20, 40]})
    assert n == 6
    assert g == {"n_estimators": [100, 300], "max_depth": [None, 20, 40]}


def test_validate_rejects_too_many_combinations():
    # 3 x 3 = 9 > 6
    with pytest.raises(ValueError, match="6 total"):
        T._validate_param_grid("random_forest",
                               {"n_estimators": [100, 200, 400],
                                "max_depth": [None, 20, 40]})


def test_validate_rejects_too_many_values_per_param():
    with pytest.raises(ValueError, match="at most 3 values"):
        T._validate_param_grid("random_forest",
                               {"n_estimators": [10, 20, 30, 40]})


def test_validate_rejects_single_value():
    with pytest.raises(ValueError, match="at least 2 values"):
        T._validate_param_grid("svr", {"C": [10]})


def test_validate_rejects_unknown_param():
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        T._validate_param_grid("random_forest", {"bogus": [1, 2]})


def test_validate_rejects_empty_grid():
    with pytest.raises(ValueError, match="non-empty"):
        T._validate_param_grid("lightgbm", {})


@pytest.mark.parametrize("alias,canonical", [
    ("rf", "random_forest"),
    ("randomforest", "random_forest"),
    ("lgbm", "lightgbm"),
])
def test_alias_normalization(alias, canonical):
    assert T._GRID_ALIASES[alias] == canonical


def test_svr_grid_is_namespaced():
    est, grid = T._build_search_estimator(
        "svr", {"C": [1, 10], "gamma": ["scale", "auto"]})
    assert [s[0] for s in est.steps] == ["clean", "scale", "svr"]
    assert set(grid) == {"svr__C", "svr__gamma"}


@pytest.mark.parametrize("spelling", ["None", "none", "null", "NULL"])
def test_validate_coerces_max_depth_null_strings(spelling):
    # If the LLM emits "None"/"null" as a JSON string (valid JSON, unlike bare
    # Python None which Ollama drops), the validator must coerce it to None so
    # sklearn's RandomForestRegressor gets unlimited depth, not a string.
    g, n = T._validate_param_grid("random_forest",
                                  {"n_estimators": [100, 200],
                                   "max_depth": [spelling, 20]})
    assert n == 4
    assert g["max_depth"] == [None, 20]


def test_rf_lightgbm_grid_names_match_directly():
    est, grid = T._build_search_estimator(
        "random_forest", {"n_estimators": [100, 200], "max_depth": [None, 20]})
    assert set(grid) == {"n_estimators", "max_depth"}
    est2, grid2 = T._build_search_estimator(
        "lightgbm", {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]})
    assert set(grid2) == {"num_leaves", "learning_rate"}


# --- end-to-end: search -> save -> manifest -> evaluate ---------------------

@pytest.mark.parametrize("model_type,grid", [
    ("random_forest", {"n_estimators": [100, 300], "max_depth": [None, 20, 40]}),
    ("lightgbm", {"n_estimators": [100, 300], "num_leaves": [15, 31]}),
    ("svr", {"C": [1, 10, 100], "gamma": ["scale", "auto"]}),
])
def test_grid_search_saves_and_manifests(seeded_run, model_type, grid):
    run_id = seeded_run
    r = T.grid_search(run_id=run_id, model_type=model_type, param_grid=grid)

    assert r["model_type"] in ("random_forest", "lightgbm", "svr")
    assert os.path.exists(r["model_path"])
    assert isinstance(r["best_cv_r2"], float)
    assert isinstance(r["r2"], float)
    assert np.isfinite(r["r2"])
    assert set(r["best_params"]) == set(grid)  # user-facing names, no svr__ prefix

    manifest = T._read_manifest(run_id)
    assert manifest["grid_search"] is True
    assert manifest["model_type"] == r["model_type"]
    assert manifest["best_params"] == r["best_params"]
    assert manifest["n_grid_combinations"] >= 2
    assert manifest["cv_folds"] >= 2


def test_grid_search_result_feeds_evaluate_model(seeded_run):
    # grid_search must save a model.pkl with the same .predict contract as
    # train_model, so evaluate_model works unchanged on the result.
    run_id = seeded_run
    T.grid_search(run_id=run_id, model_type="random_forest",
                  param_grid={"n_estimators": [100, 300],
                              "max_depth": [None, 20, 40]})
    ev = T.evaluate_model(run_id=run_id)
    assert np.isfinite(ev["r2"])
    assert ev["n_test"] > 0
    assert len(ev["predictions"]) == ev["n_test"]


def test_grid_search_rejects_unknown_model_type(seeded_run):
    with pytest.raises(ValueError, match="Unknown model_type"):
        T.grid_search(run_id=seeded_run, model_type="xgboost",
                      param_grid={"n_estimators": [100, 200]})