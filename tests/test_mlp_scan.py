"""Tests for the ``mlp_scan`` tool (5-fold-CV MLP hyperparameter scan).

Mirrors tests/test_grid_search.py and test_multi_model.py: the REAL pipeline
(tests/conftest.py) prepares and featurizes a slice of the repo's ChEMBL CSV
(ECFP, Murcko split) once per module; the module-scoped seeded_run fixture
switches CWD to a per-module tmp dir so runs/ is isolated. The scan caps
(<=3 values/param, <=6 combos) and the save/evaluate plumbing are the focus;
neural-net convergence is not asserted beyond finite r2.
"""

import os

import numpy as np
import pytest

from chemlagent import tools as T


# Small/fast training kwargs so the CV fits finish quickly in tests.
_FAST = {"epochs": 80, "patience": 20}


def test_mlp_scan_saves_best_and_evaluable(seeded_run):
    run_id = seeded_run
    r = T.mlp_scan(run_id=run_id,
                  param_grid={"neurons": [64, 250], "lr": [2e-3, 8e-3]},
                  **_FAST)
    assert r["run_id"] == run_id
    assert r["model_type"] == "mlp"
    assert r["n_combinations"] == 4  # 2 x 2; other params fixed at defaults
    assert np.isfinite(r["best_cv_r2"])
    assert np.isfinite(r["r2"])
    assert os.path.exists(r["model_path"])
    assert os.path.exists(os.path.join(T.run_dir(run_id), T._MLP_PREP))

    # The run manifest records the mlp as the active model + a per-type entry.
    m = T._read_manifest(run_id)
    assert m["model_type"] == "mlp"
    assert "mlp" in m["models"]
    assert m["models"]["mlp"]["mlp_arch"]["neurons"] in (64, 250)
    assert m["models"]["mlp"]["mlp_arch"]["skip_connection"] is True
    assert m["models"]["mlp"].get("mlp_scan") is True
    assert m["models"]["mlp"]["cv_folds"] >= 2

    # evaluate_model(run_id, 'mlp') must score the saved model on the test split.
    ev = T.evaluate_model(run_id=run_id, model_type="mlp")
    assert ev["model_type"] == "mlp"
    assert np.isfinite(ev["r2"])
    assert ev["n_test"] > 0
    assert len(ev["predictions"]) == ev["n_test"]


def test_caps_reject_oversized_grid(seeded_run):
    # 3 x 3 = 9 combinations > the 6-combo cap.
    with pytest.raises(ValueError, match="combinations"):
        T.mlp_scan(run_id=seeded_run,
                   param_grid={"neurons": [64, 250, 500],
                               "lr": [2e-3, 8e-3, 2e-2]}, **_FAST)


def test_unknown_param_rejected(seeded_run):
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        T.mlp_scan(run_id=seeded_run, param_grid={"dropout": [0.1, 0.3]},
                   **_FAST)


def test_too_few_values_rejected(seeded_run):
    with pytest.raises(ValueError, match="at least 2"):
        T.mlp_scan(run_id=seeded_run, param_grid={"lr": [2e-3]}, **_FAST)


def test_too_many_values_per_param_rejected(seeded_run):
    with pytest.raises(ValueError, match="at most 3 values"):
        T.mlp_scan(run_id=seeded_run,
                   param_grid={"lr": [1e-3, 2e-3, 5e-3, 1e-2]}, **_FAST)


def test_missing_features_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "nofeat"
    T.run_dir(run_id)  # dir exists but no features.npz
    T._write_manifest(run_id, {"run_id": run_id})
    with pytest.raises(FileNotFoundError, match="featurize_fingerprints"):
        T.mlp_scan(run_id=run_id, param_grid={"lr": [1e-3, 2e-3]}, **_FAST)


def test_partial_grid_uses_defaults(seeded_run):
    """Scanning a subset leaves the unscanned params at the tuned defaults, so
    a 2-value single-param grid is a 2-combo scan and best_params carries the
    fixed defaults for the params the caller did not mention."""
    run_id = seeded_run
    r = T.mlp_scan(run_id=run_id, param_grid={"lr": [2e-3, 8e-3]}, **_FAST)
    assert r["n_combinations"] == 2
    for k in ("neurons", "num_hidden_layers", "weight_decay", "pca_var"):
        assert k in r["best_params"]
    # The unscanned defaults are carried through unchanged.
    assert r["best_params"]["neurons"] == T._MLP_SCAN_DEFAULTS["neurons"]
    assert r["best_params"]["pca_var"] == T._MLP_SCAN_DEFAULTS["pca_var"]


def test_all_diverged_raises(seeded_run, monkeypatch):
    """If every combination's CV folds diverge (SGD loss -> NaN), the call must
    raise with a hint to lower lr / weight_decay. Forced deterministically by
    stubbing _mlp_train_once so no real training runs (avoids a flaky test that
    depends on a particular lr actually diverging)."""
    def fake_train_once(*args, **kwargs):
        return {"model": None, "prep": None, "val_score": float("nan"),
                "train_score": float("nan"), "input_dims": None,
                "epochs_run": 0, "diverged": True}
    monkeypatch.setattr(T, "_mlp_train_once", fake_train_once)
    with pytest.raises(ValueError, match="diverged"):
        T.mlp_scan(run_id=seeded_run, param_grid={"lr": [2e-3, 8e-3]},
                   epochs=10)


def test_best_combo_refit_model_matches_manifest_arch(seeded_run):
    """The saved model's architecture (mlp_arch.input_dims) must match the PCA
    width produced by the winning combo's refit, so _evaluate_mlp can rebuild
    and load the state_dict without a shape mismatch."""
    run_id = seeded_run
    r = T.mlp_scan(run_id=run_id,
                  param_grid={"neurons": [64, 250], "pca_var": [0.9, 0.99]},
                  **_FAST)
    m = T._read_manifest(run_id)
    arch = m["models"]["mlp"]["mlp_arch"]
    # input_dims is the post-PCA feature width of the refit; must be a positive
    # int and consistent with the saved prep stats (n PCA components).
    assert isinstance(arch["input_dims"], int) and arch["input_dims"] > 0
    prep = np.load(os.path.join(T.run_dir(run_id), T._MLP_PREP))
    if bool(prep["has_pca"]):
        assert arch["input_dims"] == prep["pca_components"].shape[0]