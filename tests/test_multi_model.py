"""Tests for multi-model runs: several models trained on one run stay
independently reachable via evaluate_model(run_id, model_type=...).

This is the regression test for the bug where training a second model
overwrote the manifest's single ``model_type`` flag, stranding the first
model: e.g. train LightGBM after an MLP and ``evaluate_model`` could no
longer score the LightGBM model even though its file was still on disk.

Uses the REAL pipeline (tests/conftest.py): prepare_chembl_csv +
featurize_fingerprints ECFP on a slice of the repo's ChEMBL CSV, featurized
once per module; the module-scoped seeded_run fixture switches CWD to a
per-module tmp dir so runs/ is isolated. Focus stays on the manifest +
dispatch logic.
"""

import os

import numpy as np
import pytest

from chemlagent import tools as T


def test_two_sklearn_models_coexist_and_evaluate(seeded_run):
    run_id = seeded_run
    # Train RF first, then LightGBM. LightGBM becomes the active model, but RF
    # must survive on disk and stay evaluable.
    T.train_model(run_id=run_id, model_type="random_forest", n_estimators=10)
    T.train_model(run_id=run_id, model_type="lightgbm", n_estimators=10)

    manifest = T._read_manifest(run_id)
    assert manifest["model_type"] == "lightgbm"  # active = most recent
    assert "models" in manifest
    assert {"random_forest", "lightgbm"} <= set(manifest["models"])

    # Distinct per-type files, both present.
    rf_path = manifest["models"]["random_forest"]["model_path"]
    lgbm_path = manifest["models"]["lightgbm"]["model_path"]
    assert rf_path != lgbm_path
    assert os.path.exists(rf_path) and os.path.exists(lgbm_path)
    assert rf_path.endswith("model_random_forest.pkl")
    assert lgbm_path.endswith("model_lightgbm.pkl")

    # Each is independently evaluable; the active default matches LightGBM.
    ev_rf = T.evaluate_model(run_id=run_id, model_type="random_forest")
    ev_lgbm = T.evaluate_model(run_id=run_id, model_type="lightgbm")
    ev_active = T.evaluate_model(run_id=run_id)

    assert ev_rf["model_type"] == "random_forest"
    assert ev_lgbm["model_type"] == "lightgbm"
    assert ev_active["model_type"] == "lightgbm"
    for ev in (ev_rf, ev_lgbm, ev_active):
        assert np.isfinite(ev["r2"])
        assert np.isfinite(ev["mae"])
        assert ev["n_test"] > 0
        assert len(ev["predictions"]) == ev["n_test"]


def test_evaluate_alias_and_missing_type_error(seeded_run):
    run_id = seeded_run
    T.train_model(run_id=run_id, model_type="rf", n_estimators=10)
    # Alias normalizes to the canonical type and evaluates.
    ev = T.evaluate_model(run_id=run_id, model_type="rf")
    assert ev["model_type"] == "random_forest"
    assert np.isfinite(ev["r2"])

    # Requesting a type never trained on the run raises a helpful error that
    # names what IS available.
    with pytest.raises(ValueError, match="Available model type"):
        T.evaluate_model(run_id=run_id, model_type="mlp")


def test_run_inference_rejects_non_sklearn_type(seeded_run):
    run_id = seeded_run
    T.train_model(run_id=run_id, model_type="random_forest", n_estimators=10)
    # run_inference is sklearn-only; an MLP/chemprop request must redirect,
    # before it tries to load a featurizer that the synthetic run lacks.
    with pytest.raises(ValueError, match="run_inference_mlp"):
        T.run_inference(run_id=run_id, smiles_list=["CCO"],
                        model_type="mlp")


def test_relocate_keeps_all_trained_models(seeded_run):
    from chemlagent import products as P

    run_id = seeded_run
    T.train_model(run_id=run_id, model_type="random_forest", n_estimators=10)
    T.train_model(run_id=run_id, model_type="lightgbm", n_estimators=10)

    # Both per-type model files + the manifest must survive relocation (the
    # non-active RF model must NOT be discarded as an "intermediate").
    P.relocate_all_runs(runs_root="runs")
    rw = os.path.join(P.RECENT_WORK_DIR, run_id)
    assert os.path.exists(os.path.join(rw, "model_random_forest.pkl"))
    assert os.path.exists(os.path.join(rw, "model_lightgbm.pkl"))
    assert os.path.exists(os.path.join(rw, "manifest.json"))
    assert not os.path.exists(os.path.join("runs", run_id))  # intermediates gone

    # The relocated manifest's per-type model_path fields are rewritten to point
    # at the recent_work/ copies, so a later reload resolves to real files.
    import json
    with open(os.path.join(rw, "manifest.json")) as fh:
        m = json.load(fh)
    for rec in m["models"].values():
        assert rec["model_path"].startswith(
            os.path.join(P.RECENT_WORK_DIR, run_id))
        assert os.path.exists(rec["model_path"])

def test_gp_trains_evaluates_and_gives_interval(seeded_run, real_csv):
    import pickle
    import pandas as pd
    from chemlagent.reload import LoadedModel

    # Own run (same real-data slice, same tmp CWD as seeded_run): the shared
    # 'real' run is moved out of runs/ by test_relocate_keeps_all_trained_models.
    run_id = "gp_run"
    T.prepare_chembl_csv(input_csv=real_csv, run_id=run_id, limit=200)
    T.featurize_fingerprints(run_id=run_id, fp_type="ECFP")
    out = T.train_model(run_id=run_id, model_type="gaussian_process")
    assert out["model_type"] == "gp"
    assert out["model_path"].endswith("model_gp.pkl")
    assert np.isfinite(out["r2"])

    # ECFP bits -> kernel='auto' must pick the Tanimoto kernel.
    with open(out["model_path"], "rb") as fh:
        gp = pickle.load(fh)
    assert "Tanimoto" in str(gp[-1].kernel_)

    ev = T.evaluate_model(run_id=run_id, model_type="gp")
    assert np.isfinite(ev["r2"]) and len(ev["predictions"]) == ev["n_test"]

    smiles = ["CCO", "c1ccccc1O", "CN1CCC[C@H]1c1cccnc1"]
    inf = T.run_inference(run_id=run_id, smiles_list=smiles, model_type="gp")
    lo, pred, hi = (np.array(inf[k]) for k in ("lower95", "predictions", "upper95"))
    assert len(pred) == len(smiles)
    assert np.all(lo < pred) and np.all(pred < hi)
    csv = pd.read_csv(inf["predictions_csv"])
    assert {"lower95", "upper95"} <= set(csv.columns)

    # Quit-time relocation must keep model_gp.pkl + featurizer.pkl, and the
    # deterministic reload from recent_work/ must predict the same values.
    from chemlagent import products as P
    P.relocate_all_runs(runs_root="runs")
    lm = LoadedModel.load(run_id, model_type="gp")
    np.testing.assert_allclose(lm.predict(smiles), pred, rtol=1e-6)
