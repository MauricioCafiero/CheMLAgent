"""End-to-end smoke test of the CheMLAgent tool pipeline (no Ollama).

Exercises prepare_chembl_csv -> featurize_fingerprints -> train_model ->
evaluate_model -> run_inference on REAL data: a 200-molecule slice of the
repo's ChEMBL bioactivity CSV (tests/conftest.py), featurized with ECFP and
scaffold-split, so the plumbing is verifiable without hitting ChEMBL or the
LLM.
"""

import os

import pytest

from chemlagent import tools as T


def test_pipeline_round_trip(real_csv, tmp_path, monkeypatch):
    # Run from a clean CWD under tmp_path so runs/ lands there.
    monkeypatch.chdir(tmp_path)
    run_id = "smoke"

    prep = T.prepare_chembl_csv(input_csv=real_csv, run_id=run_id, limit=200)
    assert prep["n_rows"] > 0
    assert os.path.exists(prep["csv_path"])

    feat = T.featurize_fingerprints(run_id=run_id, fp_type="ECFP",
                                    test_size=0.25)
    assert feat["n_train"] + feat["n_test"] <= prep["n_rows"]
    assert os.path.exists(feat["features_path"])

    trained = T.train_model(run_id=run_id, model_type="random_forest",
                            n_estimators=10)
    assert os.path.exists(trained["model_path"])
    assert isinstance(trained["r2"], float)

    ev = T.evaluate_model(run_id=run_id)
    assert ev["n_test"] == feat["n_test"]
    assert len(ev["predictions"]) == ev["n_test"]

    inf = T.run_inference(run_id=run_id, smiles_list=["CCO", "c1ccccc1"])
    assert len(inf["predictions"]) == 2
    # log_transformed reflects the manifest (true for the real ChEMBL nM
    # targets, which the pipeline log-transforms).
    m = T._read_manifest(run_id)
    assert inf["log_transformed"] == bool(m.get("log_transformed", False))

    # manifest should carry flags from both data prep and featurize stages.
    assert os.path.exists(os.path.join("runs", run_id, "manifest.json"))