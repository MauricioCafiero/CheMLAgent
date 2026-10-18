"""Classifier tests on REAL data: every model type, not just the fast ones.

prepare_classes -> featurize -> train -> evaluate -> inference on a small
slice of the repo's real ChEMBL bioactivity CSV (tests/conftest.py),
asserting the classification metric set (accuracy / MCC / confusion matrix,
range labels) instead of R^2. The RF round trip has its own fresh run; the
remaining model types (SVR, LightGBM grid search, MLP, mlp_scan) share a
featurized run via the cls_models fixture. Chemprop classifiers live in
tests/test_chemprop.py. Skipped if the CSV is missing so the suite still
runs in a trimmed checkout.
"""

import os

import numpy as np
import pytest

from chemlagent import tools as T

_SMIS = ["CCOc1ccc(cc1)C(=O)NC", "O=C(O)c1ccccc1OC(C)C"]


def test_classification_round_trip_real_data(real_csv, tmp_path,
                                             monkeypatch):
    # Run from a clean CWD under tmp_path so runs/ lands there.
    monkeypatch.chdir(tmp_path)
    run_id = "cls_real"

    prep = T.prepare_chembl_csv(input_csv=real_csv, run_id=run_id, limit=200)
    assert prep["n_rows"] > 0

    classes = T.prepare_classes(run_id, num_classes=3)
    assert classes["num_classes"] == 3
    assert len(classes["class_labels"]) == 3
    # Equal-count bands: every class gets a real share of the dataset.
    assert sum(classes["class_counts"]) == prep["n_rows"]
    assert min(classes["class_counts"]) > 0

    feat = T.featurize_fingerprints(run_id=run_id, fp_type="ECFP",
                                    test_size=0.2)
    assert feat["task"] == "classification"
    feats = np.load(feat["features_path"])
    # y carries int class indices, not float targets.
    assert set(np.unique(feats["y_train"])) <= {0, 1, 2}

    trained = T.train_model(run_id=run_id, model_type="random_forest",
                            n_estimators=40)
    assert 0.0 <= trained["accuracy"] <= 1.0
    assert 0.0 <= trained["accuracy_train"] <= 1.0

    ev = T.evaluate_model(run_id=run_id, model_type="random_forest")
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert -1.0 <= ev["mcc"] <= 1.0
    cm = ev["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    # Row-normalized confusion matrix: every row sums to ~1.
    assert np.allclose(np.asarray(cm).sum(axis=1), 1.0)
    # Predictions come back as int classes AND range labels.
    assert all(isinstance(p, int) for p in ev["predictions"])
    assert len(ev["prediction_labels"]) == ev["n_test"]
    assert all(" < " in lab for lab in ev["prediction_labels"])

    inf = T.run_inference(run_id=run_id, smiles_list=_SMIS)
    assert len(inf["predicted_class"]) == 2
    assert len(inf["predicted_labels"]) == 2
    assert all(c in (0, 1, 2) for c in inf["predicted_class"])


@pytest.fixture(scope="module")
def cls_models(tmp_path_factory, real_csv):
    """A classification run (3 classes, ECFP) shared by the remaining
    classifier tests, from a per-module tmp CWD."""
    d = tmp_path_factory.mktemp("cls_models")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        T.prepare_chembl_csv(input_csv=real_csv, run_id="cls_models",
                             limit=200)
        T.prepare_classes("cls_models", num_classes=3)
        T.featurize_fingerprints(run_id="cls_models", fp_type="ECFP",
                                 test_size=0.2)
        yield "cls_models"
    finally:
        os.chdir(cwd)


def test_svr_classifier_round_trip(cls_models):
    run_id = cls_models
    tr = T.train_model(run_id=run_id, model_type="svr")
    assert 0.0 <= tr["accuracy"] <= 1.0

    ev = T.evaluate_model(run_id=run_id, model_type="svr")
    assert ev["model_type"] == "svr"
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert -1.0 <= ev["mcc"] <= 1.0
    cm = ev["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)

    inf = T.run_inference(run_id=run_id, smiles_list=_SMIS)
    assert all(c in (0, 1, 2) for c in inf["predicted_class"])
    assert all(" < " in lab for lab in inf["predicted_labels"])


def test_grid_search_lightgbm_classifier(cls_models):
    run_id = cls_models
    r = T.grid_search(run_id=run_id, model_type="lightgbm",
                      param_grid={"n_estimators": [30, 60]})
    assert 0.0 <= r["best_cv_accuracy"] <= 1.0
    assert 0.0 <= r["accuracy"] <= 1.0
    assert set(r["best_params"]) == {"n_estimators"}
    assert os.path.exists(r["model_path"])

    m = T._read_manifest(run_id)
    assert m["models"]["lightgbm"]["task"] == "classification"
    assert m["grid_search"] is True

    # Best grid model must be evaluable under the standard contract.
    ev = T.evaluate_model(run_id=run_id, model_type="lightgbm")
    assert 0.0 <= ev["accuracy"] <= 1.0


def test_mlp_classifier_round_trip(cls_models):
    run_id = cls_models
    tr = T.train_mlp(run_id=run_id, epochs=15, patience=4)
    assert 0.0 <= tr["accuracy"] <= 1.0
    assert 0.0 <= tr["accuracy_train"] <= 1.0

    m = T._read_manifest(run_id)
    arch = m["models"]["mlp"]["mlp_arch"]
    assert arch["classifier_flag"] is True
    assert arch["num_classes"] == 3

    ev = T.evaluate_model(run_id=run_id, model_type="mlp")
    assert ev["model_type"] == "mlp"
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert -1.0 <= ev["mcc"] <= 1.0
    assert len(ev["confusion_matrix"]) == 3

    inf = T.run_inference_mlp(run_id=run_id, smiles_list=_SMIS)
    assert all(c in (0, 1, 2) for c in inf["predicted_class"])
    assert all(" < " in lab for lab in inf["predicted_labels"])


def test_mlp_scan_classifier(cls_models):
    run_id = cls_models
    r = T.mlp_scan(run_id=run_id, param_grid={"neurons": [32, 64]},
                   epochs=8, patience=3, cv_folds=3)
    assert r["n_combinations"] == 2
    assert 0.0 <= r["best_cv_accuracy"] <= 1.0
    assert 0.0 <= r["accuracy"] <= 1.0

    # Scored by accuracy, and the manifest records the classifier arch.
    m = T._read_manifest(run_id)
    assert m["models"]["mlp"]["mlp_scan"] is True
    assert m["models"]["mlp"]["task"] == "classification"
    assert m["models"]["mlp"]["mlp_arch"]["classifier_flag"] is True