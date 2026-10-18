"""Chemprop end-to-end tests on REAL data.

Chemprop is the slow model in the repo (graph featurization + MPNN), but slow
does not mean untested: these runs keep it cheap (1 epoch, batch_size=16, CPU
accelerator, 200-molecule slice of the repo's ChEMBL CSV) so the pytest suite
covers both paths -- the default regression MPNN and the
MulticlassClassificationFFN classifier -- through train_chemprop ->
evaluate_model -> run_inference_chemprop. Skipped if the CSV is missing.
"""

import os

import numpy as np
import pytest

from chemlagent import tools as T

_SMIS = ["CCOc1ccc(cc1)C(=O)NC", "O=C(O)c1ccccc1OC(C)C"]


@pytest.fixture(scope="module")
def reg_run(tmp_path_factory, real_csv):
    """Regression run: only prepare_chembl_csv is needed; chemprop does its
    own graph featurization + scaffold split."""
    d = tmp_path_factory.mktemp("chem_reg")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        prep = T.prepare_chembl_csv(input_csv=real_csv, run_id="chem_reg",
                                    limit=200)
        assert prep["n_rows"] > 0
        # train_chemprop loads the run's features.npz, so featurize first.
        T.featurize_fingerprints(run_id="chem_reg", fp_type="ECFP",
                                 test_size=0.2)
        yield "chem_reg"
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def cls_run(tmp_path_factory, real_csv):
    """Classification run (3 classes) for the chemprop classifier path."""
    d = tmp_path_factory.mktemp("chem_cls")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        prep = T.prepare_chembl_csv(input_csv=real_csv, run_id="chem_cls",
                                    limit=200)
        assert prep["n_rows"] > 0
        classes = T.prepare_classes("chem_cls", num_classes=3)
        assert classes["num_classes"] == 3
        # train_chemprop loads the run's features.npz, so featurize first.
        feat = T.featurize_fingerprints(run_id="chem_cls", fp_type="ECFP",
                                        test_size=0.2)
        assert feat["task"] == "classification"
        yield "chem_cls"
    finally:
        os.chdir(cwd)


def test_chemprop_regression_round_trip(reg_run):
    run_id = reg_run
    tr = T.train_chemprop(run_id=run_id, epochs=1, batch_size=16,
                          accelerator="cpu")
    # 1 epoch is not enough for a good fit; only require finite R^2.
    assert np.isfinite(tr["r2"])
    assert tr["n_train"] > 0 and tr["n_test"] > 0

    ev = T.evaluate_model(run_id=run_id, model_type="chemprop")
    # Chemprop evaluation reports the STORED metrics from training (no fresh
    # per-molecule predictions -- those come from run_inference_chemprop).
    assert ev["model_type"] == "chemprop"
    assert np.isfinite(ev["r2"])
    assert np.isfinite(ev["mae"])
    assert ev["n_test"] == tr["n_test"]

    inf = T.run_inference_chemprop(run_id=run_id, smiles_list=_SMIS,
                                   accelerator="cpu")
    assert len(inf["predictions"]) == 2
    assert all(isinstance(p, float) for p in inf["predictions"])


def test_chemprop_classifier_round_trip(cls_run):
    run_id = cls_run
    tr = T.train_chemprop(run_id=run_id, epochs=1, batch_size=16,
                          accelerator="cpu")
    assert 0.0 <= tr["accuracy"] <= 1.0
    assert -1.0 <= tr["mcc"] <= 1.0
    assert tr["n_classes"] == 3

    ev = T.evaluate_model(run_id=run_id, model_type="chemprop")
    # Stored classification metrics (confusion matrix is sklearn/MLP-only;
    # per-molecule predictions come from run_inference_chemprop below).
    assert ev["model_type"] == "chemprop"
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert -1.0 <= ev["mcc"] <= 1.0
    assert 0.0 <= ev["f1_macro"] <= 1.0

    inf = T.run_inference_chemprop(run_id=run_id, smiles_list=_SMIS,
                                   accelerator="cpu")
    assert len(inf["predicted_class"]) == 2
    assert len(inf["predicted_labels"]) == 2
    assert all(c in (0, 1, 2) for c in inf["predicted_class"])
    assert all(" < " in lab for lab in inf["predicted_labels"])