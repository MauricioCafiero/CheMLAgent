"""End-to-end classifier test on REAL data.

prepare_classes -> featurize -> train -> evaluate -> inference on a small
slice of the repo's real ChEMBL bioactivity CSV (tests/conftest.py), a
full round trip asserting the classification metric set (accuracy / MCC /
confusion matrix, range labels) instead of R^2. Skipped if the CSV is
missing so the suite still runs in a trimmed checkout.
"""

import numpy as np

import numpy as np

from chemlagent import tools as T


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

    inf = T.run_inference(
        run_id, smiles_list=["CCOc1ccc(cc1)C(=O)NC", "O=C(O)c1ccccc1OC(C)C"])
    assert len(inf["predicted_class"]) == 2
    assert len(inf["predicted_labels"]) == 2
    assert all(c in (0, 1, 2) for c in inf["predicted_class"])