"""End-to-end smoke test of the CheMLAgent tool pipeline (no Ollama).

Exercises prepare_chembl_csv -> featurize_fingerprints -> train_model ->
evaluate_model -> run_inference on a tiny synthetic dataset (SMILES vs a
deterministic target = molecular weight), so the plumbing is verifiable
without hitting ChEMBL or the LLM.
"""

import os
import shutil

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from chemlagent import tools as T


@pytest.fixture
def tiny_csv(tmp_path):
    smis = [
        "CCO", "CC(C)O", "CCC", "c1ccccc1", "CCN",
        "CCC(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "O=C(O)C1=CC=CC=C1N",
        "OC(=O)C1=CC=CC=C1O", "CC1=CC=CC=C1",
        "C1=CC=CC=C1C(=O)O", "OCC(O)C(O)C(O)C=O",
        "CCCC", "CCCCC", "CCCCCC",
        "C1=CC=NC=C1",
    ]
    rows = []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        rows.append({"SMILES": s, "IC50": float(Descriptors.MolWt(m)) + i})
    df = pd.DataFrame(rows)
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_pipeline_round_trip(tiny_csv, tmp_path, monkeypatch):
    # Run from a clean CWD under tmp_path so runs/ lands there.
    monkeypatch.chdir(tmp_path)
    run_id = "smoke"

    prep = T.prepare_chembl_csv(input_csv=tiny_csv, run_id=run_id)
    assert prep["n_rows"] > 0
    assert os.path.exists(prep["csv_path"])

    feat = T.featurize_fingerprints(run_id=run_id, fp_type="ECFP",
                                    test_size=0.25)
    assert feat["n_train"] + feat["n_test"] == prep["n_rows"]
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
    assert inf["log_transformed"] is False

    # manifest should carry flags from both data prep and featurize stages.
    assert os.path.exists(os.path.join("runs", run_id, "manifest.json"))