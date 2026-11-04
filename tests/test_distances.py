"""Tests for fingerprints.get_distances (ported from CafChem get_distances) and
the compute_distances tool. All distance types are exercised -- MCS/Fraggle on
a handful of molecules so they stay quick.
"""

import os

import numpy as np
import pandas as pd
import pytest

from chemlagent import tools as T
from chemlagent.fingerprints import DISTANCE_TYPES, get_distances

MOLS = ["CC(=O)Nc1ccc(O)cc1", "CC(=O)Nc1ccc(OC)cc1", "c1ccccc1O",
        "CCCCCCCC(=O)O", "Nc1c2c(nc3ccccc13)CCCC2"]


@pytest.mark.parametrize("distance_type", list(DISTANCE_TYPES))
def test_distance_matrix_properties(distance_type):
    D = get_distances(MOLS, distance_type=distance_type)
    assert D.shape == (5, 5)
    assert np.allclose(np.diag(D), 0) and ((D >= 0) & (D <= 1)).all()
    # paracetamol is closer to its methyl ether than to octanoic acid
    assert D[0, 1] < D[0, 3]
    R = get_distances(MOLS[:2], MOLS, distance_type=distance_type)
    assert R.shape == (2, 5)


@pytest.mark.parametrize("fp_type", ["ECFP", "MACCS", "Atom_Pair", "RDKitFingerprint",
                                     "RDKit_2D", "Mordred", "Spectral_2D"])
def test_tanimoto_accepts_fp_types(fp_type):
    D = get_distances(MOLS, distance_type="tanimoto", fp_type=fp_type)
    assert D.shape == (5, 5) and np.allclose(np.diag(D), 0)
    assert ((D >= -1e-9) & (D <= 1 + 1e-9)).all()
    assert D[0, 1] < D[0, 3]
    with pytest.raises(ValueError, match="Unknown fp_type"):
        get_distances(MOLS, distance_type="tanimoto", fp_type="nope")


def test_fraggle_failures_are_flagged_and_slow_distances_capped():
    # Fraggle cannot fragment the fused tacrine core, even against itself.
    tac = MOLS[4]
    assert np.isnan(get_distances([tac], [tac], "fraggle", fraggle_fail_value=np.nan)).all()
    assert get_distances([tac], [tac], "fraggle")[0, 0] == 1.0
    with pytest.raises(ValueError, match="too long"):
        get_distances(["C"] * 300, ["C"] * 300, "mcs")


def test_compute_distances_tool_modes(seeded_run):
    run_id = seeded_run
    data = pd.read_csv(os.path.join("runs", run_id, "data.csv"))

    r = T.compute_distances(run_id=run_id, smiles_list=MOLS[:2],
                            against_dataset=True, top_k=3)
    assert r["reference"] == "dataset" and r["n_references"] == len(data)
    assert len(r["neighbours"]) == 2 and len(r["neighbours"][0]["nearest"]) == 3
    assert "target" in r["neighbours"][0]["nearest"][0]
    d = [n["distance"] for n in r["neighbours"][0]["nearest"]]
    assert d == sorted(d)
    assert os.path.exists(r["neighbours_path"]) and os.path.exists(r["matrix_path"])

    # pairwise: a molecule is never its own neighbour
    r = T.compute_distances(run_id=run_id, smiles_list=MOLS, top_k=2)
    assert r["reference"] == "pairwise"
    for q in r["neighbours"]:
        assert all(n["smiles"] != q["query"] for n in q["nearest"])

    r = T.compute_distances(run_id=run_id, smiles_list=[MOLS[4]],
                            reference_smiles=MOLS[:4], distance_type="fraggle")
    assert r["reference"] == "reference_smiles" and r["n_failed_pairs"] >= 1


@pytest.mark.parametrize("distance_type", ["mcs", "fraggle"])
def test_compute_distances_slow_types_against_dataset(seeded_run, distance_type):
    # 2 queries (paracetamol, tacrine) x the 200-molecule real slice = 400 pairs.
    r = T.compute_distances(run_id=seeded_run, smiles_list=[MOLS[0], MOLS[4]],
                            distance_type=distance_type, against_dataset=True, top_k=3)
    assert r["distance_type"] == distance_type and r["reference"] == "dataset"
    D = np.load(r["matrix_path"])["distances"]
    assert D.shape == (2, r["n_references"]) and ((D >= 0) & (D <= 1)).all()
    for q in r["neighbours"]:
        d = [n["distance"] for n in q["nearest"]]
        assert d == sorted(d) and "target" in q["nearest"][0]
    # Fraggle fragments the QUERY: a tacrine query fails against every reference,
    # a paracetamol query fails against none (even the tacrine references).
    # MCS has no failures.
    expected = r["n_references"] if distance_type == "fraggle" else 0
    assert r["n_failed_pairs"] == expected
