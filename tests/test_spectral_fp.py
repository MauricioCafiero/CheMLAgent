"""Spectral fingerprint (chemlagent.spectral_fp, ported from hjkgrp/SpectralScore).

The port was checked once against the reference get_spectrum (max relative
difference ~5e-7, float32 precision) in both 3D and 2D scope; these tests pin
its basic properties and run both fp_types through the real tool pipeline.
"""

import os

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from chemlagent import tools as T
from chemlagent.spectral_fp import SpectralFingerprint


def test_spectral_shape_padding_and_permutation_invariance():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    # Renumbering atoms must not change the fingerprint (Laplacian eigenvalues).
    perm = list(np.random.default_rng(0).permutation(mol.GetNumAtoms()))
    shuffled = Chem.RenumberAtoms(mol, [int(i) for i in perm])
    for use_3D in (True, False):
        fp = SpectralFingerprint(n_eigs=16, use_3D=use_3D)
        X = fp.transform([mol, shuffled])
        assert X.shape == (2, 64)
        assert np.allclose(X[0], X[1], atol=1e-5)

    # Ethanol has 9 atoms -> 8 non-trivial eigenvalues per channel, rest padded.
    X = SpectralFingerprint(use_3D=False).transform([Chem.MolFromSmiles("CCO")])
    per_channel = X.reshape(4, 16)
    assert np.all(per_channel[:, 8:] == 0) and np.all(per_channel[:, :8] > 0)


@pytest.mark.parametrize("fp_type", ["Spectral_2D", "Spectral"])
def test_spectral_pipeline_round_trip(fp_type, real_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "spectral"
    T.prepare_chembl_csv(input_csv=real_csv, run_id=run_id, limit=60)
    feat = T.featurize_fingerprints(run_id=run_id, fp_type=fp_type, test_size=0.25)
    assert feat["n_features"] == 64
    assert feat["is_3d"] == (fp_type == "Spectral")
    assert feat["n_dropped_no_conformer"] == len(feat["dropped_no_conformer"]) == 0
    trained = T.train_model(run_id=run_id, model_type="random_forest",
                            n_estimators=10)
    assert os.path.exists(trained["model_path"])
    # Inference goes through the pickled SmilesFeaturizer -> SpectralFingerprint.
    inf = T.run_inference(run_id=run_id, smiles_list=["CCO", "c1ccccc1O"])
    assert len(inf["predictions"]) == 2
