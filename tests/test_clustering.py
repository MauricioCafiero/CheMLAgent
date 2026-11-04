"""Tests for chemlagent.clustering and the cluster_molecules tool.

The core functions are checked on synthetic blobs with a known answer (three
well-separated groups whose targets differ); the tool is run on the real
ChEMBL slice from conftest.seeded_run (ECFP), including the SMILES mapping and
the dimension_reduction(color_by='clusters') plot.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from chemlagent import tools as T
from chemlagent.clustering import cluster_features, cluster_quality


@pytest.fixture(scope="module")
def blobs():
    X, true = make_blobs(n_samples=300, centers=3, n_features=8,
                         cluster_std=0.5, random_state=0)
    y = true * 10.0 + np.random.default_rng(0).normal(0, 1, len(true))
    return X, y, true


@pytest.mark.parametrize("method", ["kmeans", "hdbscan"])
def test_methods_recover_three_blobs(blobs, method):
    X, y, true = blobs
    labels, info = cluster_features(X, method=method)
    assert info["n_clusters"] == 3
    if method == "kmeans":
        assert set(info["k_scan"]) == set(range(2, 21))
    keep = labels >= 0
    # every recovered cluster is pure (one true blob)
    for c in set(labels[keep]):
        assert len(set(true[labels == c])) == 1
    q = cluster_quality(y, labels)
    assert q["u_test_frac"] == 1.0
    assert sum(c["size"] for c in q["clusters"]) == keep.sum()


def test_fixed_k_and_single_cluster_quality(blobs):
    X, y, _ = blobs
    labels, info = cluster_features(X, method="kmeans", n_clusters=5)
    assert info["n_clusters"] == 5 and "k_scan" not in info
    q = cluster_quality(y, np.zeros(len(y), dtype=int))
    assert np.isnan(q["u_test_frac"]) and len(q["clusters"]) == 1


@pytest.mark.parametrize("method", ["kmeans", "hdbscan"])
def test_cluster_molecules_tool(seeded_run, method):
    r = T.cluster_molecules(run_id=seeded_run, method=method, max_clusters=8)
    assert r["fp_type"] == "ECFP" and r["n_clusters"] >= 1
    table = pd.read_csv(r["clusters_path"])
    assert len(table) == r["n_molecules"]
    assert {"SMILES", "split", "target", "cluster"} <= set(table.columns)
    assert (table["cluster"] >= 0).all() or method == "hdbscan"
    m = T._read_manifest(seeded_run)
    assert m["clusters"][f"{method}_ECFP"]["n_clusters"] == r["n_clusters"]
    assert m["latest_clusters"]["ECFP"] == r["clusters_path"]

    plot = T.dimension_reduction(run_id=seeded_run, method="pca",
                                 color_by="clusters")
    assert os.path.exists(plot["plot_path"])
