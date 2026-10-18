"""Tests for the ``dimension_reduction`` EDA tool.

Covers both projection methods (PCA / t-SNE), all three coloring modes, the
guarded outlier path, manifest recording, and that the saved PNG + manifest
plot_path survive the quit-time relocation into ``recent_work/``.

Uses the REAL pipeline (tests/conftest.py): prepare_chembl_csv +
featurize_fingerprints ECFP on a slice of the repo's ChEMBL CSV, featurized
once per module; the module-scoped seeded_run fixture switches CWD to a
per-module tmp dir so runs/ is isolated. Two tests keep tiny self-contained
synthetic runs on purpose (the 2048-dim outlier-speed regression and the
12-row perplexity clamp, neither of which real data can shape).
"""

import os

import numpy as np
import pytest

from chemlagent import tools as T


@pytest.mark.parametrize("method", ["pca", "tsne"])
@pytest.mark.parametrize("color_by", ["target", "split", "classes"])
def test_projects_and_plots_all_modes(seeded_run, method, color_by):
    run_id = seeded_run
    r = T.dimension_reduction(run_id=run_id, method=method, color_by=color_by,
                              num_classes=3, remove_outliers=True)
    assert r["run_id"] == run_id
    assert r["method"] == method
    assert r["color_by"] == color_by
    assert r["n_points"] > 0
    assert r["n_outliers_removed"] >= 0
    assert os.path.exists(r["plot_path"])
    # PNG lands under the run dir, keyed by method + fp_type (color_by is NOT
    # in the filename -- it is a view of the same projection).
    assert r["plot_path"].endswith("dim_reduction_pca_ECFP.png") \
        if method == "pca" else r["plot_path"].endswith(
        "dim_reduction_tsne_ECFP.png")
    assert r["variant"] == f"{method}_ECFP"

    if method == "pca":
        assert len(r["explained_variance"]) == 2
        assert np.isfinite(r["explained_variance"]).all()
        assert np.isnan(r["kl_divergence"])
    else:  # tsne
        assert r["explained_variance"] == []
        assert np.isfinite(r["kl_divergence"])

    # The run manifest carries an eda entry keyed by method + fp_type, and
    # records which color_by was used.
    m = T._read_manifest(run_id)
    key = f"{method}_ECFP"
    assert key in m["eda"]
    assert m["eda"][key]["plot_path"] == r["plot_path"]
    assert m["eda"][key]["color_by"] == color_by
    assert m["eda"][key]["n_points"] == r["n_points"]


def test_pca_dominant_first_component_on_real_fingerprints(seeded_run):
    # On real ECFP fingerprint data, the leading PC should still dominate the
    # second (real bits share structure; the variance drops off after PC1).
    r = T.dimension_reduction(run_id=seeded_run, method="pca", color_by="target",
                              remove_outliers=False)
    assert r["explained_variance"][0] > r["explained_variance"][1]
    assert r["explained_variance"][0] > 0.0


def test_outliers_removed_changes_count(seeded_run):
    run_id = seeded_run
    # With outlier removal on, the reported removed count is non-negative and
    # the n_points reflects the kept rows; with it off, n_points == all rows.
    on = T.dimension_reduction(run_id=run_id, method="pca", color_by="target",
                              remove_outliers=True)
    off = T.dimension_reduction(run_id=run_id, method="pca", color_by="target",
                               remove_outliers=False)
    assert on["n_outliers_removed"] >= 0
    assert off["n_outliers_removed"] == 0
    assert off["n_points"] == on["n_points"] + on["n_outliers_removed"]


def test_invalid_method_and_color_by_raise(seeded_run):
    with pytest.raises(ValueError, match="Unknown method"):
        T.dimension_reduction(run_id=seeded_run, method="umap")
    with pytest.raises(ValueError, match="Unknown color_by"):
        T.dimension_reduction(run_id=seeded_run, color_by="scaffold")


def test_missing_features_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = "no_feats"
    T.run_dir(run_id)  # create the dir but no features.npz
    T._write_manifest(run_id, {"run_id": run_id})
    with pytest.raises(FileNotFoundError, match="featurize_fingerprints"):
        T.dimension_reduction(run_id=run_id)


def test_high_dim_outlier_reduction_is_fast_and_clean(tmp_path, monkeypatch):
    """The regression test for the hang: EllipticEnvelope on a 2048-dim
    fingerprint matrix (5992-point ECFP run) used to grind for many minutes and
    emit a "not full rank" UserWarning. Now outlier detection runs in a
    PCA-reduced subspace, so it must finish in seconds and stay silent.
    """
    import time
    import warnings

    monkeypatch.chdir(tmp_path)
    run_id = "highdim"
    rng = np.random.default_rng(0)
    # Mimic a real ECFP run: 2048 features, ~600 rows (enough to be slow under
    # the old code and full-rank-ish under MCD). Sparse-ish like fingerprints.
    n, p = 600, 2048
    X = rng.random((n, p))
    X[X < 0.9] = 0.0  # ~90% sparse, like a fingerprint
    y = X[:, 0] * 5 + rng.normal(scale=0.1, size=n)
    d = T.run_dir(run_id)
    np.savez(os.path.join(d, T._FEATURES),
             X_train=X[:480], X_test=X[480:], y_train=y[:480], y_test=y[480:])
    T._write_manifest(run_id, {"run_id": run_id, "fp_type": "ECFP",
                               "target_units": "nM", "log_transformed": False})

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = T.dimension_reduction(run_id=run_id, method="pca",
                                  color_by="target", remove_outliers=True)
    dt = time.time() - t0

    assert os.path.exists(r["plot_path"])
    assert r["n_outliers_removed"] >= 0
    # Must not take minutes -- the old raw-space MCD was the hang. Generous
    # ceiling so this stays robust on a loaded CI/laptop.
    assert dt < 60, f"outlier reduction took {dt:.1f}s (expected <60s)"
    # No "not full rank" warning may escape -- that was the symptom.
    msgs = [str(w.message) for w in caught]
    assert not any("not full rank" in m for m in msgs), msgs


def test_tsne_perplexity_clamped_for_tiny_run(tmp_path, monkeypatch):
    # A run with fewer rows than the default perplexity (30) must not crash;
    # perplexity is clamped to n_samples - 1 internally.
    monkeypatch.chdir(tmp_path)
    run_id = "tiny"
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 8))
    y = X[:, 0]
    d = T.run_dir(run_id)
    np.savez(os.path.join(d, T._FEATURES),
             X_train=X[:8], X_test=X[8:], y_train=y[:8], y_test=y[8:])
    T._write_manifest(run_id, {"run_id": run_id})
    r = T.dimension_reduction(run_id=run_id, method="tsne",
                              remove_outliers=False)
    assert r["n_points"] == 12
    assert os.path.exists(r["plot_path"])


def test_different_fingerprints_do_not_overwrite(seeded_run):
    """The regression for the report: 'previous results were overwritten' when
    re-run on a different set of fingerprints. Re-featurizing the same run_id
    with a different fp_type and re-running EDA must leave the prior plot in
    place -- the file is keyed by method + fp_type, not just method."""
    run_id = seeded_run

    # First featurization (ECFP) + EDA.
    a = T.dimension_reduction(run_id=run_id, method="pca", color_by="target",
                              remove_outliers=False)
    assert a["plot_path"].endswith("dim_reduction_pca_ECFP.png")
    assert os.path.exists(a["plot_path"])

    # Simulate re-featurizing with a different fingerprint: overwrite the
    # manifest's fp_type (and features.npz, though the plot only needs the
    # manifest label) the way featurize_fingerprints would.
    m = T._read_manifest(run_id)
    m["fp_type"] = "Mordred"
    T._write_manifest(run_id, m)

    b = T.dimension_reduction(run_id=run_id, method="pca", color_by="target",
                              remove_outliers=False)
    assert b["plot_path"].endswith("dim_reduction_pca_Mordred.png")

    # The FIRST plot must still be on disk -- this is the bug that was reported.
    assert os.path.exists(a["plot_path"]), \
        "ECFP plot was clobbered by the Mordred run"
    assert os.path.exists(b["plot_path"])
    assert a["plot_path"] != b["plot_path"]

    # Two distinct manifest entries, one per fingerprint.
    m = T._read_manifest(run_id)
    assert {"pca_ECFP", "pca_Mordred"} <= set(m["eda"])


def test_recoloring_same_projection_overwrites(seeded_run):
    """color_by is a view of the same projection, not a separate result, so
    re-coloring writes the same file (latest wins). This documents the
    intended behavior that follows from not keying the file on color_by."""
    run_id = seeded_run
    # Earlier tests in this module may have re-labeled the manifest's fp_type
    # (test_different_fingerprints switches it to 'Mordred'), so key on whatever
    # fingerprint the manifest currently records.
    fp = T._read_manifest(run_id)["fp_type"]
    a = T.dimension_reduction(run_id=run_id, method="pca", color_by="target",
                              remove_outliers=False)
    b = T.dimension_reduction(run_id=run_id, method="pca", color_by="split",
                              remove_outliers=False)
    assert a["plot_path"] == b["plot_path"]  # same file
    m = T._read_manifest(run_id)
    # One entry, holding the latest color_by.
    assert m["eda"][f"pca_{fp}"]["color_by"] == "split"


def test_relocate_moves_png_and_rewrites_plot_path(seeded_run):
    """Runs LAST in this module: relocation moves the run out of runs/, so it
    must not disturb the tests above that keep using the shared run."""
    from chemlagent import products as P

    run_id = seeded_run
    r = T.dimension_reduction(run_id=run_id, method="pca", color_by="target",
                              remove_outliers=False)

    P.relocate_all_runs(runs_root="runs")
    rw = os.path.join(P.RECENT_WORK_DIR, run_id)
    assert os.path.exists(os.path.join(rw, "dim_reduction_pca_ECFP.png"))
    assert os.path.exists(os.path.join(rw, "manifest.json"))
    assert not os.path.exists(os.path.join("runs", run_id))  # intermediates gone

    # The relocated manifest's eda plot_path is rewritten to point at the
    # recent_work/ copy, so a later reload resolves to a real file.
    import json
    with open(os.path.join(rw, "manifest.json")) as fh:
        m = json.load(fh)
    p = m["eda"]["pca_ECFP"]["plot_path"]
    assert p.startswith(os.path.join(P.RECENT_WORK_DIR, run_id))
    assert os.path.exists(p)