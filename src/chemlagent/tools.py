"""Ollama-callable tools for the CheMLAgent pipeline.

Each function here is a plain Python function with a numpydoc-style docstring.
Ollama infers its JSON tool schema from the signature + docstring, so the
docstrings are written to be informative for the model, not just for humans.

State between tool calls lives on disk under ``runs/<run_id>/`` (fitted
models, featurizers, feature matrices, a manifest), referenced by path/ID in
the JSON return values — Ollama tool calls can only exchange JSON, not live
Python objects.

MVP pipeline (all reuse existing logic in the sibling modules):

    prepare_chembl_csv  ->  featurize_fingerprints  ->  train_model
                                                       ->  evaluate_model
                                                       ->  run_inference
"""

# NOTE: do NOT use `from __future__ import annotations` here. Ollama builds the
# tool JSON schema from the function annotations via pydantic, which needs the
# annotations to be real resolved objects (Optional[str], list[str], ...) not
# PEP-563 stringized forward references.

import json
import os
import pickle

import numpy as np
import pandas as pd
import requests

# Import torch BEFORE scikit-fingerprints / lightgbm. On Apple Silicon both
# torch and lightgbm bundle OpenMP (libomp); if lightgbm's libomp loads first,
# a subsequent torch op segfaults (libomp double-load). Loading torch first
# makes its libomp win and the process is stable. Guarded so a base-only
# install (no torch extra) still imports -- in that env torch is never used, so
# there is no conflict to avoid.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from chemlagent.data import prepare_chembl_csv, run_dir, chembl_flag, new_client
from chemlagent.fingerprints import get_fingerprints
from chemlagent.models import (
    random_forest_regression,
    lightgbm_regression,
    svr_regression,
)
from chemlagent.products import publish_products

__all__ = [
    "search_uniprot",
    "list_bioactives",
    "search_targets",
    "prepare_chembl_csv",
    "featurize_fingerprints",
    "train_model",
    "evaluate_model",
    "run_inference",
    "train_mlp",
    "run_inference_mlp",
    "train_chemprop",
    "run_inference_chemprop",
]

_MANIFEST = "manifest.json"
_FEATURES = "features.npz"
_FEATURIZER = "featurizer.pkl"
_MODEL = "model.pkl"

# Broadcast print flag: the agent sets chemlagent.tools.print_flag = <args.print>
# at startup (see agent.py, mirroring src/agent_template.py). Every tool banner
# / progress print below is gated on it so tools are silent unless --print.
print_flag = False


# --- manifest helpers ------------------------------------------------------

def _manifest_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), _MANIFEST)


def _write_manifest(run_id: str, manifest: dict) -> str:
    path = _manifest_path(run_id)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def _read_manifest(run_id: str) -> dict:
    with open(_manifest_path(run_id)) as fh:
        return json.load(fh)


def _write_predictions_csv(run_id: str, smiles_list, preds) -> str:
    """Write (SMILES, prediction) to runs/<run_id>/predictions.csv and publish
    it to recent_work/. Returns the predictions CSV path."""
    path = os.path.join(run_dir(run_id), "predictions.csv")
    pd.DataFrame({"SMILES": list(smiles_list),
                  "prediction": np.asarray(preds, dtype=float).reshape(-1)}
                 ).to_csv(path, index=False)
    publish_products(run_id, run_dir(run_id), "predictions.csv", "manifest.json")
    return path


# --- tools -----------------------------------------------------------------

# --- discovery: protein / disease -> ChEMBL target ID ----------------------
#
# These three tools are the front door of the pipeline when the user gives a
# protein name or a disease instead of a ChEMBL ID. They reimplement the
# discovery nodes in modrag_protein_functions.py (uniprot_node,
# listbioactives_node, target_node) as stateless, JSON-returning Ollama tools:
# no ../scratch or ../images writes (those leak outside the project and can't
# cross the JSON tool boundary anyway). Once a chembl_id is found, hand it to
# prepare_chembl_csv(chembl_id=...) to build the training CSV.


def search_uniprot(
    protein_names: list[str],
    human_only: bool = False,
) -> dict:
    """Search UniProt for proteins by name and return their UniProt entries.

    For each protein name, queries the UniProt REST API and returns the matching
    entries (UniProt accession, gene names, organism, protein name). This is the
    first step in discovering a ChEMBL target ID from a protein name: feed the
    returned accession IDs into ``list_bioactives`` to get ChEMBL target IDs.

    Args:
        protein_names: One or more protein names / gene symbols to search for
            (e.g. ["EGFR"], ["BRCA1", "ESR1"]). Names are searched verbatim
            against UniProt's full-text search.
        human_only: If True, keep only entries whose Organism is
            "Homo sapiens (Human)". Default False keeps all organisms. Use True
            for drug-discovery QSAR work where you want the human target.

    Returns:
        dict with keys: results (list of {protein, uniprot_ids, info}), summary
        (a human-readable string of all matches, suitable to show the user).
    """
    print("UniProt search tool")
    print("=" * 55)
    results = []
    summary_parts = []
    for name in protein_names:
        try:
            url = "https://rest.uniprot.org/uniprotkb/search"
            resp = requests.get(url, params={"query": name, "format": "tsv"},
                                timeout=60)
            prot_df = pd.read_csv(pd.io.common.StringIO(resp.text), sep="\t")
        except Exception as exc:
            results.append({"protein": name, "uniprot_ids": [], "info": ""})
            summary_parts.append(f"No proteins found for {name} ({exc!r})\n"
                                 "==========\n")
            continue

        if human_only and "Organism" in prot_df.columns:
            prot_df = prot_df[prot_df["Organism"] == "Homo sapiens (Human)"]

        info_lines, ids = [], []
        for _, row in prot_df.iterrows():
            entry = row.get("Entry")
            gene = row.get("Gene Names", "")
            organism = row.get("Organism", "")
            pname = row.get("Protein names", "")
            if not entry:
                continue
            ids.append(entry)
            info_lines.append(
                f"Protein {name}, ID: {entry}, Gene: {gene}, "
                f"Organism: {organism}, Name: {pname}")
        info = "\n".join(info_lines)
        results.append({"protein": name, "uniprot_ids": ids, "info": info})
        summary_parts.append(
            (info if info else f"No proteins found for {name}")
            + "\n==========\n")

    return {"results": results, "summary": "".join(summary_parts)}


def list_bioactives(
    uniprot_ids: list[str],
    activity_type: str = "IC50",
) -> dict:
    """Map UniProt accession IDs to ChEMBL target IDs and their bioactive counts.

    For each UniProt ID, looks up the ChEMBL targets that map to it (via
    ``target_components__accession``), then counts the ChEMBL activity
    measurements of the requested type with an exact (``relation='='``) value.
    This is the second step in the discovery chain: the returned ChEMBL target
    IDs can be passed to ``prepare_chembl_csv(chembl_id=...)`` to build a
    training dataset. Pick the ChEMBL ID with the largest count for the most
    data.

    Args:
        uniprot_ids: UniProt accession IDs (e.g. ["P00533"] for human EGFR), as
            returned by ``search_uniprot``.
        activity_type: Activity type to count (default "IC50"). Must match the
            ``type`` field used by ChEMBL activity records. Use the same value
            you will later pass to prepare_chembl_csv.

    Returns:
        dict with keys: results (list of {uniprot_id, chembl_ids, counts}),
        summary (human-readable string). chembl_ids and counts are parallel
        lists: counts[i] is the number of activity_type measurements for
        chembl_ids[i].
    """
    print("List bioactives tool")
    print("=" * 55)
    if not chembl_flag:
        print("ChEMBL client not available at this time")
        return {
            "results": [{"uniprot_id": up, "chembl_ids": [], "counts": []}
                        for up in uniprot_ids],
            "summary": "ChEMBL client not available.\n",
        }

    targets = new_client.target
    bioact = new_client.activity

    results = []
    summary_parts = []
    for up_id in uniprot_ids:
        try:
            target_info = targets.get(target_components__accession=up_id).only(
                "target_chembl_id", "organism", "pref_name", "target_type")
            target_info = pd.DataFrame.from_records(target_info)
            chembl_ids = list(set(target_info["target_chembl_id"].tolist())) \
                if len(target_info) else []
        except Exception:
            chembl_ids = []

        counts = []
        for chembl_id in chembl_ids:
            chosen = bioact.filter(target_chembl_id=chembl_id,
                                   type=activity_type, relation="=").only(
                "molecule_chembl_id")
            counts.append(len(chosen))

        results.append({"uniprot_id": up_id, "chembl_ids": chembl_ids,
                        "counts": counts})
        if chembl_ids:
            lines = [f"For Uniprot {up_id}: ChEMBL ID {cid} -> {n} "
                     f"{activity_type} measurements"
                     for cid, n in zip(chembl_ids, counts)]
            summary_parts.append("\n".join(lines) + "\n==========\n")
        else:
            summary_parts.append(f"No bioactives found for Uniprot {up_id}\n"
                                 "==========\n")

    return {"results": results, "summary": "".join(summary_parts)}


def search_targets(disease_names: list[str]) -> dict:
    """Search Open Targets for proteins associated with a disease.

    For each disease name, queries the Open Targets GraphQL API: resolves the
    name to a disease EFO ID, then returns the associated targets ranked by
    association score. The returned gene symbols are protein names you can feed
    into ``search_uniprot`` (then ``list_bioactives``) to reach a ChEMBL target
    ID. Use this when the user names a disease rather than a protein.

    Args:
        disease_names: One or more disease names (e.g. ["breast cancer"],
            ["Alzheimer's disease"]). Searched verbatim against Open Targets.

    Returns:
        dict with keys: results (list of {disease, targets}), summary
        (human-readable string). targets is a ranked list of approved gene
        symbols (strings).
    """
    print("Open Targets tool")
    print("=" * 55)
    base_url = "https://api.platform.opentargets.org/api/v4/graphql"
    disease_query = """
      query searchEntity($queryString: String!) {
        search(queryString: $queryString){ total hits { id entity description } }
      }"""
    target_query = """
      query associatedTargets($efo_id: String!) {
        disease(efoId: $efo_id) {
          associatedTargets {
            rows { target { id approvedSymbol } score }
          }
        }
      }"""

    results = []
    summary_parts = []
    for disease in disease_names:
        targets = []
        try:
            r = requests.post(base_url, json={"query": disease_query,
                                              "variables": {"queryString": disease}},
                              timeout=60)
            disease_ids = [h["id"] for h in r.json()["data"]["search"]["hits"]
                           if h["entity"] == "disease"] if r.status_code == 200 else []
            if disease_ids:
                q = requests.post(base_url, json={
                    "query": target_query,
                    "variables": {"efo_id": disease_ids[0]}}, timeout=60)
                if q.status_code == 200:
                    rows = q.json()["data"]["disease"]["associatedTargets"]["rows"]
                    targets = [row["target"]["approvedSymbol"] for row in rows]
        except Exception:
            targets = []

        results.append({"disease": disease, "targets": targets})
        if targets:
            listing = "\n".join(f"{i+1}. {t}" for i, t in enumerate(targets))
            summary_parts.append(f"Possible targets for {disease}:\n{listing}\n"
                                 "==========\n")
        else:
            summary_parts.append(f"No targets found for {disease}\n==========\n")

    return {"results": results, "summary": "".join(summary_parts)}


def featurize_fingerprints(
    run_id: str,
    fp_type: str = "ECFP",
    test_size: float = 0.2,
) -> dict:
    """Featurize the prepared dataset into molecular fingerprints and split it.

    Reads the CSV produced by ``prepare_chembl_csv`` (runs/<run_id>/data.csv),
    computes the requested fingerprint for each molecule, and performs a Murcko
    scaffold-aware train/test split. The resulting feature matrices and a
    SMILES->features featurizer are saved under runs/<run_id>/ for use by
    train_model / evaluate_model / run_inference.

    Args:
        run_id: Pipeline run identifier; reads runs/<run_id>/data.csv and
            writes runs/<run_id>/features.npz, featurizer.pkl, manifest.json.
        fp_type: Fingerprint type. One of: ECFP (Morgan, default), Atom_Pair,
            Mordred, RDKit_2D, MACCS, PubChem, EState, Functional_Groups,
            RDKitFingerprint, E3FP, Autocorr, MORSE, RDF. The last four are 3D
            and require conformer generation (slower).
        test_size: Fraction of scaffolds held out for testing (default 0.2).

    Returns:
        dict with keys: run_id, fp_type, features_path, featurizer_path,
        n_train, n_test, n_features, is_3d.
    """
    print("Featurize fingerprints tool")
    print("=" * 55)
    data_csv = os.path.join(run_dir(run_id), "data.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"{data_csv} not found. Run prepare_chembl_csv for run_id "
            f"{run_id!r} first.")
    df = pd.read_csv(data_csv)
    smiles = df["SMILES"].tolist()
    targets = df["target"].to_numpy(dtype=float)

    gfp = get_fingerprints(
        smiles_list=smiles,
        target_list=targets,
        transform_flag=False,  # target already log-transformed in data.csv
        n_jobs=-1,
    )
    X_train, X_test, y_train, y_test = gfp.create(
        fp_type=fp_type, many_conf=True, num_confs=5, test_size=test_size,
    )

    out = os.path.join(run_dir(run_id), _FEATURES)
    np.savez(out, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # Pickle a SMILES->features wrapper (not the raw skfp estimator) so the
    # inference tools can call .transform(smiles) directly.
    from chemlagent.fingerprints import SmilesFeaturizer
    featurizer = SmilesFeaturizer(gfp.fp, is_3d=(fp_type in gfp.types_3d),
                                  many_conf=True, num_confs=5, n_jobs=-1)
    featurizer_path = os.path.join(run_dir(run_id), _FEATURIZER)
    with open(featurizer_path, "wb") as fh:
        pickle.dump(featurizer, fh)

    manifest = _read_manifest(run_id) if os.path.exists(
        _manifest_path(run_id)) else {}
    manifest.update({
        "run_id": run_id,
        "fp_type": fp_type,
        "is_3d": fp_type in gfp.types_3d,
        "many_conf": True,
        "num_confs": 5,
        "features_path": out,
        "featurizer_path": featurizer_path,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    })
    _write_manifest(run_id, manifest)
    return manifest


def train_model(
    run_id: str,
    model_type: str = "random_forest",
    n_estimators: int = 100,
) -> dict:
    """Train a regression model on the featurized dataset and save it.

    Loads runs/<run_id>/features.npz, trains the requested model, pickles it to
    runs/<run_id>/model.pkl, and records metrics + model type in the run
    manifest. The saved model is a sklearn estimator (or Pipeline for SVR)
    whose .predict takes the same fingerprint features.

    SVR uses literature-tuned poly-kernel hyperparameters internally (the ones
    models.svr_regression defaults to); lightgbm uses a single OpenMP thread to
    avoid the Apple-Silicon/MPS segfault. Only the model type and (for the tree
    models) the estimator count are exposed, to keep tool calls simple.

    Args:
        run_id: Pipeline run identifier; reads features.npz, writes model.pkl
            and updates manifest.json.
        model_type: One of 'random_forest', 'lightgbm', or 'svr'.
        n_estimators: Number of trees/estimators for random_forest and
            lightgbm (default 100). Ignored for svr.

    Returns:
        dict with keys: run_id, model_type, model_path, r2, r2_train,
        manifest_path.
    """
    print("Train model tool")
    print("=" * 55)
    feats = np.load(os.path.join(run_dir(run_id), _FEATURES))
    X_train, X_test = feats["X_train"], feats["X_test"]
    y_train, y_test = feats["y_train"], feats["y_test"]

    model_type = model_type.lower()
    if model_type in ("random_forest", "rf", "randomforest"):
        r2, r2_train, model = random_forest_regression(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators, random_state=42)
    elif model_type in ("lightgbm", "lgbm"):
        r2, r2_train, model = lightgbm_regression(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators, random_state=42, n_jobs=1)
    elif model_type == "svr":
        # Tuned defaults from models.svr_regression (poly kernel, no PCA).
        r2, r2_train, model = svr_regression(X_train, y_train, X_test, y_test)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Use 'random_forest', "
            f"'lightgbm', or 'svr'.")

    model_path = os.path.join(run_dir(run_id), _MODEL)
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    manifest = _read_manifest(run_id)
    manifest.update({
        "model_type": model_type,
        "model_path": model_path,
        "r2": float(r2),
        "r2_train": float(r2_train),
    })
    manifest_path = _write_manifest(run_id, manifest)
    publish_products(run_id, run_dir(run_id), _MODEL, _MANIFEST)

    return {
        "run_id": run_id,
        "model_type": model_type,
        "model_path": model_path,
        "r2": float(r2),
        "r2_train": float(r2_train),
        "manifest_path": manifest_path,
    }


def evaluate_model(run_id: str) -> dict:
    """Evaluate the saved model on the held-out test split.

    Loads runs/<run_id>/model.pkl and features.npz, predicts on the test
    features, and returns regression metrics plus per-molecule predictions.

    Args:
        run_id: Pipeline run identifier; reads model.pkl and features.npz.

    Returns:
        dict with keys: run_id, r2, mae, n_test, predictions (list[float]),
        truths (list[float]).
    """
    print("Evaluate model tool")
    print("=" * 55)
    from sklearn.metrics import r2_score, mean_absolute_error

    manifest = _read_manifest(run_id)
    model_path = manifest["model_path"]
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    feats = np.load(os.path.join(run_dir(run_id), _FEATURES))
    X_test, y_test = feats["X_test"], feats["y_test"]
    preds = np.asarray(model.predict(X_test)).reshape(-1)
    truths = np.asarray(y_test, dtype=float).reshape(-1)

    return {
        "run_id": run_id,
        "r2": float(r2_score(truths, preds)),
        "mae": float(mean_absolute_error(truths, preds)),
        "n_test": int(len(truths)),
        "predictions": preds.tolist(),
        "truths": truths.tolist(),
    }


def run_inference(run_id: str, smiles_list: list[str]) -> dict:
    """Predict the target for novel SMILES using a trained model.

    Featurizes the given SMILES with the saved fingerprint estimator, predicts
    with the saved model, and inverts the log10 transform applied during data
    prep if the manifest records log_transformed=True (so predictions come back
    in the original units, e.g. IC50 nM).

    Args:
        run_id: Pipeline run identifier; reads model.pkl, featurizer.pkl, and
            manifest.json.
        smiles_list: List of SMILES strings to predict for.

    Returns:
        dict with keys: run_id, log_transformed, predictions (list[float], in
        original units), smiles (list[str], the input echoed back).
    """
    print("Run inference tool")
    print("=" * 55)
    manifest = _read_manifest(run_id)
    with open(manifest["model_path"], "rb") as fh:
        model = pickle.load(fh)
    with open(manifest["featurizer_path"], "rb") as fh:
        featurizer = pickle.load(fh)

    X = featurizer.transform(smiles_list)
    preds = np.asarray(model.predict(X), dtype=float).reshape(-1)

    # Recover original-unit target. prepare_chembl_csv records whether the
    # target was log10-transformed; if so, invert with 10**pred.
    log_transformed = manifest.get("log_transformed", False)
    if log_transformed:
        preds = np.power(10.0, preds)

    pred_path = _write_predictions_csv(run_id, smiles_list, preds)
    return {
        "run_id": run_id,
        "log_transformed": bool(log_transformed),
        "smiles": list(smiles_list),
        "predictions": preds.tolist(),
        "predictions_csv": pred_path,
    }


# --- MLP (PyTorch) ---------------------------------------------------------

_MLP_MODEL = "mlp_model.pt"
_MLP_PREP = "mlp_prep.npz"


def train_mlp(
    run_id: str,
    epochs: int = 500,
    patience: int = 75,
    min_delta: float = 1e-5,
) -> dict:
    """Train a PyTorch MLP regressor on the featurized dataset and save it.

    Loads runs/<run_id>/features.npz, applies the MLP's preprocessing
    (aggressive descriptor cleaning + standardization + PCA to 95% variance,
    fit on train only), and trains the wide-and-deep sigmoid MLP (input +
    2 sigmoid hidden(250) + linear out, skip connection) with the tuned
    hyperparameters: SGD(lr=0.002, weight_decay=0.2), batch 32, raw targets.
    Saves weights + preprocessing stats under runs/<run_id>/.

    Training stops early once the epoch-averaged training loss stops
    improving: if it does not fall by at least ``min_delta`` below the best
    seen so far for ``patience`` consecutive epochs, the loop breaks (so a
    plateaued run does not run all ``epochs``).

    Args:
        run_id: Pipeline run identifier; reads features.npz, writes
            mlp_model.pt, mlp_prep.npz, and updates manifest.json.
        epochs: Maximum number of training epochs (default 500). Early
            stopping (patience/min_delta) usually ends training before this
            ceiling is reached.
        patience: Early stopping patience (default 75). Stop after this many
            consecutive epochs with no loss improvement >= min_delta.
        min_delta: Minimum decrease in epoch loss counted as an improvement
            (default 1e-5). Looser than a strict plateau check so tiny
            noise-level drops do not reset the patience counter.

    Returns:
        dict with keys: run_id, model_type ('mlp'), model_path, prep_path,
        r2, r2_train, manifest_path.
    """
    print("Train MLP tool")
    print("=" * 55)
    import torch
    import torch.nn as nn
    from chemlagent.pytorch_mlp import MLP_Model, prep_data, train, evaluate_regression

    feats = np.load(os.path.join(run_dir(run_id), _FEATURES))
    X_train, X_test = feats["X_train"], feats["X_test"]
    y_train, y_test = feats["y_train"], feats["y_test"]

    rd = os.path.abspath(run_dir(run_id))
    cwd = os.getcwd()
    # MLP_Model writes a params file to CWD on construction; run from the run
    # dir so it lands with the other artifacts instead of the project root.
    os.chdir(rd)
    try:
        prep = prep_data(batch_size=32, shuffle=True, reduce_dim="pca",
                         pca_var=0.95)
        _, _, train_loader, _ = prep.create_data_loader(
            X_train, y_train, X_test, y_test)
        input_dims = int(prep.X_train.shape[1])
        model = MLP_Model(neurons=250, input_dims=input_dims,
                          num_hidden_layers=1, skip_connection=True)
        loss_fn = nn.MSELoss()
        # Tuned config: SGD, lr=0.002, weight_decay=0.2 (PyTorch wd adds wd*w
        # to the gradient, so wd=0.2 ~ Keras L2=0.1).
        opt = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay=0.2)
        best_loss = float('inf')
        stale = 0
        epochs_run = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            epoch_loss = train(train_loader, model, loss_fn, opt)
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    if print_flag:
                        print(f"Early stopping at epoch {epochs_run} "
                              f"(no loss improvement for {patience} epochs, "
                              f"best={best_loss:.7f})")
                    break

        train_r2, test_r2 = evaluate_regression(
            prep.X_train, np.asarray(y_train), prep.X_test,
            np.asarray(y_test), model)

        model_path = os.path.join(rd, _MLP_MODEL)
        torch.save(model.state_dict(), model_path)
        prep_path = os.path.join(rd, _MLP_PREP)
        prep.save_stats(prep_path)
    finally:
        os.chdir(cwd)

    manifest = _read_manifest(run_id)
    manifest.update({
        "model_type": "mlp",
        "model_path": model_path,
        "mlp_prep_path": prep_path,
        "mlp_arch": {"neurons": 250, "input_dims": input_dims,
                     "num_hidden_layers": 1, "skip_connection": True},
        "r2": float(test_r2),
        "r2_train": float(train_r2),
        "epochs_run": int(epochs_run),
    })
    manifest_path = _write_manifest(run_id, manifest)
    publish_products(run_id, run_dir(run_id), _MLP_MODEL, _MLP_PREP, _MANIFEST)
    return {
        "run_id": run_id,
        "model_type": "mlp",
        "model_path": model_path,
        "prep_path": prep_path,
        "r2": float(test_r2),
        "r2_train": float(train_r2),
        "epochs_run": int(epochs_run),
        "manifest_path": manifest_path,
    }


def run_inference_mlp(run_id: str, smiles_list: list[str]) -> dict:
    """Predict the target for novel SMILES using a trained MLP.

    Loads the saved MLP weights + preprocessing stats, featurizes the SMILES
    with the run's saved featurizer, applies the identical train-time
    preprocessing, predicts, and inverts the log10 transform if the manifest
    records log_transformed=True.

    Args:
        run_id: Pipeline run identifier; reads mlp_model.pt, mlp_prep.npz,
            featurizer.pkl, manifest.json.
        smiles_list: List of SMILES strings to predict for.

    Returns:
        dict with keys: run_id, log_transformed, smiles, predictions
        (list[float], in original units).
    """
    print("Run inference (MLP) tool")
    print("=" * 55)
    import torch
    from chemlagent.pytorch_mlp import MLP_Model, prep_data, predict_single_value

    manifest = _read_manifest(run_id)
    arch = manifest["mlp_arch"]
    rd = run_dir(run_id)

    prep = prep_data.load_stats(os.path.join(rd, _MLP_PREP))
    model = MLP_Model(neurons=arch["neurons"], input_dims=arch["input_dims"],
                      num_hidden_layers=arch["num_hidden_layers"],
                      skip_connection=arch["skip_connection"])
    model.load_state_dict(torch.load(os.path.join(rd, _MLP_MODEL),
                                     weights_only=True))
    with open(manifest["featurizer_path"], "rb") as fh:
        featurizer = pickle.load(fh)

    inverse = (lambda p: np.power(10.0, p)) if manifest.get(
        "log_transformed", False) else None
    preds = predict_single_value(model, featurizer, smiles_list, prep=prep,
                                 inverse_func=inverse, verbose=False)
    preds = np.asarray(preds, dtype=float).reshape(-1)
    pred_path = _write_predictions_csv(run_id, smiles_list, preds)
    return {
        "run_id": run_id,
        "log_transformed": bool(manifest.get("log_transformed", False)),
        "smiles": list(smiles_list),
        "predictions": preds.tolist(),
        "predictions_csv": pred_path,
    }


# --- Chemprop MPNN ---------------------------------------------------------

_CHEMPROP_MODEL = "chemprop_model.pt"
# Path for the CheMeleon foundation message-passing weights. UNUSED right now:
# train_chemprop only trains from-scratch MPNNs (the foundation model blows up
# memory on this machine). Kept here so the foundation path can be re-enabled
# later by passing it as foundation_path to chemprop_model(from_foundation=...).
_FOUNDATION_CACHE = os.path.join("runs", "chemeleon_mp.pt")


def _scaffold_train_val_test(smiles, targets, test_size=0.2, val_frac=0.1,
                             random_state=132):
    """Murcko-scaffold split into train/val/test (val carved from train)."""
    from chemlagent.fingerprints import scaffold_train_test_split
    tr_smi, te_smi, tr_y, te_y = scaffold_train_test_split(
        smiles, targets, test_size=test_size, random_state=random_state)
    tr_y = np.asarray(tr_y, dtype=float)
    rng = np.random.RandomState(random_state)
    n_val = max(1, int(len(tr_smi) * val_frac))
    val_idx = rng.choice(len(tr_smi), n_val, replace=False)
    mask = np.ones(len(tr_smi), dtype=bool)
    mask[val_idx] = False
    va_smi = [tr_smi[i] for i in val_idx]
    va_y = tr_y[val_idx]
    tr_smi2 = [tr_smi[i] for i in np.where(mask)[0]]
    tr_y2 = tr_y[mask]
    return tr_smi2, tr_y2, va_smi, va_y, list(te_smi), np.asarray(te_y, float)


def train_chemprop(
    run_id: str,
    epochs: int = 30,
    batch_size: int = 64,
    accelerator: str = "auto",
) -> dict:
    """Train a Chemprop message-passing neural network and save it.

    Reads runs/<run_id>/data.csv (SMILES + target), performs a Murcko
    scaffold split (train/val/test), builds a chemprop MPNN on molecular
    graphs, trains with a lightning Trainer (best-val-loss checkpoint), and
    saves the model to runs/<run_id>/chemprop_model.pt. Targets are read
    as-is from data.csv (log-transformed or not); predictions come back in
    that same space, and the manifest's log_transformed flag lets inference
    invert the log.

    data.csv is the SAME file the fingerprint pipeline uses, so chemprop
    respects the row cap applied by prepare_chembl_csv (default 2000 rows);
    it does not bypass that limit.

    Trains a from-scratch MPNN (chemprop's standard small model: d_h=300,
    depth=3). The CheMeleon foundation-MP path (pretrained weights, d_h=2048/
    depth=6/~8.7M params) is intentionally NOT exposed -- it blows up memory on
    this machine. Its loading machinery is retained in chemprop_model for a
    possible future re-enable, but train_chemprop only does from-scratch.

    Memory: even the from-scratch MPNN trains on MPS via accelerator='auto'
    with a per-epoch cache flush. If a run still pressures memory, lower
    batch_size (e.g. 16 or 32) or set accelerator='cpu' (slower but bounded
    RAM). All third-party logging/warnings are silenced unless --print is
    given; only the tool banner prints.

    Args:
        run_id: Pipeline run identifier; reads data.csv, writes
            chemprop_model.pt and updates manifest.json.
        epochs: Number of training epochs (default 30).
        batch_size: Training/eval batch size (default 64). Lower this (16-32)
            if a run runs low on memory.
        accelerator: 'auto' (default; MPS on Apple Silicon) or 'cpu'. Use
            'cpu' if MPS memory pressure is a problem -- slower but bounded.

    Returns:
        dict with keys: run_id, model_type ('chemprop'), model_path,
        is_foundation, r2, mae, n_train, n_test, manifest_path.
    """
    print("Train Chemprop tool")
    print("=" * 55)
    from sklearn.metrics import r2_score, mean_absolute_error
    from chemlagent.chemprop_model import chemprop_data, chemprop_model

    data_csv = os.path.join(run_dir(run_id), "data.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"{data_csv} not found. Run prepare_chembl_csv for run_id "
            f"{run_id!r} first.")
    df = pd.read_csv(data_csv)
    smiles = df["SMILES"].tolist()
    targets = df["target"].to_numpy(dtype=float)

    tr_smi, tr_y, va_smi, va_y, te_smi, te_y = _scaffold_train_val_test(
        smiles, targets)

    cd = chemprop_data()
    cd.load_pre_split(tr_smi, tr_y, va_smi, va_y, te_smi, te_y)
    scaler = cd.featurize()
    train_loader, val_loader, test_loader = cd.make_loaders(batch_size=batch_size)

    # A from-scratch MPNN only (chemprop's standard small model: d_h=300,
    # depth=3). The CheMeleon foundation-MP path (d_h=2048, depth=6, ~8.7M
    # params) is disabled for now -- it blows up memory on this machine. The
    # loading machinery (chemprop_model._load_foundation_mp / from_foundation)
    # is kept intact for a possible future re-enable; train_chemprop simply
    # does not expose it.
    cm = chemprop_model()
    cm.construct_model(scaler)
    cm.train_model(
        train_loader, val_loader, epochs=epochs,
        checkpoint_dir=os.path.join(run_dir(run_id), "checkpoints"),
        accelerator=accelerator)

    preds = np.asarray(cm.get_preds(test_loader, accelerator=accelerator),
                       dtype=float)
    r2 = float(r2_score(te_y, preds))
    mae = float(mean_absolute_error(te_y, preds))

    model_path = os.path.join(run_dir(run_id), _CHEMPROP_MODEL)
    cm.save_model(model_path)

    manifest = _read_manifest(run_id)
    manifest.update({
        "model_type": "chemprop",
        "model_path": model_path,
        "is_foundation": False,
        "r2": r2,
        "mae": mae,
        "n_train": int(len(tr_smi)),
        "n_test": int(len(te_smi)),
    })
    manifest_path = _write_manifest(run_id, manifest)
    publish_products(run_id, run_dir(run_id), _CHEMPROP_MODEL, _MANIFEST)
    return {
        "run_id": run_id,
        "model_type": "chemprop",
        "model_path": model_path,
        "is_foundation": False,
        "r2": r2,
        "mae": mae,
        "n_train": int(len(tr_smi)),
        "n_test": int(len(te_smi)),
        "manifest_path": manifest_path,
    }


def run_inference_chemprop(run_id: str, smiles_list: list[str],
                           batch_size: int = 64,
                           accelerator: str = "auto") -> dict:
    """Predict the target for novel SMILES using a trained Chemprop MPNN.

    Loads runs/<run_id>/chemprop_model.pt, builds a molecular-graph dataloader
    for the new SMILES, predicts (in the data.csv target space), and inverts
    the log10 transform if the manifest records log_transformed=True.

    Inference runs in-process on the project's Python 3.14 env (MPS, or CPU if
    accelerator='cpu'). All third-party logging/warnings are silenced unless
    --print is given; only the tool banner prints.

    Args:
        run_id: Pipeline run identifier; reads chemprop_model.pt, manifest.json.
        smiles_list: List of SMILES strings to predict for.
        batch_size: Inference batch size (default 64). Lower if memory-tight.
        accelerator: 'auto' (default; MPS on Apple Silicon) or 'cpu'.

    Returns:
        dict with keys: run_id, log_transformed, smiles, predictions
        (list[float], in original units).
    """
    print("Run inference (Chemprop) tool")
    print("=" * 55)
    from chemprop import featurizers
    from chemprop.models.model import MPNN
    from chemlagent.chemprop_model import chemprop_data, chemprop_model

    manifest = _read_manifest(run_id)
    rd = run_dir(run_id)

    cd = chemprop_data()
    cd.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    loader = cd.make_new_dataloader(smiles_list, batch_size=batch_size)

    cm = chemprop_model()  # host instance; .mpnn is overridden by the load
    cm.mpnn = MPNN.load_from_file(os.path.join(rd, _CHEMPROP_MODEL))
    preds = np.asarray(cm.get_preds(loader, accelerator=accelerator),
                       dtype=float).reshape(-1)

    log_transformed = manifest.get("log_transformed", False)
    if log_transformed:
        preds = np.power(10.0, preds)
    pred_path = _write_predictions_csv(run_id, smiles_list, preds)
    return {
        "run_id": run_id,
        "log_transformed": bool(log_transformed),
        "smiles": list(smiles_list),
        "predictions": preds.tolist(),
        "predictions_csv": pred_path,
    }