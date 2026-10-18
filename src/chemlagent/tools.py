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
    random_forest_classifier,
    lightgbm_classifier,
    svc_classifier,
)
from chemlagent.classify import (
    make_classes_df,
    classification_metrics,
    class_label_strings,
)
from chemlagent.products import publish_products

# Hyperparameter-search dependencies. Imported here (not just in models.py) so
# grid_search can build the same estimators train_model uses and wrap them in
# GridSearchCV. n_jobs is pinned to 1 everywhere to avoid forking workers, which
# re-triggers the libomp double-load segfault (see memory: torch-before-lightgbm).
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, LGBMClassifier

from chemlagent.descriptor_cleaning import impute_only_cleaner

__all__ = [
    "search_uniprot",
    "list_bioactives",
    "search_targets",
    "prepare_chembl_csv",
    "prepare_classes",
    "featurize_fingerprints",
    "train_model",
    "grid_search",
    "evaluate_model",
    "run_inference",
    "train_mlp",
    "run_inference_mlp",
    "train_chemprop",
    "run_inference_chemprop",
    "dimension_reduction",
    "mlp_scan",
]

_MANIFEST = "manifest.json"
_FEATURES = "features.npz"
_FEATURIZER = "featurizer.pkl"
# Legacy single-model filename from older runs (pre per-type files). Kept for
# reference/back-compat; train_model / grid_search now write model_<type>.pkl.
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


# model_type aliases -> canonical name, across every tool that takes a
# model_type (train_model, grid_search, evaluate_model, run_inference). Keeps
# spelling tolerant ("rf", "lgbm", "randomforest") while the manifest always
# stores the canonical key.
_MODEL_ALIASES = {
    "random_forest": "random_forest", "rf": "random_forest",
    "randomforest": "random_forest",
    "lightgbm": "lightgbm", "lgbm": "lightgbm",
    "svr": "svr",
    "mlp": "mlp",
    "chemprop": "chemprop",
}


def _canonical_model_type(model_type: str) -> str:
    """Normalize a model_type spelling to its canonical name, or raise."""
    key = model_type.lower()
    canonical = _MODEL_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Use 'random_forest', "
            f"'lightgbm', 'svr', 'mlp', or 'chemprop'.")
    return canonical


def _record_model(run_id: str, model_type: str, model_path: str,
                  **fields) -> str:
    """Record a trained model in the run manifest and return the manifest path.

    Writes TWO things so multiple models trained on the same run stay
    independently reachable:

      * the flat "active model" fields (``model_type`` / ``model_path`` plus the
        given metrics) -- this is the most-recently-trained model and stays
        back-compatible with runs / callers that expect a single model per run;
      * a per-type entry under ``manifest["models"][model_type]`` holding the
        same ``model_path`` + metrics, so ``evaluate_model(run_id, model_type=...)``
        can pick any model trained on the run, not just the active one.

    Each model type writes its own file (e.g. ``model_lightgbm.pkl``,
    ``mlp_model.pt``), so training a second model never overwrites a first.
    """
    manifest = _read_manifest(run_id) if os.path.exists(
        _manifest_path(run_id)) else {}
    manifest["model_type"] = model_type
    manifest["model_path"] = model_path
    manifest.update(fields)
    models = manifest.setdefault("models", {})
    models[model_type] = {"model_path": model_path, **fields}
    return _write_manifest(run_id, manifest)


def _run_task(run_id: str, record: dict | None = None) -> str:
    """Task type ('regression' or 'classification') that a tool call should use.

    set by prepare_classes (``manifest["task"] = "classification"``). When a
    per-model ``record`` is given (e.g. manifest["models"]["mlp"]), a per-model
    ``task`` field wins over the manifest-level one, so a run that holds both a
    regression and a classification model still dispatches correctly.
    """
    if record is not None and record.get("task"):
        return record["task"]
    if os.path.exists(_manifest_path(run_id)):
        return _read_manifest(run_id).get("task", "regression")
    return "regression"


def _write_predictions_csv(run_id: str, smiles_list, preds,
                           pred_labels=None) -> str:
    """Write (SMILES, prediction) to runs/<run_id>/predictions.csv and publish
    it to recent_work/. Returns the predictions CSV path. For classification
    (pred_labels given) the columns are predicted_class (int indices) +
    predicted_label (range strings) instead of the float prediction."""
    path = os.path.join(run_dir(run_id), "predictions.csv")
    if pred_labels is not None:
        pd.DataFrame({"SMILES": list(smiles_list),
                      "predicted_class": np.asarray(preds, dtype=int),
                      "predicted_label": list(pred_labels)}
                     ).to_csv(path, index=False)
    else:
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


# --- classification: turn the target into classes ---------------------------
#
# prepare_classes ports the user's CafChemClassifiers.make_classes to the run
# pipeline: equal-count bands over the sorted target, written as int class
# indices 0..K-1 plus human-readable range labels, and a manifest task flag
# that every downstream tool (featurize / train / grid_search / evaluate /
# inference) keys off. Regression runs simply never call it.

def prepare_classes(run_id: str, num_classes: int = 2) -> dict:
    """Divide the run's target column into classes for classification.

    Reads runs/<run_id>/data.csv (from prepare_chembl_csv), bins the target
    into ``num_classes`` approximately equal-count bands (adapted from the
    user's CafChemClassifiers.make_classes: rows sorted by target, cut at
    multiples of n // num_classes, plus the bottom/top values as cutoffs), and
    writes two columns back to data.csv: ``class`` (int indices 0..K-1) and
    ``class_labels`` (human-readable ranges like '4.1 < 5.6'). It also sets
    manifest task='classification' with the n_classes + class_labels, which
    makes all downstream tools (featurize_fingerprints, train_model,
    grid_search, train_mlp, train_chemprop, evaluate_model, run_inference)
    train classifier versions and report accuracy/MCC/F1 instead of R^2/MAE.

    Call AFTER prepare_chembl_csv and BEFORE featurize_fingerprints (and
    before train_chemprop, which reads data.csv directly). Counter-ions
    ([Na+], [Cl-], etc.) are stripped from the SMILES column.

    Args:
        run_id: Pipeline run identifier; rewrites runs/<run_id>/data.csv in
            place and updates manifest.json.
        num_classes: Number of classes to divide the dataset into (default 2
            = binary active/inactive). Must be >= 2 and <= half the row count.

    Returns:
        dict with keys: run_id, num_classes (the ACTUAL number of classes; the
        band algorithm can produce one fewer if cutoffs collide), n_samples,
        samples_per_class, class_labels (list of range strings),
        class_counts (list of sample counts per class), manifest_path.
    """
    print("Prepare classes tool")
    print("=" * 55)
    data_csv = os.path.join(run_dir(run_id), "data.csv")
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"{data_csv} not found. Run prepare_chembl_csv for run_id "
            f"{run_id!r} first.")
    df = pd.read_csv(data_csv)
    df, class_labels = make_classes_df(df, "target", num_classes)
    df.to_csv(data_csv, index=False)

    n_classes = len(class_labels)
    counts = np.bincount(df["class"].to_numpy(dtype=int),
                         minlength=n_classes).tolist()

    manifest = _read_manifest(run_id) if os.path.exists(
        _manifest_path(run_id)) else {}
    manifest.update({
        "run_id": run_id,
        "task": "classification",
        "n_classes": int(n_classes),
        "class_labels": class_labels,
    })
    manifest_path = _write_manifest(run_id, manifest)
    publish_products(run_id, run_dir(run_id), "data.csv", _MANIFEST)

    if print_flag:
        for label, count in zip(class_labels, counts):
            print(f"  {label}: {count} samples")

    return {
        "run_id": run_id,
        "num_classes": int(n_classes),
        "n_samples": int(len(df)),
        "samples_per_class": int(len(df) // n_classes),
        "class_labels": class_labels,
        "class_counts": [int(c) for c in counts],
        "manifest_path": manifest_path,
    }


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

    If prepare_classes has been run on this run_id (task=classification), the
    split is against the int CLASS labels (for classifier training) instead of
    the numeric target.

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
    # Classification runs (prepare_classes set task='classification') featurize
    # against the int class column; regression against the numeric target.
    task = (_read_manifest(run_id).get("task", "regression")
            if os.path.exists(_manifest_path(run_id)) else "regression")
    if task == "classification":
        if "class" not in df.columns:
            raise ValueError(
                f"Run {run_id!r} is a classification run but data.csv has no "
                f"'class' column. Run prepare_classes(run_id, num_classes=...) "
                f"before featurize_fingerprints.")
        targets = df["class"].to_numpy(dtype=np.int64)
    else:
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
        "task": task,
        "n_classes": int(len(np.unique(targets))) if task == "classification"
        else None,
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
    runs/<run_id>/model_<model_type>.pkl, and records metrics + model type in
    the run manifest. The saved model is a sklearn estimator (or Pipeline for
    SVR) whose .predict takes the same fingerprint features.

    SVR uses literature-tuned poly-kernel hyperparameters internally (the ones
    models.svr_regression defaults to); lightgbm uses a single OpenMP thread to
    avoid the Apple-Silicon/MPS segfault. Only the model type and (for the tree
    models) the estimator count are exposed, to keep tool calls simple.

    Args:
        run_id: Pipeline run identifier; reads features.npz, writes
            model_<model_type>.pkl and updates manifest.json.
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

    model_type = _canonical_model_type(model_type)
    # prepare_classes set task='classification': train the classifier variant
    # of the requested model (same estimator family, int class targets from
    # features.npz). Accuracy replaces R^2 everywhere below.
    cls = _run_task(run_id) == "classification"
    if cls:
        y_train = np.asarray(y_train, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)

    if model_type == "random_forest":
        if cls:
            acc, acc_train, model = random_forest_classifier(
                X_train, y_train, X_test, y_test,
                n_estimators=n_estimators, random_state=42)
        else:
            r2, r2_train, model = random_forest_regression(
                X_train, y_train, X_test, y_test,
                n_estimators=n_estimators, random_state=42)
    elif model_type == "lightgbm":
        if cls:
            acc, acc_train, model = lightgbm_classifier(
                X_train, y_train, X_test, y_test,
                n_estimators=n_estimators, random_state=42, n_jobs=1)
        else:
            r2, r2_train, model = lightgbm_regression(
                X_train, y_train, X_test, y_test,
                n_estimators=n_estimators, random_state=42, n_jobs=1)
    elif model_type == "svr":
        # Tuned defaults from models.svr_regression (poly kernel, no PCA);
        # for classification runs the SVC counterpart (models.svc_classifier).
        if cls:
            acc, acc_train, model = svc_classifier(
                X_train, y_train, X_test, y_test)
        else:
            r2, r2_train, model = svr_regression(
                X_train, y_train, X_test, y_test)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Use 'random_forest', "
            f"'lightgbm', or 'svr'.")

    # Per-type filename so training a second sklearn model on the same run
    # never overwrites a first (e.g. train random_forest then lightgbm and both
    # survive for evaluate_model(run_id, model_type=...)).
    model_path = os.path.join(run_dir(run_id), f"model_{model_type}.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    if cls:
        manifest_path = _record_model(
            run_id, model_type, model_path, task="classification",
            accuracy=float(acc), accuracy_train=float(acc_train))
        publish_products(run_id, run_dir(run_id),
                         os.path.basename(model_path), _MANIFEST)
        return {
            "run_id": run_id,
            "model_type": model_type,
            "model_path": model_path,
            "accuracy": float(acc),
            "accuracy_train": float(acc_train),
            "manifest_path": manifest_path,
        }

    manifest_path = _record_model(
        run_id, model_type, model_path,
        r2=float(r2), r2_train=float(r2_train))
    publish_products(run_id, run_dir(run_id), os.path.basename(model_path),
                     _MANIFEST)

    return {
        "run_id": run_id,
        "model_type": model_type,
        "model_path": model_path,
        "r2": float(r2),
        "r2_train": float(r2_train),
        "manifest_path": manifest_path,
    }


# --- hyperparameter search ------------------------------------------------
#
# grid_search tunes ONE of the three sklearn models (random_forest, lightgbm,
# svr) over a small user-supplied grid, then saves the best model exactly as
# train_model does (model_<type>.pkl + manifest), so evaluate_model / run_inference
# work unchanged on the result. Search uses k-fold CV on the TRAIN split only
# (the held-out test split in features.npz is never seen during tuning), then
# refits the best params on the full train and reports test r2/mae.

# Cap the search cost so a tool call can't fan out into hundreds of fits:
# at most 3 values per hyperparameter and at most 6 combinations total (so a
# 3x3=9 grid is rejected). With cv=5 that's <= 30 fits, which is seconds for
# RF/LightGBM/SVR single-threaded.
_GRID_MAX_VALUES_PER_PARAM = 3
_GRID_MAX_COMBINATIONS = 6
_GRID_CV_FOLDS = 5

# Tunable hyperparameters exposed per model type. Anything outside this set is
# rejected, so the agent can't pass unsupported or mis-spelled knobs. The SVR
# poly-kernel shape (degree=2, coef0=7) is fixed at the literature value and
# NOT searched; only C / gamma / epsilon are tunable.
_GRID_ALLOWED_PARAMS: dict[str, set[str]] = {
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf",
                      "max_features"},
    "lightgbm": {"n_estimators", "num_leaves", "learning_rate",
                 "min_child_samples"},
    "svr": {"C", "gamma", "epsilon"},
}

# model_type aliases -> canonical name (mirrors train_model's acceptance set).
_GRID_ALIASES = {
    "random_forest": "random_forest", "rf": "random_forest",
    "randomforest": "random_forest",
    "lightgbm": "lightgbm", "lgbm": "lightgbm",
    "svr": "svr",
}


def _validate_param_grid(model_type: str,
                         param_grid: dict[str, list]) -> tuple[dict[str, list], int]:
    """Coerce + validate a user-supplied param grid for the given model type.

    Returns (clean_grid, n_combinations). Raises ValueError on any violation:
    unknown param, fewer than 2 values, more than _GRID_MAX_VALUES_PER_PARAM
    values, or more than _GRID_MAX_COMBINATIONS total combinations.
    """
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError(
            "param_grid must be a non-empty dict of hyperparameter -> list of "
            "values, e.g. {'n_estimators': [100, 200, 400]}.")
    allowed = _GRID_ALLOWED_PARAMS[model_type]
    unknown = set(param_grid) - allowed
    if unknown:
        raise ValueError(
            f"Unknown hyperparameter(s) for {model_type!r}: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}.")
    clean: dict[str, list] = {}
    n_combos = 1
    for key, vals in param_grid.items():
        values = list(vals)
        # max_depth is the only param where None ("unlimited") is meaningful.
        # If the model emitted the literal strings "None"/"null" (valid JSON,
        # unlike bare Python None which Ollama drops before we're reached),
        # coerce them to None so sklearn gets what it expects.
        if key == "max_depth":
            values = [None if (isinstance(v, str) and v.lower() in ("none", "null"))
                      else v for v in values]
        if len(values) < 2:
            raise ValueError(
                f"{key}: a grid search needs at least 2 values to compare, "
                f"got {len(values)}.")
        if len(values) > _GRID_MAX_VALUES_PER_PARAM:
            raise ValueError(
                f"{key}: at most {_GRID_MAX_VALUES_PER_PARAM} values per "
                f"hyperparameter, got {len(values)}.")
        n_combos *= len(values)
        clean[key] = values
    if n_combos > _GRID_MAX_COMBINATIONS:
        raise ValueError(
            f"param_grid expands to {n_combos} combinations; the limit is "
            f"{_GRID_MAX_COMBINATIONS} total. Drop a value or a hyperparameter.")
    return clean, n_combos


def _build_search_estimator(model_type: str, user_grid: dict[str, list],
                            classification: bool = False
                            ) -> tuple[object, dict[str, list]]:
    """Build the base estimator + sklearn-form grid for one model type.

    RF/LightGBM get the bare estimator (param names match the user grid
    directly). SVR/SVC get the same clean -> scale -> poly-kernel Pipeline as
    the models.py functions, with user params namespaced as 'svr__C' etc. so
    GridSearchCV routes them to the SVR step. classification=True swaps every
    estimator for its classifier counterpart (models.py classifier defaults:
    SVC poly degree=3, coef0=2).
    """
    if model_type == "random_forest":
        if classification:
            est = RandomForestClassifier(random_state=42, n_jobs=1)
        else:
            est = RandomForestRegressor(random_state=42, n_jobs=1)
        return est, dict(user_grid)
    if model_type == "lightgbm":
        if classification:
            est = LGBMClassifier(random_state=42, n_jobs=1, verbose=-1)
        else:
            est = LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)
        return est, dict(user_grid)
    # svr / svc
    name = "svc" if classification else "svr"
    cls_ = SVC(kernel="poly", degree=3, coef0=2) if classification else \
        SVR(kernel="poly", degree=2, coef0=7)
    est = Pipeline([
        ("clean", impute_only_cleaner()),
        ("scale", StandardScaler()),
        (name, cls_),
    ])
    grid = {f"{name}__{k}": v for k, v in user_grid.items()}
    return est, grid


def grid_search(
    run_id: str,
    model_type: str,
    param_grid: dict[str, list],
) -> dict:
    """Grid-search hyperparameters for a model and save the best fit.

    Runs k-fold (5-fold) cross-validated grid search over a SMALL
    hyperparameter grid on the TRAIN split only (the held-out test split in
    features.npz is never seen during tuning), refits the best combination on
    the full train set, evaluates on the held-out test split, and saves the
    result exactly like train_model (model_<model_type>.pkl + manifest
    update) so evaluate_model / run_inference work unchanged on the result.

    The grid is capped to keep tool calls bounded: at most 3 values per
    hyperparameter and at most 6 total combinations (so a 3x3=9 grid is
    rejected with an error). With 5-fold CV that is at most 30 fits, which is
    seconds single-threaded. Search runs single-threaded (n_jobs=1) to avoid
    forking workers, which would re-trigger the Apple-Silicon libomp
    double-load segfault (see train_model's note on lightgbm n_jobs).

    Only one model type is searched per call. The tunable hyperparameters are
    whitelisted per model type; unknown ones are rejected:

      - random_forest: n_estimators, max_depth, min_samples_leaf, max_features
      - lightgbm: n_estimators, num_leaves, learning_rate, min_child_samples
      - svr: C, gamma, epsilon (the poly kernel's degree=2 / coef0=7 are fixed
        at the literature value and are NOT searched)

    Classification runs (prepare_classes set task='classification') search the
    classifier variants (RandomForestClassifier / LGBMClassifier / a poly-SVC
    Pipeline, degree=3 / coef0=2 fixed) and score CV folds by accuracy; the
    result keys are then best_cv_accuracy / accuracy instead of best_cv_r2 /
    r2.

    Use the SAME run_id as the preceding featurize_fingerprints call. This
    saves model_<model_type>.pkl (per-type, so it does not overwrite a
    different model type already trained on the run) and records it in the
    manifest, making it the run's active model.

    Args:
        run_id: Pipeline run identifier; reads features.npz, writes
            model_<model_type>.pkl and updates manifest.json.
        model_type: One of 'random_forest' (aliases 'rf', 'randomforest'),
            'lightgbm' (alias 'lgbm'), or 'svr'.
        param_grid: dict of hyperparameter name -> list of values to search.
            Emit this as a JSON object whose values are JSON arrays. Use JSON
            null (never the Python literal None) for "unlimited", e.g.
            {"n_estimators": [100, 300], "max_depth": [null, 20, 40]}.
            Values may be ints, floats, strings, or null (max_depth=null means
            unlimited depth; only max_depth accepts null). Writing Python None
            in the tool call is invalid JSON and the call will be dropped.
            At most 3 values per key and 6 combinations total.

    Returns:
        dict with keys: run_id, model_type, best_params (dict, user-facing
        names), best_cv_r2 (float, mean CV r2 of the best combo), r2, r2_train,
        model_path, manifest_path.
    """
    print("Grid search tool")
    print("=" * 55)

    canonical = _GRID_ALIASES.get(model_type.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Use 'random_forest', "
            f"'lightgbm', or 'svr'.")
    model_type = canonical

    user_grid, n_combos = _validate_param_grid(model_type, param_grid)

    feats = np.load(os.path.join(run_dir(run_id), _FEATURES))
    X_train, X_test = feats["X_train"], feats["X_test"]
    y_train, y_test = feats["y_train"], feats["y_test"]

    # prepare_classes set task='classification': search the classifier
    # variant and score CV folds by accuracy (regression keeps scoring r2).
    cls = _run_task(run_id) == "classification"
    if cls:
        y_train = np.asarray(y_train, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)
    estimator, sk_grid = _build_search_estimator(model_type, user_grid,
                                                 classification=cls)

    # Shrink CV folds for small train sets so each fold still has enough rows.
    n_train = int(np.asarray(y_train).reshape(-1).shape[0])
    cv = _GRID_CV_FOLDS
    while cv > 2 and n_train // cv < 5:
        cv -= 1

    print(f"Searching {model_type} over {n_combos} combination(s) "
          f"({cv}-fold CV, n_jobs=1).")
    scoring = "accuracy" if cls else "r2"
    gs = GridSearchCV(
        estimator, sk_grid, cv=cv, scoring=scoring,
        n_jobs=1, refit=True, error_score="raise",
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_          # refit on full X_train (refit=True)
    best_cv = float(gs.best_score_)

    # best_params_ uses sklearn namespaced names for SVR/SVC ('svr__C'); strip
    # the prefix so the return value uses the user-facing hyperparameter names.
    best_params = {
        (k.split("__", 1)[1] if "__" in k else k): v
        for k, v in gs.best_params_.items()
    }

    y_pred = best.predict(X_test)
    y_train_pred = best.predict(X_train)
    if cls:
        from sklearn.metrics import accuracy_score
        acc = float(accuracy_score(y_test, y_pred))
        acc_train = float(accuracy_score(y_train, y_train_pred))
    else:
        r2 = float(r2_score(y_test, y_pred))
        r2_train = float(r2_score(y_train, y_train_pred))

    if print_flag:
        print(f"CV results (sorted by mean {scoring}):")
        rows = sorted(gs.cv_results_["mean_test_score"], reverse=True)
        for rank, mean in enumerate(rows, 1):
            print(f"  {rank}. mean_cv_{scoring}={mean:.4f}")
        print(f"Best params: {best_params}  (best_cv_{scoring}={best_cv:.4f})")

    model_path = os.path.join(run_dir(run_id), f"model_{model_type}.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(best, fh)

    if cls:
        manifest_path = _record_model(
            run_id, model_type, model_path, task="classification",
            accuracy=acc, accuracy_train=acc_train,
            grid_search=True,
            best_params={k: (None if v is None else v) for k, v in
                         best_params.items()},
            best_cv_accuracy=best_cv,
            n_grid_combinations=n_combos,
            cv_folds=cv)
        publish_products(run_id, run_dir(run_id),
                         os.path.basename(model_path), _MANIFEST)
        return {
            "run_id": run_id,
            "model_type": model_type,
            "best_params": best_params,
            "best_cv_accuracy": best_cv,
            "accuracy": acc,
            "accuracy_train": acc_train,
            "model_path": model_path,
            "manifest_path": manifest_path,
        }

    manifest_path = _record_model(
        run_id, model_type, model_path,
        r2=r2, r2_train=r2_train,
        grid_search=True,
        best_params={k: (None if v is None else v) for k, v in best_params.items()},
        best_cv_r2=best_cv,
        n_grid_combinations=n_combos,
        cv_folds=cv)
    publish_products(run_id, run_dir(run_id), os.path.basename(model_path),
                     _MANIFEST)

    return {
        "run_id": run_id,
        "model_type": model_type,
        "best_params": best_params,
        "best_cv_r2": best_cv,
        "r2": r2,
        "r2_train": r2_train,
        "model_path": model_path,
        "manifest_path": manifest_path,
    }


# --- MLP hyperparameter scan (5-fold CV) -----------------------------------
#
# mlp_scan is the neural-net counterpart to grid_search: it sweeps a SMALL grid
# of MLP architecture / optimizer hyperparameters with k-fold cross-validation
# on the TRAIN split, refits the best combination on the full train set,
# evaluates on the held-out test split, and saves the result exactly like
# train_mlp (model_mlp.pt + mlp_prep.npz + mlp_arch in the manifest) so
# evaluate_model(run_id, 'mlp') / run_inference_mlp work unchanged on the result.
#
# Adapted from big_datasets/scripts/mlp_grid_scan.py (branch origin/linux-bigdata),
# which scans neurons x num_hidden_layers x lr x weight_decay x pca_var against
# the single train/test split. The differences here: bounded grid (<=6 combos,
# for the LLM-driven agent), k-fold CV for model selection (the script used the
# held-out test split directly -- mildly leaky), and the best model is saved as
# a runnable artifact (the script only wrote a CSV). epochs is deliberately NOT
# searched: a decent max + early stopping (patience/min_delta) already handles
# it, per the project's grid-search plan.

# Tunable hyperparameters exposed to the scan. Anything outside this set is
# rejected. skip_connection is fixed True (the tuned config) and not searched.
_MLP_SCAN_PARAMS: set[str] = {
    "neurons", "num_hidden_layers", "lr", "weight_decay", "pca_var"}

# Defaults for params the caller omits (so a partial grid scans a subset and
# leaves the rest at the tuned train_mlp values). Mirrors train_mlp's config.
_MLP_SCAN_DEFAULTS: dict[str, object] = {
    "neurons": 250, "num_hidden_layers": 1, "lr": 2e-3,
    "weight_decay": 0.2, "pca_var": 0.95,
}

# Fixed iteration order so the grid product is deterministic across calls.
_MLP_SCAN_KEY_ORDER = ("neurons", "num_hidden_layers", "lr",
                      "weight_decay", "pca_var")


def _validate_mlp_grid(
        param_grid: dict[str, list]) -> tuple[dict[str, list], int]:
    """Coerce + validate an MLP scan grid. Returns (clean_grid, n_combos).

    Same caps as grid_search: at most 3 values per hyperparameter and at most
    6 combinations total. Allowed keys: neurons, num_hidden_layers, lr,
    weight_decay, pca_var. Omitted keys are filled from _MLP_SCAN_DEFAULTS
    (each counting as one value, i.e. fixed across the scan), so a partial grid
    like {'neurons':[64,250], 'lr':[2e-3,8e-3]} is a 2x2 = 4-combo scan.
    """
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError(
            "param_grid must be a non-empty dict of hyperparameter -> list of "
            "values, e.g. {'neurons': [64, 250], 'lr': [2e-3, 8e-3]}.")
    unknown = set(param_grid) - _MLP_SCAN_PARAMS
    if unknown:
        raise ValueError(
            f"Unknown hyperparameter(s) for MLP scan: {sorted(unknown)}. "
            f"Allowed: {sorted(_MLP_SCAN_PARAMS)}.")
    clean: dict[str, list] = {}
    n_combos = 1
    for key, vals in param_grid.items():
        values = list(vals)
        if len(values) < 2:
            raise ValueError(
                f"{key}: a scan needs at least 2 values to compare, "
                f"got {len(values)}.")
        if len(values) > _GRID_MAX_VALUES_PER_PARAM:
            raise ValueError(
                f"{key}: at most {_GRID_MAX_VALUES_PER_PARAM} values per "
                f"hyperparameter, got {len(values)}.")
        if key in ("neurons", "num_hidden_layers"):
            values = [int(v) for v in values]
        else:  # lr, weight_decay, pca_var
            values = [float(v) for v in values]
        n_combos *= len(values)
        clean[key] = values
    if n_combos > _GRID_MAX_COMBINATIONS:
        raise ValueError(
            f"param_grid expands to {n_combos} combinations; the limit is "
            f"{_GRID_MAX_COMBINATIONS} total. Drop a value or a hyperparameter.")
    return clean, n_combos


def _mlp_train_once(X_train, y_train, X_val, y_val, combo: dict,
                    batch_size: int, epochs: int, patience: int,
                    min_delta: float, rd_abs: str,
                    classifier: bool = False, num_classes: int = 1) -> dict:
    """Train one MLP for one hyperparameter combo and score it on (X_val, y_val).

    Mirrors the per-combo block in big_datasets/scripts/mlp_grid_scan.py:
    prep_data (aggressive clean -> standardize -> PCA(combo['pca_var'])) fit on
    X_train, MLP_Model(neurons, num_hidden_layers, skip_connection=True), SGD(
    lr, weight_decay) + MSE (NLLLoss + LogSoftmax head when classifier=True)
    + early stopping (patience/min_delta), then evaluate_regression
    (evaluate_classification when classifier=True) for train/val scores — R^2
    for regression, accuracy for classification. SGD divergence (loss -> NaN,
    most often at high lr) is caught and reported as diverged=True rather than
    crashing the scan. Runs from rd_abs so MLP_Model's params file lands in the
    run dir, not the project root. Returns the trained model + fitted prep so
    the caller can either keep them (the final refit) or discard them (CV folds).
    """
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")  # evaluate_regression plots; never pop a GUI window
    import matplotlib.pyplot as plt
    from chemlagent.pytorch_mlp import (
        MLP_Model, prep_data, train, evaluate_regression,
        evaluate_classification)

    cwd = os.getcwd()
    os.chdir(os.path.abspath(rd_abs))
    model = prep = None
    val_score = train_score = float("nan")
    input_dims = None
    epochs_run = 0
    diverged = False
    try:
        prep = prep_data(batch_size=batch_size, shuffle=True,
                         reduce_dim="pca", pca_var=combo["pca_var"],
                         classifier_flag=classifier)
        _, _, train_loader, _ = prep.create_data_loader(
            X_train, y_train, X_val, y_val)
        input_dims = int(prep.X_train.shape[1])
        model = MLP_Model(neurons=combo["neurons"], input_dims=input_dims,
                          num_hidden_layers=combo["num_hidden_layers"],
                          skip_connection=True,
                          classifier_flag=classifier, num_classes=num_classes)
        loss_fn = nn.NLLLoss() if classifier else nn.MSELoss()
        opt = torch.optim.SGD(model.parameters(), lr=combo["lr"],
                              weight_decay=combo["weight_decay"])
        best_loss = float("inf")
        stale = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            epoch_loss = train(train_loader, model, loss_fn, opt,
                               classifier_flag=classifier,
                               num_classes=num_classes)
            if not np.isfinite(epoch_loss):
                diverged = True
                break
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if not diverged:
            if classifier:
                # Class labels: int64 for NLLLoss / accuracy_score.
                train_score, val_score = evaluate_classification(
                    prep.X_train, np.asarray(y_train, dtype=np.int64),
                    prep.X_test, np.asarray(y_val, dtype=np.int64), model)
            else:
                train_score, val_score = evaluate_regression(
                    prep.X_train, np.asarray(y_train, dtype=float),
                    prep.X_test, np.asarray(y_val, dtype=float), model)
            val_score = float(val_score)
            if not np.isfinite(val_score):
                diverged = True
                val_score = float("nan")
        plt.close("all")
    except Exception as exc:
        diverged = True
        if print_flag:
            print(f"  [diverged] {combo}: {exc!r}")
    finally:
        os.chdir(cwd)
    return {"model": model, "prep": prep, "val_score": float(val_score),
            "train_score": float(train_score), "input_dims": input_dims,
            "epochs_run": int(epochs_run), "diverged": bool(diverged)}


def _mlp_cv(combo: dict, X_train, y_train, cv: int, batch_size: int,
            epochs: int, patience: int, min_delta: float, rd_abs: str,
            classifier: bool = False, num_classes: int = 1
            ) -> tuple[float, int]:
    """k-fold CV for one combo. Returns (mean_val_score, n_diverged_folds).

    Each fold fits prep_data on the fold-train, trains the MLP, and scores on
    the held-out fold-val (R^2 for regression, accuracy for classification). A
    diverged fold (SGD NaN) contributes no score but is counted; the combo's
    score is the mean over the non-diverged folds (NaN if every fold diverged).
    Single-threaded (no worker fork) to avoid the Apple-Silicon libomp
    double-load segfault (see memory torch-before-lightgbm).
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    val_scores: list[float] = []
    n_div = 0
    for tr_idx, val_idx in kf.split(X_train):
        res = _mlp_train_once(
            X_train[tr_idx], y_train[tr_idx], X_train[val_idx],
            y_train[val_idx], combo, batch_size, epochs, patience, min_delta,
            rd_abs, classifier=classifier, num_classes=num_classes)
        if res["diverged"] or not np.isfinite(res["val_score"]):
            n_div += 1
        else:
            val_scores.append(res["val_score"])
        # Free the fold's model + prep so 30 CV trainings don't accumulate.
        res["model"] = None
        res["prep"] = None
    mean_score = float(np.mean(val_scores)) if val_scores else float("nan")
    return mean_score, n_div


def mlp_scan(
    run_id: str,
    param_grid: dict[str, list],
    batch_size: int = 32,
    epochs: int = 500,
    patience: int = 75,
    min_delta: float = 1e-5,
    cv_folds: int = 5,
) -> dict:
    """Grid-scan MLP hyperparameters with k-fold CV and save the best fit.

    Sweeps a SMALL grid of MLP architecture / optimizer hyperparameters with
    k-fold (default 5-fold) cross-validation on the TRAIN split only (the
    held-out test split in features.npz is never seen during selection), refits
    the best combination on the full train set, evaluates on the held-out test
    split, and saves the result exactly like train_mlp (model_mlp.pt +
    mlp_prep.npz + mlp_arch in the manifest), so evaluate_model(run_id, 'mlp')
    and run_inference_mlp work unchanged on the result.

    Classification runs (prepare_classes set task='classification'): the scan
    trains the classifier MLP (LogSoftmax head, NLLLoss) and CV/refit scores
    are accuracies; result keys are best_cv_accuracy / accuracy instead of
    best_cv_r2 / r2.

    The grid is capped to keep a tool call bounded: at most 3 values per
    hyperparameter and at most 6 total combinations (a 3x3=9 grid is rejected).
    With 5-fold CV that is at most 30 fold-trainings + 1 refit. Each training is
    single-threaded (no worker fork) to avoid the Apple-Silicon libomp
    double-load segfault. Omitted hyperparameters keep their tuned train_mlp
    defaults (so a partial grid scans a subset).

    Only the following are tunable; unknown ones are rejected:

      - neurons: width of each hidden layer (default 250)
      - num_hidden_layers: extra sigmoid hidden layers beyond the input layer
        (default 1; 0 = input layer only)
      - lr: SGD learning rate (default 2e-3)
      - weight_decay: SGD weight decay (default 0.2)
      - pca_var: PCA explained-variance fraction for input reduction
        (default 0.95)

    epochs is NOT searched: a decent max + early stopping (patience/min_delta)
    already handles it. SGD divergence (loss -> NaN, typically at high lr) is
    caught per fold and counted; a combination whose folds all diverge scores
    NaN and cannot win. If every combination diverges, the call raises with a
    hint to lower lr / weight_decay.

    Use the SAME run_id as the preceding featurize_fingerprints call. This is
    the PyTorch-MLP counterpart of grid_search (which covers random_forest /
    lightgbm / svr).

    Args:
        run_id: Pipeline run identifier; reads features.npz, writes
            mlp_model.pt, mlp_prep.npz, and updates manifest.json.
        param_grid: dict of hyperparameter -> list of values to search. Emit
            as a JSON object with JSON-array values, e.g.
            {"neurons": [64, 250], "lr": [2e-3, 8e-3]}. At most 3 values per key
            and 6 combinations total. Omitted keys keep their defaults.
        batch_size: Training batch size (default 32), fixed across the scan.
        epochs: Max training epochs per fit (default 500). Early stopping
            (patience/min_delta) usually ends each fit well before this.
        patience: Early stopping patience (default 75).
        min_delta: Minimum epoch-loss decrease counted as improvement
            (default 1e-5).
        cv_folds: Number of CV folds for model selection (default 5). Shrunk
            automatically for small train sets so each fold still has enough
            rows.

    Returns:
        dict with keys: run_id, model_type ('mlp'), best_params (dict of the
        winning hyperparameters), best_cv_r2 (mean CV r2 of the winner), r2,
        r2_train (test/train r2 of the refit winner), n_combinations,
        n_diverged_combos, cv_folds, epochs_run, model_path, manifest_path.
    """
    print("MLP scan tool")
    print("=" * 55)
    import itertools
    import time
    import torch

    clean, n_combos = _validate_mlp_grid(param_grid)

    feat_path = os.path.join(run_dir(run_id), _FEATURES)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"{feat_path} not found. Run featurize_fingerprints for run_id "
            f"{run_id!r} first.")
    feats = np.load(feat_path)
    X_train = np.ascontiguousarray(feats["X_train"], dtype=np.float64)
    X_test = np.ascontiguousarray(feats["X_test"], dtype=np.float64)
    # Classification runs (prepare_classes set task='classification'): int
    # class labels + accuracy as the CV/refit score. Regression: float
    # targets + R^2.
    cls = _run_task(run_id) == "classification"
    classifier = num_classes = None
    if cls:
        classifier = True
        num_classes = int(_read_manifest(run_id).get("n_classes", 0))
        if num_classes < 2:
            raise ValueError(
                f"Run {run_id!r} is a classification run but the manifest has "
                f"no n_classes. Run prepare_classes first.")
        y_train = np.asarray(feats["y_train"], dtype=np.int64)
        y_test = np.asarray(feats["y_test"], dtype=np.int64)
    else:
        y_train = np.asarray(feats["y_train"], dtype=float)
        y_test = np.asarray(feats["y_test"], dtype=float)

    n_train = int(y_train.reshape(-1).shape[0])
    cv = int(cv_folds)
    while cv > 2 and n_train // cv < 5:
        cv -= 1

    # Full grid: scanned keys from clean, omitted keys fixed at their defaults.
    grid = {k: clean.get(k, [_MLP_SCAN_DEFAULTS[k]]) for k in _MLP_SCAN_KEY_ORDER}
    combos = [dict(zip(_MLP_SCAN_KEY_ORDER, vals))
              for vals in itertools.product(*(grid[k] for k in _MLP_SCAN_KEY_ORDER))]

    rd_abs = os.path.abspath(run_dir(run_id))
    score_name = "cv_accuracy" if cls else "cv_r2"
    print(f"Scanning MLP over {n_combos} combination(s) "
          f"({cv}-fold CV, {len(combos)} total fits).")

    best = None  # {"combo", "mean_cv_score", "n_diverged"}
    n_diverged_combos = 0
    t0 = time.time()
    for idx, combo in enumerate(combos, 1):
        mean_score, n_div = _mlp_cv(
            combo, X_train, y_train, cv, batch_size, epochs, patience,
            min_delta, rd_abs, classifier=bool(cls), num_classes=num_classes or 1)
        finite = np.isfinite(mean_score)
        if not finite:
            n_diverged_combos += 1
        if finite and (best is None or mean_score > best["mean_cv_score"]):
            best = {"combo": combo, "mean_cv_score": float(mean_score),
                    "n_diverged_folds": n_div}
        if print_flag:
            print(f"[{idx}/{len(combos)}] {combo} -> {score_name}="
                  f"{mean_score if finite else float('nan'):.4f} "
                  f"(diverged folds: {n_div}/{cv})")

    if best is None:
        raise ValueError(
            f"All {n_combos} MLP combination(s) diverged during {cv}-fold CV "
            f"(SGD loss -> NaN). Try a lower lr or smaller weight_decay in "
            f"param_grid.")

    # Refit the winning combo on the full train set and score on the held-out
    # test split. This is the model saved as the run's 'mlp' artifact.
    refit = _mlp_train_once(
        X_train, y_train, X_test, y_test, best["combo"], batch_size, epochs,
        patience, min_delta, rd_abs, classifier=bool(cls),
        num_classes=num_classes or 1)
    if refit["diverged"] or not np.isfinite(refit["val_score"]):
        raise ValueError(
            f"Best-by-CV combination {best['combo']} diverged when refit on "
            f"the full train set. Try a lower lr or smaller weight_decay.")

    model, prep = refit["model"], refit["prep"]
    input_dims = refit["input_dims"]
    test_score = float(refit["val_score"])
    train_score = float(refit["train_score"])
    epochs_run = int(refit["epochs_run"])

    model_path = os.path.join(run_dir(run_id), _MLP_MODEL)
    torch.save(model.state_dict(), model_path)
    prep_path = os.path.join(run_dir(run_id), _MLP_PREP)
    prep.save_stats(prep_path)
    # Free the refit model + prep (state_dict / stats are now on disk).
    del model, prep

    mlp_arch_out = {"neurons": best["combo"]["neurons"],
                    "input_dims": input_dims,
                    "num_hidden_layers": best["combo"]["num_hidden_layers"],
                    "skip_connection": True}
    if cls:
        mlp_arch_out.update({"classifier_flag": True,
                             "num_classes": int(num_classes)})

    if cls:
        manifest_path = _record_model(
            run_id, "mlp", model_path,
            mlp_prep_path=prep_path,
            mlp_arch=mlp_arch_out,
            task="classification",
            accuracy=test_score,
            accuracy_train=train_score,
            epochs_run=epochs_run,
            mlp_scan=True,
            best_params=best["combo"],
            best_cv_accuracy=float(best["mean_cv_score"]),
            n_grid_combinations=int(n_combos),
            cv_folds=int(cv),
            n_diverged_combos=int(n_diverged_combos))
        publish_products(run_id, run_dir(run_id), _MLP_MODEL, _MLP_PREP,
                         _MANIFEST)
        if print_flag:
            print(f"Best params: {best['combo']}  (best_cv_accuracy="
                  f"{best['mean_cv_score']:.4f}, refit test accuracy="
                  f"{test_score:.4f})")
        return {
            "run_id": run_id,
            "model_type": "mlp",
            "best_params": best["combo"],
            "best_cv_accuracy": float(best["mean_cv_score"]),
            "accuracy": test_score,
            "accuracy_train": train_score,
            "n_combinations": int(n_combos),
            "n_diverged_combos": int(n_diverged_combos),
            "cv_folds": int(cv),
            "epochs_run": epochs_run,
            "model_path": model_path,
            "manifest_path": manifest_path,
        }

    manifest_path = _record_model(
        run_id, "mlp", model_path,
        mlp_prep_path=prep_path,
        mlp_arch={"neurons": best["combo"]["neurons"],
                  "input_dims": input_dims,
                  "num_hidden_layers": best["combo"]["num_hidden_layers"],
                  "skip_connection": True},
        r2=test_score,
        r2_train=train_score,
        epochs_run=epochs_run,
        mlp_scan=True,
        best_params=best["combo"],
        best_cv_r2=float(best["mean_cv_score"]),
        n_grid_combinations=int(n_combos),
        cv_folds=int(cv),
        n_diverged_combos=int(n_diverged_combos))
    publish_products(run_id, run_dir(run_id), _MLP_MODEL, _MLP_PREP, _MANIFEST)

    if print_flag:
        print(f"Best params: {best['combo']}  (best_cv_r2="
              f"{best['mean_cv_score']:.4f}, refit test r2={test_score:.4f})")

    return {
        "run_id": run_id,
        "model_type": "mlp",
        "best_params": best["combo"],
        "best_cv_r2": float(best["mean_cv_score"]),
        "r2": test_score,
        "r2_train": train_score,
        "n_combinations": int(n_combos),
        "n_diverged_combos": int(n_diverged_combos),
        "cv_folds": int(cv),
        "epochs_run": epochs_run,
        "model_path": model_path,
        "manifest_path": manifest_path,
    }


def evaluate_model(run_id: str, model_type: str | None = None) -> dict:
    """Evaluate a saved model on the held-out test split.

    Loads a saved model for the run and predicts on the held-out test split in
    features.npz, returning regression metrics plus per-molecule predictions.
    A run can hold several trained models (one per type, each in its own file);
    by default the run's active (most-recently-trained) model is evaluated, or
    pass ``model_type`` to evaluate a specific one -- e.g. after training both
    a LightGBM model and an MLP on the same run, call
    ``evaluate_model(run_id, "lightgbm")`` and ``evaluate_model(run_id, "mlp")``
    to compare them side by side without retraining either.

    Classification runs (prepare_classes set task='classification'): accuracy,
    balanced accuracy, MCC, macro F1 and a row-normalized confusion matrix are
    reported instead of R^2/MAE, and predictions come back as int class
    indices plus their range labels from the manifest.

    Dispatches on the model type:

      - random_forest / lightgbm / svr: unpickle the model's
        runs/<run_id>/model_<type>.pkl and call .predict on the raw test
        features (the sklearn path).
      - mlp: load the torch state_dict (mlp_model.pt) + the saved train-time
        preprocessing stats (mlp_prep.npz), apply that SAME preprocessing to
        the raw test features, and run the MLP forward pass. NB: the test
        features in features.npz are RAW (un-preprocessed) -- train_mlp
        applies cleaning/standardization/PCA internally and only stores the
        r2 in the manifest, so to re-evaluate we must replay that transform
        from the saved stats (mirrors predict_single_value's prep block).
      - chemprop: the held-out test metrics (r2/mae) are computed during
        train_chemprop via a Murcko scaffold split and stored in the
        manifest; they are returned as-is. Per-molecule test predictions are
        not stored, so predictions/truths are empty for this type -- use
        run_inference_chemprop for new molecules.

    Args:
        run_id: Pipeline run identifier; reads model_<type>.pkl / mlp_model.pt
            and features.npz (sklearn/mlp), or manifest.json (chemprop).
        model_type: Optional. Which trained model to evaluate: 'random_forest'
            (aliases 'rf', 'randomforest'), 'lightgbm' (alias 'lgbm'), 'svr',
            'mlp', or 'chemprop'. If omitted, evaluates the run's active
            (most-recently-trained) model. Raises a helpful error naming the
            available types if the requested one was never trained on the run.

    Returns:
        dict with keys: run_id, model_type, r2, mae, n_test, predictions
        (list[float], empty for chemprop), truths (list[float], empty for
        chemprop).
    """
    print("Evaluate model tool")
    print("=" * 55)
    from sklearn.metrics import r2_score, mean_absolute_error

    manifest = _read_manifest(run_id)
    if model_type is None:
        model_type = manifest.get("model_type", "")
        record = manifest  # flat fields are the active model's record
    else:
        model_type = _canonical_model_type(model_type)
        models = manifest.get("models", {})
        if model_type not in models:
            available = sorted(models) or [manifest.get("model_type", "?")]
            raise ValueError(
                f"Run {run_id!r} has no trained {model_type!r} model. "
                f"Available model type(s) for this run: {available}.")
        record = models[model_type]

    feats = np.load(os.path.join(run_dir(run_id), _FEATURES))
    X_test, y_test = feats["X_test"], feats["y_test"]
    # Classification runs (prepare_classes) store int class indices in
    # features.npz; regression stores float targets. The per-model record's
    # own task field wins (mixed regression+classification runs).
    cls = _run_task(run_id, record) == "classification"
    if cls:
        truths = np.asarray(y_test, dtype=np.int64).reshape(-1)
    else:
        truths = np.asarray(y_test, dtype=float).reshape(-1)

    if model_type in ("random_forest", "lightgbm", "svr"):
        with open(record["model_path"], "rb") as fh:
            model = pickle.load(fh)
        preds = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        if cls:
            preds = preds.astype(np.int64)
    elif model_type == "mlp":
        preds = _evaluate_mlp(record, run_dir(run_id), X_test)
    elif model_type == "chemprop":
        # test-split metrics were computed at train time (scaffold split) and
        # stored in the manifest; replay them. Per-molecule predictions are
        # not persisted for chemprop. Classification records accuracy/MCC/F1;
        # regression records r2/mae.
        if cls:
            return {
                "run_id": run_id,
                "model_type": "chemprop",
                "accuracy": float(record.get("accuracy", float("nan"))),
                "balanced_accuracy": float(record.get(
                    "balanced_accuracy", float("nan"))),
                "mcc": float(record.get("mcc", float("nan"))),
                "f1_macro": float(record.get("f1_macro", float("nan"))),
                "n_test": int(record.get("n_test", 0)),
                "predictions": [],
                "truths": [],
            }
        return {
            "run_id": run_id,
            "model_type": "chemprop",
            "r2": float(record.get("r2", float("nan"))),
            "mae": float(record.get("mae", float("nan"))),
            "n_test": int(record.get("n_test", 0)),
            "predictions": [],
            "truths": [],
        }
    else:
        raise ValueError(
            f"Cannot evaluate run {run_id!r}: manifest model_type is "
            f"{model_type!r} (expected random_forest/lightgbm/svr/mlp/"
            f"chemprop). Train a model for this run first.")

    if cls:
        m = classification_metrics(truths, preds)
        pred_labels = class_label_strings(preds,
                                          _read_manifest(run_id)
                                          .get("class_labels", []))
        return {
            "run_id": run_id,
            "model_type": model_type,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "f1_macro": m["f1_macro"],
            "confusion_matrix": m["confusion_matrix"],
            "n_test": m["n_test"],
            "predictions": preds.tolist(),
            "prediction_labels": pred_labels,
            "truths": truths.tolist(),
        }

    return {
        "run_id": run_id,
        "model_type": model_type,
        "r2": float(r2_score(truths, preds)),
        "mae": float(mean_absolute_error(truths, preds)),
        "n_test": int(len(truths)),
        "predictions": preds.tolist(),
        "truths": truths.tolist(),
    }


def _evaluate_mlp(record: dict, rd: str, X_test) -> np.ndarray:
    """Run the saved MLP on the RAW test features and return predictions.

    ``record`` is the MLP's manifest entry -- either the per-type record from
    manifest["models"]["mlp"] (when evaluate_model was given model_type="mlp")
    or the flat manifest itself (when evaluating the active model). Both carry
    mlp_arch / model_path / mlp_prep_path; we fall back to the canonical
    filenames for older runs that predate the per-type records.

    Replays the train-time preprocessing (DescriptorCleaner keep_mask + median
    impute -> standardize with train mean/std -> clip +/-100 -> optional PCA)
    from the saved mlp_prep.npz, loads the torch state_dict, and runs a single
    inference-mode forward pass. Mirrors predict_single_value's prep block so
    the test split is scored exactly as it was during train_mlp.
    """
    import torch
    from chemlagent.pytorch_mlp import MLP_Model, prep_data

    arch = record["mlp_arch"]
    prep_path = record.get("mlp_prep_path") or os.path.join(rd, _MLP_PREP)
    model_path = record.get("model_path") or os.path.join(rd, _MLP_MODEL)
    prep = prep_data.load_stats(prep_path)

    # Apply the saved train-time transform to the raw test features (float64
    # through PCA to avoid overflow on out-of-distribution molecules, then
    # downcast to float32 for the model). See predict_single_value for the
    # identical sequence used at inference time.
    X = np.asarray(X_test, dtype=np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    X = X[:, prep.keep_mask]
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X = X.copy()
        X[inds] = np.take(prep.medians, inds[1])
    fstd = np.asarray(prep.feature_std, dtype=np.float64).copy()
    fstd[fstd == 0] = 1.0
    X = (X - np.asarray(prep.feature_mean, dtype=np.float64)) / fstd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -100.0, 100.0)
    if prep.pca_components is not None:
        comps = np.asarray(prep.pca_components, dtype=np.float64)
        pmean = np.asarray(prep.pca_mean, dtype=np.float64)
        X = (X - pmean) @ comps.T
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))

    # MLP_Model writes a params file to CWD on construction; run from the run
    # dir so it lands with the other artifacts instead of the project root.
    cwd = os.getcwd()
    os.chdir(os.path.abspath(rd))
    try:
        model = MLP_Model(neurons=arch["neurons"], input_dims=arch["input_dims"],
                          num_hidden_layers=arch["num_hidden_layers"],
                          skip_connection=arch["skip_connection"],
                          classifier_flag=bool(arch.get("classifier_flag", False)),
                          num_classes=int(arch.get("num_classes", 1)))
    finally:
        os.chdir(cwd)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    Xt = Xt.to(next(model.parameters()).device)
    model.eval()
    with torch.inference_mode():
        out = model(Xt).cpu().numpy()
    # classifier_flag=True models output (n, num_classes) log-softmax; the
    # prediction is the argmax class index. Regression outputs (n,) values.
    if arch.get("classifier_flag"):
        return out.argmax(axis=1).astype(np.int64)
    return out.reshape(-1)


def run_inference(run_id: str, smiles_list: list[str],
                  model_type: str | None = None) -> dict:
    """Predict the target for novel SMILES using a trained fingerprint model.

    Featurizes the given SMILES with the saved fingerprint estimator, predicts
    with the saved model, and inverts the log10 transform applied during data
    prep if the manifest records log_transformed=True (so predictions come back
    in the original units, e.g. IC50 nM).

    Classification runs (prepare_classes set task='classification'): the saved
    classifier's predictions are int class indices; the run's range labels
    (e.g. '4.1 < 5.6') come back alongside them and no log transform is
    inverted.

    This tool covers the sklearn fingerprint models (random_forest / lightgbm /
    svr). For an MLP use run_inference_mlp; for a Chemprop MPNN use
    run_inference_chemprop.

    Args:
        run_id: Pipeline run identifier; reads model_<type>.pkl,
            featurizer.pkl, and manifest.json.
        smiles_list: List of SMILES strings to predict for.
        model_type: Optional. Which trained sklearn model to predict with:
            'random_forest' (aliases 'rf', 'randomforest'), 'lightgbm' (alias
            'lgbm'), or 'svr'. If omitted, uses the run's active
            (most-recently-trained) model. Raises a helpful error naming the
            available types if the requested one was never trained on the run,
            or points to run_inference_mlp / run_inference_chemprop if a
            non-sklearn type is requested.

    Returns:
        dict with keys: run_id, log_transformed, predictions (list[float], in
        original units), smiles (list[str], the input echoed back).
    """
    print("Run inference tool")
    print("=" * 55)
    manifest = _read_manifest(run_id)
    if model_type is None:
        model_type = manifest.get("model_type", "")
        record = manifest
    else:
        model_type = _canonical_model_type(model_type)
        if model_type not in ("random_forest", "lightgbm", "svr"):
            raise ValueError(
                f"run_inference covers the sklearn fingerprint models only; "
                f"got {model_type!r}. For an MLP use run_inference_mlp, for a "
                f"Chemprop MPNN use run_inference_chemprop.")
        models = manifest.get("models", {})
        if model_type not in models:
            available = sorted(t for t in models
                               if t in ("random_forest", "lightgbm", "svr"))
            raise ValueError(
                f"Run {run_id!r} has no trained {model_type!r} model. "
                f"Available sklearn model type(s): {available or 'none'}.")
        record = models[model_type]
    with open(record["model_path"], "rb") as fh:
        model = pickle.load(fh)
    with open(manifest["featurizer_path"], "rb") as fh:
        featurizer = pickle.load(fh)

    X = featurizer.transform(smiles_list)
    preds = np.asarray(model.predict(X), dtype=float).reshape(-1)

    # Classification runs: predictions are int class indices -> range labels.
    # No log transform / no original-unit inversion for classes.
    cls = _run_task(run_id, record if model_type is not None else manifest) \
        == "classification"
    if cls:
        pred_classes = preds.astype(np.int64)
        pred_labels = class_label_strings(pred_classes,
                                          manifest["class_labels"])
        pred_path = _write_predictions_csv(run_id, smiles_list, pred_classes,
                                           pred_labels)
        return {
            "run_id": run_id,
            "task": "classification",
            "smiles": list(smiles_list),
            "predicted_class": pred_classes.tolist(),
            "predicted_labels": pred_labels,
            "predictions_csv": pred_path,
        }

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

    Classification runs (prepare_classes set task='classification'): trains
    the classifier variant (LogSoftmax head over the run's n_classes, NLLLoss
    on int class labels) and reports accuracy instead of R^2.

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
    from chemlagent.pytorch_mlp import (
        MLP_Model, prep_data, train, evaluate_regression, evaluate_classification)

    feats = np.load(os.path.join(run_dir(run_id), _FEATURES))
    X_train, X_test = feats["X_train"], feats["X_test"]
    y_train, y_test = feats["y_train"], feats["y_test"]

    # prepare_classes set task='classification': train MLP_Model with
    # classifier_flag=True (LogSoftmax head over n_classes). Targets are int
    # class indices and the loss is NLLLoss (CrossEntropyLoss would
    # double-apply the log-softmax). Labels must be int64 for NLLLoss.
    cls = _run_task(run_id) == "classification"
    n_classes = 1
    if cls:
        manifest = _read_manifest(run_id)
        n_classes = int(manifest.get("n_classes", 0))
        if n_classes < 2:
            raise ValueError(
                f"Run {run_id!r} is a classification run but the manifest has "
                f"no n_classes. Run prepare_classes first.")
        y_train = np.asarray(y_train, dtype=np.int64)
        y_test = np.asarray(y_test, dtype=np.int64)

    rd = os.path.abspath(run_dir(run_id))
    cwd = os.getcwd()
    # MLP_Model writes a params file to CWD on construction; run from the run
    # dir so it lands with the other artifacts instead of the project root.
    os.chdir(rd)
    try:
        prep = prep_data(batch_size=32, shuffle=True, reduce_dim="pca",
                         pca_var=0.95, classifier_flag=cls)
        _, _, train_loader, _ = prep.create_data_loader(
            X_train, y_train, X_test, y_test)
        input_dims = int(prep.X_train.shape[1])
        model = MLP_Model(neurons=250, input_dims=input_dims,
                          num_hidden_layers=1, skip_connection=True,
                          classifier_flag=cls, num_classes=n_classes)
        loss_fn = nn.NLLLoss() if cls else nn.MSELoss()
        # Tuned config: SGD, lr=0.002, weight_decay=0.2 (PyTorch wd adds wd*w
        # to the gradient, so wd=0.2 ~ Keras L2=0.1).
        opt = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay=0.2)
        best_loss = float('inf')
        stale = 0
        epochs_run = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            epoch_loss = train(train_loader, model, loss_fn, opt,
                               classifier_flag=cls, num_classes=n_classes)
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

        if cls:
            train_acc, test_acc = evaluate_classification(
                prep.X_train, y_train, prep.X_test, y_test, model)

        else:
            train_r2, test_r2 = evaluate_regression(
                prep.X_train, np.asarray(y_train), prep.X_test,
                np.asarray(y_test), model)

        model_path = os.path.join(rd, _MLP_MODEL)
        torch.save(model.state_dict(), model_path)
        prep_path = os.path.join(rd, _MLP_PREP)
        prep.save_stats(prep_path)
    finally:
        os.chdir(cwd)

    if cls:
        manifest_path = _record_model(
            run_id, "mlp", model_path,
            mlp_prep_path=prep_path,
            mlp_arch={"neurons": 250, "input_dims": input_dims,
                      "num_hidden_layers": 1, "skip_connection": True,
                      "classifier_flag": True, "num_classes": n_classes},
            task="classification",
            accuracy=float(test_acc),
            accuracy_train=float(train_acc),
            epochs_run=int(epochs_run))
        publish_products(run_id, run_dir(run_id), _MLP_MODEL, _MLP_PREP,
                         _MANIFEST)
        return {
            "run_id": run_id,
            "model_type": "mlp",
            "model_path": model_path,
            "prep_path": prep_path,
            "accuracy": float(test_acc),
            "accuracy_train": float(train_acc),
            "epochs_run": int(epochs_run),
            "manifest_path": manifest_path,
        }

    manifest_path = _record_model(
        run_id, "mlp", model_path,
        mlp_prep_path=prep_path,
        mlp_arch={"neurons": 250, "input_dims": input_dims,
                  "num_hidden_layers": 1, "skip_connection": True},
        r2=float(test_r2),
        r2_train=float(train_r2),
        epochs_run=int(epochs_run))
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
    # classifier_flag=True models (mlp_arch records it) predict class indices;
    # num_classes is needed to build the matching output layer.
    cls_flag = bool(arch.get("classifier_flag", False))
    n_classes = int(arch.get("num_classes", 1))

    prep = prep_data.load_stats(os.path.join(rd, _MLP_PREP))
    model = MLP_Model(neurons=arch["neurons"], input_dims=arch["input_dims"],
                      num_hidden_layers=arch["num_hidden_layers"],
                      classifier_flag=cls_flag, num_classes=n_classes,
                      skip_connection=arch["skip_connection"])
    model.load_state_dict(torch.load(os.path.join(rd, _MLP_MODEL),
                                     weights_only=True))
    with open(manifest["featurizer_path"], "rb") as fh:
        featurizer = pickle.load(fh)

    if cls_flag:
        # Classification: argmax over the (n, n_classes) log-softmax outputs.
        # Returns int class indices + the run's range labels; no log inversion.
        preds = predict_single_value(model, featurizer, list(smiles_list),
                                     prep=prep, verbose=False,
                                     classifier_flag=True)
        preds = np.asarray(preds, dtype=np.int64).reshape(-1)
        pred_labels = class_label_strings(preds,
                                          manifest.get("class_labels", []))
        pred_path = _write_predictions_csv(run_id, smiles_list, preds,
                                           pred_labels)
        return {
            "run_id": run_id,
            "task": "classification",
            "smiles": list(smiles_list),
            "predicted_class": preds.tolist(),
            "predicted_labels": pred_labels,
            "predictions_csv": pred_path,
        }

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

    Classification runs (prepare_classes set task='classification'): targets
    are the int class labels and the MPNN gets a MulticlassClassificationFFN
    (CrossEntropyLoss) head; accuracy/balanced-accuracy/MCC/F1 are reported
    instead of r2/mae.

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
    # Classification runs (prepare_classes set task='classification'): targets
    # are the int class labels from data.csv; the MPNN gets a
    # MulticlassClassificationFFN head.
    cls = _run_task(run_id) == "classification"
    if cls:
        targets = df["class"].to_numpy(dtype=np.int64)
        class_labels = _read_manifest(run_id).get("class_labels", [])
        n_classes = int(_read_manifest(run_id).get("n_classes", 0))
    else:
        targets = df["target"].to_numpy(dtype=float)

    tr_smi, tr_y, va_smi, va_y, te_smi, te_y = _scaffold_train_val_test(
        smiles, targets)

    cd = chemprop_data()
    cd.load_pre_split(tr_smi, tr_y, va_smi, va_y, te_smi, te_y)
    scaler = cd.featurize(classification=cls)
    train_loader, val_loader, test_loader = cd.make_loaders(batch_size=batch_size)

    # A from-scratch MPNN only (chemprop's standard small model: d_h=300,
    # depth=3). The CheMeleon foundation-MP path (d_h=2048, depth=6, ~8.7M
    # params) is disabled for now -- it blows up memory on this machine. The
    # loading machinery (chemprop_model._load_foundation_mp / from_foundation)
    # is kept intact for a possible future re-enable; train_chemprop simply
    # does not expose it.
    cm = chemprop_model()
    if cls:
        cm.construct_classifier_model(n_classes)
    else:
        cm.construct_model(scaler)
    cm.train_model(
        train_loader, val_loader, epochs=epochs,
        checkpoint_dir=os.path.join(run_dir(run_id), "checkpoints"),
        accelerator=accelerator)

    if cls:
        # Multiclass head outputs softmax probabilities over n_classes; argmax
        # -> class index; metrics computed post-hoc like the regression r2/mae.
        probs = cm.get_class_probs(test_loader, accelerator=accelerator)
        int_te_y = np.asarray(te_y, dtype=np.int64)
        preds = probs.argmax(axis=1).astype(np.int64)
        m = classification_metrics(int_te_y, preds)
        model_path = os.path.join(run_dir(run_id), _CHEMPROP_MODEL)
        cm.save_model(model_path)

        manifest_path = _record_model(
            run_id, "chemprop", model_path, task="classification",
            n_classes=n_classes,
            accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"],
            mcc=m["mcc"], f1_macro=m["f1_macro"],
            n_train=int(len(tr_smi)),
            n_test=int(len(te_smi)))
        publish_products(run_id, run_dir(run_id), _CHEMPROP_MODEL, _MANIFEST)
        return {
            "run_id": run_id,
            "model_type": "chemprop",
            "model_path": model_path,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "f1_macro": m["f1_macro"],
            "n_classes": n_classes,
            "n_train": int(len(tr_smi)),
            "n_test": int(len(te_smi)),
            "manifest_path": manifest_path,
        }

    preds = np.asarray(cm.get_preds(test_loader, accelerator=accelerator),
                       dtype=float)
    r2 = float(r2_score(te_y, preds))
    mae = float(mean_absolute_error(te_y, preds))

    model_path = os.path.join(run_dir(run_id), _CHEMPROP_MODEL)
    cm.save_model(model_path)

    manifest_path = _record_model(
        run_id, "chemprop", model_path,
        is_foundation=False,
        r2=r2,
        mae=mae,
        n_train=int(len(tr_smi)),
        n_test=int(len(te_smi)))
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

    # Classification runs: the MPNN head is MulticlassClassificationFFN, so
    # predictions are softmax class probabilities -> argmax class index +
    # range labels. No log transform.
    if manifest.get("task") == "classification":
        probs = cm.get_class_probs(loader, accelerator=accelerator)
        pred_classes = probs.argmax(axis=1).astype(np.int64)
        pred_labels = class_label_strings(pred_classes,
                                          manifest.get("class_labels", []))
        pred_path = _write_predictions_csv(run_id, smiles_list, pred_classes,
                                           pred_labels)
        return {
            "run_id": run_id,
            "task": "classification",
            "smiles": list(smiles_list),
            "predicted_class": pred_classes.tolist(),
            "predicted_labels": pred_labels,
            "predictions_csv": pred_path,
        }

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


# --- EDA: dimension reduction + visualization -----------------------------
#
# dimension_reduction is the exploratory step: after featurize_fingerprints it
# reads features.npz, standardizes the combined train+test features, optionally
# drops multivariate outliers, projects to 2D (PCA or t-SNE), and saves a PNG
# scatter colored by the target (continuous), by train/test split, or by
# quantile-binned potency classes. It mirrors the CafChemEDA notebook's
# dimreduction flow but is tool-shaped (returns JSON + a saved PNG, not an
# interactive plt show) and is more robust: t-SNE is pre-reduced with PCA when
# the raw feature space is high-dimensional (the sklearn-recommended pipeline),
# perplexity is clamped to n_samples-1, and outlier removal is guarded so a
# high-feature/low-sample run degrades gracefully instead of crashing.

_DR_METHODS = ("pca", "tsne")
_DR_COLOR_BY = ("target", "split", "classes")


def _elliptic_outliers(X, contamination):
    """Fit EllipticEnvelope on X, return a boolean inlier mask.

    For high-dimensional feature spaces (fingerprints/Mordred, 1000+ columns)
    EllipticEnvelope's MCD robust-covariance estimate is impractically slow
    AND rank-deficient -- molecular fingerprints are sparse and collinear, so
    the covariance is not full rank and sklearn warns/errors. The CafChemEDA
    notebook only ever ran this on 217-dim RDKit descriptors, where it was
    fine; it does not scale to fingerprint widths.

    So when n_features > 50 we first project to a <=50-component PCA subspace
    (randomized SVD, fast) and run MCD there: the PCA components are orthogonal
    (full rank -> no rank warning), and MCD on 50 dims is seconds, not minutes.
    Outlier labels are computed in that subspace and returned as a row mask on
    the original X. The reduction is unsupervised and uses no target, so it
    leaks nothing.

    Guarded: on any remaining failure return all-True (no rows dropped) so the
    EDA plot still renders. MCD's slogdet can also emit numpy linalg
    RuntimeWarnings on near-singular covariance; those are expected and silenced
    here, as is sklearn's "not full rank" UserWarning, so tool output stays clean.
    """
    import warnings
    from sklearn.covariance import EllipticEnvelope
    from sklearn.decomposition import PCA

    n_samples, n_features = X.shape
    Xd = X
    if n_features > 50:
        n_pre = max(2, min(50, n_samples - 1, n_features - 1))
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                Xd = PCA(n_components=n_pre, svd_solver="randomized",
                          random_state=42).fit_transform(X)
        except Exception:
            Xd = X  # fall back to raw; the fit guard below handles failure
        if print_flag:
            print(f"Outlier detection in PCA-reduced {Xd.shape[1]}-dim space "
                  f"(from {n_features}).")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore", message=".*not full rank.*", category=UserWarning)
            det = EllipticEnvelope(contamination=contamination,
                                    support_fraction=None)
            det.fit(Xd)
            return det.predict(Xd) == 1
    except Exception as exc:
        if print_flag:
            print(f"Outlier detection skipped ({exc!r}); keeping all rows.")
        return np.ones(X.shape[0], dtype=bool)


def dimension_reduction(
    run_id: str,
    method: str = "pca",
    color_by: str = "target",
    num_classes: int = 3,
    remove_outliers: bool = True,
    outlier_contamination: float = 0.01,
    perplexity: int = 30,
) -> dict:
    """Reduce the featurized dataset to 2D and save an EDA scatter plot.

    Reads runs/<run_id>/features.npz (the train/test split from
    featurize_fingerprints), standardizes the combined features, optionally
    drops multivariate outliers (EllipticEnvelope), projects to 2 dimensions
    with PCA or t-SNE, and saves a PNG scatter plot under runs/<run_id>/.
    Use this as the exploratory step to see whether the feature space separates
    potency classes or clusters by scaffold, before or after training.

    This reuses the SAME featurization as train_model (features.npz), so call
    featurize_fingerprints first with the SAME run_id. It is unsupervised --
    train and test points are combined for the projection -- so it does not
    leak the test target into anything model-facing.

    Robustness improvements over a plain PCA/t-SNE scatter:
      - t-SNE is run on a <=50-component PCA projection first when the raw
        feature space is high-dimensional (the sklearn-recommended pipeline;
        faster and more stable than t-SNE on 1000+ raw features).
      - perplexity is clamped to n_samples-1 so small runs do not crash.
      - outlier removal is dimension-aware: for high-dimensional feature
        spaces (fingerprints/Mordred, 1000+ columns) EllipticEnvelope's MCD
        robust covariance is run in a <=50-component PCA subspace, not the raw
        space -- MCD on 2048-dim fingerprints is impractically slow and rank-
        deficient. If the fit still fails it is skipped and reported.

    Args:
        run_id: Pipeline run identifier; reads features.npz + manifest.json,
            writes dim_reduction_<method>.png and updates manifest.json.
        method: 'pca' (default) or 'tsne'.
        color_by: How to color points: 'target' (default, continuous color
            gradient by the target value with a colorbar), 'split' (train vs
            test), or 'classes' (target binned into num_classes quantile
            classes, like the EDA notebook's make_classes).
        num_classes: Number of quantile classes when color_by='classes'
            (default 3). Ignored for other color_by values.
        remove_outliers: Drop multivariate outliers (EllipticEnvelope) before
            projecting (default True).
        outlier_contamination: Expected outlier fraction for EllipticEnvelope
            (default 0.01). Ignored when remove_outliers is False.
        perplexity: t-SNE perplexity (default 30). Clamped to n_samples-1.
            Ignored for method='pca'.

    Returns:
        dict with keys: run_id, method, color_by, n_points, n_outliers_removed,
        explained_variance (list of 2 floats, PCA only; empty for t-SNE),
        kl_divergence (float, t-SNE only; NaN for PCA), plot_path,
        manifest_path.
    """
    print("Dimension reduction (EDA) tool")
    print("=" * 55)

    method = method.lower()
    if method not in _DR_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Use 'pca' or 'tsne'.")
    color_by = color_by.lower()
    if color_by not in _DR_COLOR_BY:
        raise ValueError(
            f"Unknown color_by {color_by!r}. Use 'target', 'split', or "
            f"'classes'.")

    feat_path = os.path.join(run_dir(run_id), _FEATURES)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"{feat_path} not found. Run featurize_fingerprints for run_id "
            f"{run_id!r} first.")
    feats = np.load(feat_path)
    X = np.vstack([feats["X_train"], feats["X_test"]]).astype(np.float64)
    y = np.concatenate([feats["y_train"], feats["y_test"]]).astype(float)
    n_train = int(feats["X_train"].shape[0])
    split = np.concatenate([np.zeros(n_train, dtype=int),
                            np.ones(len(y) - n_train, dtype=int)])

    # Replace non-finite values (NaN/inf from some descriptor featurizers)
    # before scaling so StandardScaler / the projectors do not see them.
    X = np.where(np.isfinite(X), X, 0.0)

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    n_outliers = 0
    if remove_outliers:
        inliers = _elliptic_outliers(X, outlier_contamination)
        n_outliers = int((~inliers).sum())
        X, y, split = X[inliers], y[inliers], split[inliers]
    if print_flag:
        print(f"Combined features: {X.shape}; removed {n_outliers} outliers.")

    n_samples, n_features = X.shape

    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=42)
        Z = proj.fit_transform(X)
        explained_variance = [float(v) for v in proj.explained_variance_ratio_]
        kl_divergence = float("nan")
        subtitle = (f"PCA · explained var = {explained_variance[0]:.3f}, "
                    f"{explained_variance[1]:.3f}")
    else:  # tsne
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        # Pre-reduce with PCA when the raw space is high-dimensional: t-SNE
        # on 1000+ features is slow and poorly behaved. 50 comps is the common
        # heuristic; clamp to n_samples-1 and n_features-1 for tiny runs.
        if n_features > 50:
            n_pre = max(2, min(50, n_samples - 1, n_features - 1))
            X = PCA(n_components=n_pre, random_state=42).fit_transform(X)
            if print_flag:
                print(f"t-SNE pre-reduced to {n_pre} PCA components.")
        perp = int(min(perplexity, max(1, n_samples - 1)))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init="pca", learning_rate="auto")
        Z = tsne.fit_transform(X)
        explained_variance = []
        kl_divergence = float(tsne.kl_divergence_)
        subtitle = f"t-SNE · perplexity = {perp} · KL = {kl_divergence:.4f}"

    # --- plot --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # non-interactive CLI; never pop a GUI window
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    manifest = _read_manifest(run_id) if os.path.exists(
        _manifest_path(run_id)) else {}
    units = manifest.get("target_units") or ""
    log_t = manifest.get("log_transformed", False)
    cbar_label = "target"
    if units:
        cbar_label = f"target ({units})"
    if log_t:
        cbar_label = f"log10 {cbar_label}"

    fig, ax = plt.subplots(figsize=(8, 8))
    if color_by == "target":
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=18,
                        edgecolors="none", alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
    elif color_by == "split":
        cmap = ListedColormap(["#1f77b4", "#d62728"])
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=split, cmap=cmap, s=18,
                        edgecolors="none", alpha=0.85)
        ax.legend(handles=sc.legend_elements()[0], labels=["train", "test"],
                  title="split")
    else:  # classes -- quantile-bin the continuous target
        nc = max(2, int(num_classes))
        qs = np.linspace(0, 1, nc + 1)
        edges = np.quantile(y, qs)
        # Open the first/last edges so every point lands in a bin.
        edges[0], edges[-1] = -np.inf, np.inf
        codes = np.searchsorted(edges[1:-1], y, side="right")
        cmap = plt.get_cmap("tab10", nc)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=codes, cmap=cmap, s=18,
                        edgecolors="none", alpha=0.85)
        handles, _ = sc.legend_elements()
        labels = []
        for i in range(nc):
            lo, hi = edges[i], edges[i + 1]
            lo = y.min() if not np.isfinite(lo) else lo
            hi = y.max() if not np.isfinite(hi) else hi
            labels.append(f"{lo:.3g} ≤ {cbar_label} < {hi:.3g}")
        ax.legend(handles, labels, title=f"{nc} classes", fontsize=8)

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"{run_id} · {method.upper()} 2D projection\n{subtitle}",
                fontsize=10)
    fig.tight_layout()

    # Each projection gets its own file keyed by method + the fingerprint type
    # it was built from, so re-featurizing the same run_id with a different
    # fp_type (the "compare fingerprints" workflow) does NOT clobber the prior
    # plot. color_by is a view of the same projection, not a separate result,
    # so it is NOT in the filename -- re-coloring overwrites the same file (the
    # latest coloring wins), and the color_by used is recorded in the manifest
    # entry.
    fp_type = manifest.get("fp_type") or "features"
    variant = f"{method}_{fp_type}"
    plot_path = os.path.join(run_dir(run_id), f"dim_reduction_{variant}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    publish_products(run_id, run_dir(run_id), os.path.basename(plot_path),
                     _MANIFEST)

    manifest = _read_manifest(run_id) if os.path.exists(
        _manifest_path(run_id)) else {}
    eda = manifest.setdefault("eda", {})
    eda[variant] = {
        "method": method,
        "fp_type": fp_type,
        "color_by": color_by,
        "n_points": int(n_samples),
        "n_outliers_removed": int(n_outliers),
        "explained_variance": explained_variance,
        "kl_divergence": kl_divergence,
        "plot_path": plot_path,
    }
    manifest_path = _write_manifest(run_id, manifest)

    return {
        "run_id": run_id,
        "method": method,
        "color_by": color_by,
        "variant": variant,
        "n_points": int(n_samples),
        "n_outliers_removed": int(n_outliers),
        "explained_variance": explained_variance,
        "kl_divergence": kl_divergence,
        "plot_path": plot_path,
        "manifest_path": manifest_path,
    }