"""ChEMBL CSV preparation — the entry point of the CheMLAgent pipeline.

`prepare_chembl_csv` normalizes a bioactivity dataset into a canonical
(SMILES, target) CSV that the rest of the pipeline consumes. It has two input
modes:

  * **local CSV** (``input_csv``): load and normalize a user-supplied file.
    This is how test datasets feed the pipeline without hitting ChEMBL.
  * **ChEMBL fetch** (``chembl_id``): query ChEMBL for IC50 measurements
    (relation "=", units "nM") for a target, merge in canonical SMILES,
    dedupe, drop nulls, sort by potency.

Either way the target may be log10-transformed (IC50 nM -> pIC50-like) so the
downstream regressors see an approximately normal target. The transform is
recorded in the returned summary and the run manifest so inference can invert
it.
"""

# NOTE: no `from __future__ import annotations` — prepare_chembl_csv is exposed
# to Ollama as a tool, and pydantic must resolve its annotations to real objects.

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# ChEMBL client import, done the same way as the original
# modrag_protein_functions.py: imported ONCE at module load (not lazily inside
# each tool call) so it fires at agent startup, not mid-tool-call.
# chembl_webresource_client/__init__ imports pkg_resources for its __version__.
# torch 2.12.1 (pinned here; 2.13 segfaults with chemprop/lightning on MPS) hard
# requires setuptools<82, which still ships pkg_resources, and pkg_resources
# emits a deprecation UserWarning on import (removed in setuptools>=82). The
# other repo gets torch 2.13.0 + setuptools 83 (no pkg_resources) and is
# silent; we keep torch 2.12.1, so we suppress that one warning here — scoped
# to this import and matched to the pkg_resources message only, so nothing
# else is hidden. chembl_flag gates every ChEMBL-using tool so they degrade
# gracefully (print "ChEMBL client not available at this time" and return an
# empty result) instead of raising.
try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        from chembl_webresource_client.new_client import new_client
    chembl_flag = True
except Exception:
    print("Failed to import chembl_webresource_client. "
          "Some functionality may be limited.")
    chembl_flag = False
    new_client = None


def run_dir(run_id: str) -> str:
    """Return (creating) the ``runs/<run_id>/`` artifact directory."""
    path = os.path.join("runs", run_id)
    os.makedirs(path, exist_ok=True)
    return path


# Broadcast print flag: the agent sets chemlagent.data.print_flag = <args.print>
# at startup (see agent.py, mirroring src/agent_template.py). Tool banners and
# status notices always print; this flag gates any future debug/progress
# prints. Default False = debug-silent unless --print.
print_flag = False


# Column-name variants recognized when autodetecting the SMILES / target
# columns of a user-supplied CSV. Matching is case-insensitive and tolerates
# underscores/spaces, so "SMILES", "smiles", "Canonical_SMILES",
# "IsomericSMILES", "smiles_stereo", etc. all resolve without the caller having
# to know the exact header.
_SMILES_HINTS = ("smiles", "smile", "canonical_smi", "isomericsmi")
_TARGET_HINTS = (
    "ic50", "pic50", "ec50", "ki", "kd", "potency", "activity",
    "standard_value", "standard_value_norm", "value", "lmax", "lambda_max",
    "l_max", "lamda_max", "absorption_max", "lambda",
)


def _find_column(df, explicit, hints, label):
    """Return the column name to use, autodetecting if `explicit` is absent.

    If `explicit` names a real column, use it. Otherwise scan `df.columns` for
    the first name that case-insensitively contains any of `hints`. Raises
    ValueError listing the available columns if nothing matches.
    """
    cols = list(df.columns)
    if explicit and explicit in cols:
        return explicit
    norm = {c: c.lower().replace(" ", "") for c in cols}
    for hint in hints:
        for c in cols:
            if hint in norm[c]:
                return c
    raise ValueError(
        f"Could not find a {label} column in {cols!r}. "
        f"Pass {label}_col=<name> explicitly.")


def _fetch_chembl(chembl_id: str, activity_type: str, units: str) -> pd.DataFrame:
    """Query ChEMBL for IC50/Ki/... measurements for one target CHEMBL_ID.

    Returns a DataFrame with columns ``['SMILES', 'target']`` (target in the
    raw units reported by ChEMBL, typically nM). Raises RuntimeError if the
    ChEMBL client is unavailable or the query fails.
    """
    if not chembl_flag:
        raise RuntimeError("ChEMBL client not available at this time")

    chembl_id = chembl_id.upper()
    activity = new_client.activity
    compounds = new_client.molecule

    chosen = activity.filter(
        target_chembl_id=chembl_id, type=activity_type, relation="="
    ).only("molecule_chembl_id", "type", "standard_units",
           "relation", "standard_value")

    mols, vals = [], []
    for record in chosen:
        if record.get("standard_units") == units and record.get("standard_value"):
            mols.append(record["molecule_chembl_id"])
            vals.append(float(record["standard_value"]))

    act_df = pd.DataFrame({"molecule_chembl_id": mols, "target": vals})
    act_df = act_df.drop_duplicates(subset=["molecule_chembl_id"], keep="last")

    if act_df.empty:
        raise RuntimeError(
            f"No {activity_type} measurements (relation '=', units '{units}') "
            f"found in ChEMBL for target {chembl_id}.")

    structs = compounds.filter(
        molecule_chembl_id__in=act_df["molecule_chembl_id"].to_list()
    ).only("molecule_chembl_id", "molecule_structures")

    smi = {}
    for rec in structs:
        ms = rec.get("molecule_structures")
        if ms and ms.get("canonical_smiles"):
            smi[rec["molecule_chembl_id"]] = ms["canonical_smiles"]

    df = act_df.assign(SMILES=act_df["molecule_chembl_id"].map(smi))
    df = df.dropna(subset=["SMILES", "target"]).drop_duplicates(
        subset=["SMILES"], keep="last").sort_values("target")
    return df[["SMILES", "target"]].reset_index(drop=True)


def prepare_chembl_csv(
    chembl_id: Optional[str] = None,
    input_csv: Optional[str] = None,
    run_id: str = "default",
    limit: int = 2000,
    units: str = "nM",
    activity_type: str = "IC50",
) -> dict:
    """Normalize a bioactivity dataset into a canonical (SMILES, target) CSV.

    Exactly one of ``chembl_id`` (fetch from ChEMBL) or ``input_csv`` (load a
    local file) must be given. The result is written to
    ``runs/<run_id>/data.csv`` and a JSON-serializable summary is returned.

    The SMILES and target columns of a local CSV are autodetected from the
    headers (any header containing 'smiles' for SMILES; common activity names
    such as IC50/pIC50/EC50/Ki/Kd/standard_value/activity/Lmax/lambda_max for
    the target). Whether the target is log10-transformed is also autodetected:
    it is logged when the target range (max - min) is >= 1000 (e.g. nM IC50
    spanning orders of magnitude) and left linear otherwise (e.g. nm
    wavelengths). The decision is recorded in the manifest so inference can
    invert the transform.

    Args:
        chembl_id: ChEMBL target ID to fetch bioactives for (e.g. "CHEMBLxxx").
            Mutually exclusive with input_csv.
        input_csv: Path to a local CSV with a SMILES column and a target
            (activity) column. Mutually exclusive with chembl_id.
        run_id: Identifier for this pipeline run; artifacts land under
            runs/<run_id>/.
        limit: If > 0, keep only the first `limit` rows (after sorting by
            potency for a fetch). Default 2000, matching the original pipeline
            cap (modrag_protein_functions sampled 2000 points for large
            ChEMBL dumps). 0 keeps all rows.
        units: Activity units to filter the ChEMBL query by (default 'nM').
            Ignored for input_csv.
        activity_type: Activity type to query (default 'IC50'). Ignored for
            input_csv.

    Returns:
        dict with keys: run_id, csv_path, n_rows, smiles_col, target_col,
        log_transformed, target_units, n_dropped.
    """
    print("Prepare ChEMBL CSV tool")
    print("=" * 55)
    if bool(chembl_id) == bool(input_csv):
        raise ValueError(
            "Provide exactly one of chembl_id (ChEMBL fetch) or input_csv "
            "(local file).")

    if chembl_id and not chembl_flag:
        # Mirrors the original getbioactives_node: if the ChEMBL client never
        # imported, print the friendly banner and return gracefully (no data)
        # instead of raising — lets the model fall back to input_csv.
        print("ChEMBL client not available at this time")
        return {
            "run_id": run_id,
            "error": "ChEMBL client not available at this time",
            "csv_path": None,
            "n_rows": 0,
            "smiles_col": None,
            "target_col": None,
            "log_transformed": False,
            "target_units": units,
            "n_dropped": 0,
        }

    out = os.path.join(run_dir(run_id), "data.csv")

    if input_csv:
        df = pd.read_csv(input_csv)
        smiles_col = _find_column(df, None, _SMILES_HINTS, "SMILES")
        target_col = _find_column(df, None, _TARGET_HINTS, "target")
        df = df.rename(columns={smiles_col: "SMILES", target_col: "target"})
        units = "input_csv"
    else:
        df = _fetch_chembl(chembl_id, activity_type, units)

    n_before = len(df)
    df = df.dropna(subset=["SMILES", "target"])
    df = df.drop_duplicates(subset=["SMILES"], keep="last")
    # Drop non-finite targets always.
    df = df[np.isfinite(df["target"].astype(float))]

    # Autodetect log_transform from the target range: a range >= 1000 signals
    # multi-order-of-magnitude data (e.g. nM IC50) that benefits from a log10
    # transform; a small range (e.g. nm wavelengths) stays linear.
    tvals = df["target"].astype(float)
    log_transform = bool(tvals.max() - tvals.min() >= 1000)

    # log10(<=0) is -inf/nan and poisons training -- the ChEMBL dumps contain
    # IC50=0.0 sentinel rows -- so drop non-positive targets only when logging.
    if log_transform:
        df = df[df["target"].astype(float) > 0]
    n_dropped = n_before - len(df)

    if limit and len(df) > limit:
        df = df.iloc[:limit]

    log_transformed = bool(log_transform)
    if log_transformed:
        df["target"] = np.log10(df["target"].astype(float))

    df = df.reset_index(drop=True)
    df.to_csv(out, index=False)

    summary = {
        "run_id": run_id,
        "csv_path": out,
        "n_rows": int(len(df)),
        "smiles_col": "SMILES",
        "target_col": "target",
        "log_transformed": log_transformed,
        "target_units": units,
        "n_dropped": int(n_dropped),
    }

    # Seed the run manifest so later stages (featurize/train/infer) can read
    # the log_transformed / target_units flags.
    import json
    with open(os.path.join(run_dir(run_id), "manifest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Publish the prepared dataset to recent_work/ so the user can find it.
    from chemlagent.products import publish_products
    publish_products(run_id, run_dir(run_id), "data.csv", "manifest.json")

    return summary