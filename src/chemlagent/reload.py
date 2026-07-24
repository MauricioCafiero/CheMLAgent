"""Deterministic (non-agent) model reload + inference from ``recent_work/``.

After a CheMLAgent session, each run's products sit in
``recent_work/<run_id>/`` (model, featurizer, prepared dataset, predictions,
manifest). This module lets you use them without going through Ollama:

  * :func:`list_available_models` / ``python -m chemlagent.reload list``
    inventory the saved runs (model type, metrics, foundation flag, paths).
  * :func:`LoadedModel.load` / ``python -m chemlagent.reload predict <run_id>
    <SMILES...>`` reload a model and predict for new SMILES, dispatching on the
    manifest's ``model_type`` (random_forest / lightgbm / svr / mlp / chemprop).
    Log-transform inversion is applied per the manifest, so predictions come
    back in original units.

Paths resolve from the ``recent_work/<run_id>/`` directory directly (by the
known filenames for each model type), so this works whether or not the
manifest's path fields were rewritten at quit.
"""

# Import torch before any chemlagent import that could pull lightgbm/skfp, to
# avoid the Apple-Silicon libomp double-load segfault (see tools.py). Guarded so
# a sklearn-only reload still works in an env without torch.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from chemlagent.products import RECENT_WORK_DIR, recent_work_path

# Filenames per model type, mirroring what the tools save + relocate.
_SKLEAN_FILES = ("model.pkl", "featurizer.pkl")
_MLP_FILES = ("mlp_model.pt", "mlp_prep.npz", "featurizer.pkl")
_CHEMPROP_FILES = ("chemprop_model.pt",)


def list_available_models(root: str | None = None) -> list[dict]:
    """Inventory the saved runs under ``recent_work/``.

    Returns one dict per run (sorted by run_id) with: run_id, model_type,
    is_foundation, fp_type, r2, mae, n_train, n_test, log_transformed, and the
    absolute directory path. Runs without a manifest are skipped.
    """
    base = recent_work_path() if root is None else root
    out: list[dict] = []
    if not os.path.isdir(base):
        return out
    for run_id in sorted(os.listdir(base)):
        rd = os.path.join(base, run_id)
        mpath = os.path.join(rd, "manifest.json")
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath) as fh:
                m = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "run_id": run_id,
            "model_type": m.get("model_type"),
            "is_foundation": m.get("is_foundation"),
            "fp_type": m.get("fp_type"),
            "r2": m.get("r2"),
            "mae": m.get("mae"),
            "n_train": m.get("n_train"),
            "n_test": m.get("n_test"),
            "log_transformed": m.get("log_transformed"),
            "dir": os.path.abspath(rd),
        })
    return out


@dataclass
class LoadedModel:
    """A reloaded model ready for ``.predict(smiles_list)``.

    Construct via :meth:`load` (which reads the manifest and loads the right
    artifacts by model type); don't build the fields by hand.
    """
    run_id: str
    model_type: str
    log_transformed: bool
    run_dir: str
    model: object = None
    featurizer: object = None
    prep: object = None
    arch: dict = field(default_factory=dict)

    @classmethod
    def load(cls, run_id: str, root: str | None = None) -> "LoadedModel":
        """Load a saved run from ``recent_work/<run_id>/`` by its model type."""
        rd = os.path.join(recent_work_path() if root is None else root, run_id)
        if not os.path.isdir(rd):
            raise FileNotFoundError(
                f"No saved run {run_id!r} in {os.path.abspath(recent_work_path())}. "
                f"Use list_available_models() to see what's available.")
        with open(os.path.join(rd, "manifest.json")) as fh:
            manifest = json.load(fh)
        model_type = manifest.get("model_type")
        if not model_type:
            raise ValueError(f"Run {run_id!r} has no trained model in its manifest.")
        log_transformed = bool(manifest.get("log_transformed", False))

        self = cls(run_id=run_id, model_type=model_type,
                   log_transformed=log_transformed, run_dir=rd)

        if model_type in ("random_forest", "lightgbm", "svr"):
            self._load_sklearn(rd)
        elif model_type == "mlp":
            self.arch = manifest.get("mlp_arch", {})
            self._load_mlp(rd)
        elif model_type == "chemprop":
            self._load_chemprop(rd)
        else:
            raise ValueError(f"Unknown model_type {model_type!r} for run {run_id!r}.")
        return self

    # -- per-type loaders ----------------------------------------------------

    def _load_sklearn(self, rd: str) -> None:
        with open(os.path.join(rd, "model.pkl"), "rb") as fh:
            self.model = pickle.load(fh)
        with open(os.path.join(rd, "featurizer.pkl"), "rb") as fh:
            self.featurizer = pickle.load(fh)

    def _load_mlp(self, rd: str) -> None:
        from chemlagent.pytorch_mlp import MLP_Model, prep_data
        arch = self.arch
        prep = prep_data.load_stats(os.path.join(rd, "mlp_prep.npz"))
        # MLP_Model writes a params file to CWD on construction; run from the
        # run dir so it lands with the other artifacts instead of the CWD.
        cwd = os.getcwd()
        os.chdir(rd)
        try:
            model = MLP_Model(neurons=arch["neurons"], input_dims=arch["input_dims"],
                              num_hidden_layers=arch["num_hidden_layers"],
                              skip_connection=arch["skip_connection"])
        finally:
            os.chdir(cwd)
        model.load_state_dict(torch.load(os.path.join(rd, "mlp_model.pt"),
                                         weights_only=True))
        with open(os.path.join(rd, "featurizer.pkl"), "rb") as fh:
            featurizer = pickle.load(fh)
        self.model, self.featurizer, self.prep = model, featurizer, prep

    def _load_chemprop(self, rd: str) -> None:
        from chemprop import featurizers
        from chemprop.models.model import MPNN
        from chemlagent.chemprop_model import chemprop_data, chemprop_model
        cd = chemprop_data()
        cd.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        cm = chemprop_model()
        cm.mpnn = MPNN.load_from_file(os.path.join(rd, "chemprop_model.pt"))
        self.model = cm
        self.featurizer = cd  # carries the featurizer; used in predict()

    # -- inference -----------------------------------------------------------

    def predict(self, smiles_list: list[str]) -> np.ndarray:
        """Predict the target for SMILES in original units (log inverted)."""
        smiles_list = list(smiles_list)
        if self.model_type in ("random_forest", "lightgbm", "svr"):
            X = self.featurizer.transform(smiles_list)
            preds = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        elif self.model_type == "mlp":
            from chemlagent.pytorch_mlp import predict_single_value
            inverse = (lambda p: np.power(10.0, p)) if self.log_transformed else None
            preds = np.asarray(
                predict_single_value(self.model, self.featurizer, smiles_list,
                                     prep=self.prep, inverse_func=inverse,
                                     verbose=False), dtype=float).reshape(-1)
            return preds  # already in original units
        elif self.model_type == "chemprop":
            loader = self.featurizer.make_new_dataloader(smiles_list, batch_size=64)
            preds = np.asarray(self.model.get_preds(loader), dtype=float).reshape(-1)
        else:
            raise ValueError(f"Unknown model_type {self.model_type!r}")
        if self.log_transformed:
            preds = np.power(10.0, preds)
        return preds


# --- CLI -------------------------------------------------------------------

def _print_table(models: list[dict]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        for m in models:
            print(m)
        return
    console = Console()
    if not models:
        console.print("[dim]No saved models in recent_work/.[/dim]")
        return
    table = Table(title="Saved models in recent_work/", show_lines=False)
    for col in ("run_id", "model_type", "fp_type", "is_foundation",
                "r2", "mae", "n_train", "n_test", "log_transformed"):
        table.add_column(col)
    for m in models:
        table.add_row(
            str(m["run_id"]), str(m["model_type"]), str(m.get("fp_type") or "-"),
            str(m.get("is_foundation") if m.get("is_foundation") is not None else "-"),
            f"{m['r2']:.3f}" if m.get("r2") is not None else "-",
            f"{m['mae']:.2f}" if m.get("mae") is not None else "-",
            str(m.get("n_train") or "-"), str(m.get("n_test") or "-"),
            str(m.get("log_transformed")))
    console.print(table)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Deterministic model reload + inference from recent_work/.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list saved models in recent_work/")
    p = sub.add_parser("predict", help="predict with a saved model")
    p.add_argument("run_id")
    p.add_argument("smiles", nargs="*", help="SMILES to predict (space-separated)")
    p.add_argument("--csv", help="read SMILES from a CSV (uses --smiles-col)")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--out", help="write predictions CSV here")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        _print_table(list_available_models())
        return 0
    if args.cmd == "predict":
        smiles = list(args.smiles)
        if args.csv:
            df = pd.read_csv(args.csv)
            col = args.smiles_col if args.smiles_col in df.columns else next(
                (c for c in df.columns if "smiles" in c.lower()), args.smiles_col)
            smiles = df[col].astype(str).tolist()
        if not smiles:
            parser.error("provide SMILES on the CLI or with --csv")
        lm = LoadedModel.load(args.run_id)
        preds = lm.predict(smiles)
        out = pd.DataFrame({"SMILES": smiles, "prediction": preds})
        if args.out:
            out.to_csv(args.out, index=False)
            print(f"wrote {len(out)} predictions to {args.out}")
        else:
            print(out.to_string(index=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())