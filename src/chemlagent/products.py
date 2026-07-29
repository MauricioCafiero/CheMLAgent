"""Collect a run's product files into a visible ``recent_work/`` folder.

As each pipeline stage completes it calls :func:`publish_products` to copy its
user-facing outputs (the prepared dataset, trained models, inference CSVs, the
manifest) from ``runs/<run_id>/`` into ``recent_work/<run_id>/``. The working
files stay in ``runs/`` so the pipeline can keep reading them mid-session; the
copies in ``recent_work/`` are the curated, easy-to-find products. The agent
CLI lists ``recent_work/`` on quit so the user knows where their results are.
"""

from __future__ import annotations

import json
import os
import shutil

RECENT_WORK_DIR = "recent_work"
RUNS_DIR = "runs"

# Files worth keeping as products, by model type. Everything else under a run
# dir (feature matrices, the featurizer pickle for chemprop, lightning
# checkpoint dirs) is an intermediate and is discarded at quit. The sklearn
# models each write their own model_<type>.pkl (see tools._record_model), so a
# run can hold several of them at once; relocate_all_runs keeps the union over
# every trained type, not just the active one.
_KEEP_BY_MODEL: dict[str, tuple[str, ...]] = {
    "random_forest": ("model_random_forest.pkl", "featurizer.pkl"),
    "lightgbm": ("model_lightgbm.pkl", "featurizer.pkl"),
    "svr": ("model_svr.pkl", "featurizer.pkl"),
    "mlp": ("mlp_model.pt", "mlp_prep.npz", "featurizer.pkl"),
    "chemprop": ("chemprop_model.pt",),
}
# Always-kept products regardless of model type.
_ALWAYS_KEEP = ("data.csv", "predictions.csv", "manifest.json")

# Manifest fields that hold a path to a product file. On relocate these are
# rewritten to the file's recent_work/<id>/ location (only when that file was
# actually moved there), so a manifest saved post-quit still points at real
# files. features_path is deliberately NOT rewritten -- features.npz is an
# intermediate that's discarded, so the field is left pointing at the (now-gone)
# runs/ path rather than at a nonexistent recent_work file.
_PATH_FIELDS = ("csv_path", "featurizer_path", "model_path", "mlp_prep_path")


def recent_work_path(run_id: str | None = None) -> str:
    """Path to the recent-work folder, or a run's subfolder if run_id is given."""
    return os.path.join(RECENT_WORK_DIR, run_id) if run_id else RECENT_WORK_DIR


def publish_products(run_id: str, run_root: str, *names: str) -> list[str]:
    """Copy product files from ``run_root`` into ``recent_work/<run_id>/``.

    Only files that exist are copied (so a caller can pass several candidate
    names, e.g. the model file which varies by model type). Returns the
    destination paths of the files actually copied, newest-last so a later
    manifest copy overwrites an earlier one.
    """
    dst = recent_work_path(run_id)
    os.makedirs(dst, exist_ok=True)
    copied: list[str] = []
    for name in names:
        src = os.path.join(run_root, name)
        if os.path.exists(src):
            dest = os.path.join(dst, name)
            shutil.copy2(src, dest)
            copied.append(dest)
    return copied


def list_recent_work(root: str | None = None) -> list[str]:
    """Return absolute paths of every file under the recent-work folder.

    Empty list if the folder does not exist or is empty. Used by the agent CLI
    to remind the user where their products are on quit.
    """
    base = os.path.abspath(recent_work_path() if root is None else root)
    if not os.path.isdir(base):
        return []
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


def relocate_all_runs(runs_root: str = RUNS_DIR) -> list[str]:
    """At quit: move each run's product files into ``recent_work/<id>/`` and
    delete the rest of ``runs/<id>/`` (intermediates) plus the run dir itself.

    The keep-set is chosen from the run's ``manifest.json`` model_type (the
    prepared dataset, the trained model + any files needed to reload it, the
    predictions CSV, and the manifest). Loose files at the runs/ root (e.g. the
    CheMeleon foundation cache) are left in place. Returns the destination
    paths of files moved this call.
    """
    moved: list[str] = []
    if not os.path.isdir(runs_root):
        return moved
    for run_id in sorted(os.listdir(runs_root)):
        run_path = os.path.join(runs_root, run_id)
        if not os.path.isdir(run_path):
            continue  # skip loose files (e.g. the foundation-model cache)
        data: dict = {}
        model_type = None
        mpath = os.path.join(run_path, "manifest.json")
        if os.path.exists(mpath):
            try:
                with open(mpath) as fh:
                    data = json.load(fh)
                    model_type = data.get("model_type")
            except (OSError, json.JSONDecodeError):
                data, model_type = {}, None

        # Keep the union of product files across EVERY model trained on the run
        # (manifest["models"]), not just the active one, so training several
        # models on one run doesn't strand the non-active ones' files as
        # "intermediates" at quit. Per-type records carry their own model_path /
        # mlp_prep_path, which we add explicitly; _KEEP_BY_MODEL covers the
        # type's other known files (featurizer.pkl etc.).
        keep = set(_ALWAYS_KEEP)
        types = set(data.get("models", {}).keys())
        if model_type:
            types.add(model_type)
        for t in types:
            rec = data.get("models", {}).get(t)
            if rec:
                for f in ("model_path", "mlp_prep_path"):
                    p = rec.get(f)
                    if p:
                        keep.add(os.path.basename(str(p)))
            keep.update(_KEEP_BY_MODEL.get(t, ()))
        # Always keep the active model_path basename too -- older runs predate
        # per-type files and saved a single model.pkl, which the loop above
        # would otherwise miss. Harmless (set) when it's already covered.
        active_path = data.get("model_path")
        if active_path:
            keep.add(os.path.basename(str(active_path)))

        dst_dir = recent_work_path(run_id)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in sorted(keep):
            src = os.path.join(run_path, fn)
            if os.path.exists(src):
                dest = os.path.join(dst_dir, fn)
                if os.path.exists(dest):
                    os.remove(dest)  # overwrite the mid-session copy
                shutil.move(src, dest)
                moved.append(dest)

        # Rewrite the manifest's product path fields to their recent_work
        # locations, so the saved manifest still resolves to real files after
        # runs/<id>/ is deleted. Rewrite both the flat active-model fields and
        # each per-type record's path fields.
        mdest = os.path.join(dst_dir, "manifest.json")
        if os.path.exists(mdest):
            try:
                with open(mdest) as fh:
                    data = json.load(fh)

                def _rewrite(obj: dict) -> None:
                    for field in _PATH_FIELDS:
                        if field in obj:
                            base = os.path.basename(str(obj[field]))
                            if os.path.exists(os.path.join(dst_dir, base)):
                                obj[field] = os.path.join(
                                    RECENT_WORK_DIR, run_id, base)

                _rewrite(data)
                for rec in data.get("models", {}).values():
                    if isinstance(rec, dict):
                        _rewrite(rec)
                with open(mdest, "w") as fh:
                    json.dump(data, fh, indent=2)
            except (OSError, json.JSONDecodeError):
                pass

        # Discard intermediates (features.npz, checkpoints/, etc.) and the
        # now-empty run dir. The products live on in recent_work/<id>/.
        shutil.rmtree(run_path, ignore_errors=True)
    return moved