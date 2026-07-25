# CheMLAgent

![CheMLAgent](chemlagent.png)

An agentic system for building and training ML models for chemistry (QSAR /
bioactivity / property prediction), driven by **Ollama tool calls**.

A chat model is handed a small set of plain Python functions — discover a
target, prepare a CSV, featurize SMILES, train a model, evaluate it, run
inference — and decides which to call to satisfy a modeling request. Ollama
infers each tool's JSON schema from the function signature + docstring, so the
"agent" is just a thin dispatch loop over real, reusable chemistry code.

## Pipeline

```
[discovery, optional]
  search_targets   (disease → gene symbols)
  search_uniprot   (protein name → UniProt IDs)
  list_bioactives  (UniProt → ChEMBL target ID + counts)
        │
        ▼
prepare_chembl_csv  ──►  featurize_fingerprints  ──►  train_model   ──►  evaluate_model
  (CSV / ChEMBL ID)        (fingerprints /           (RF / LightGBM /      (held-out test)
  ← or input_csv directly   descriptors)              SVR)
                                                   ──►  train_mlp          ──►  run_inference(_mlp/_chemprop)
                                                   ──►  train_chemprop
                                                        (MPNN on mol graphs)
```

The discovery tools are optional — use them when the request gives a protein
name or disease rather than a ChEMBL ID. Once you have a `chembl_id`,
`prepare_chembl_csv` fetches and normalizes the dataset (give it a local
`input_csv` instead to skip discovery and fetch entirely).

State between tool calls lives on disk under `runs/<run_id>/` (fitted models,
featurizers, feature matrices, a `manifest.json`), referenced by path in the
JSON returned to the model — Ollama tool calls exchange JSON, not live objects.
Thread the same `run_id` through every stage of one task.

As each stage completes, its user-facing products (the prepared `data.csv`,
trained model, inference `predictions.csv`, and `manifest.json`) are copied into
`recent_work/<run_id>/` so they're easy to find mid-session. On quit the CLI
**moves** the final products into `recent_work/<run_id>/`, deletes the
intermediate scratch left in `runs/<run_id>/` (feature matrices, lightning
checkpoints, the run dir itself), and prints the absolute paths of the
**runs produced this session only** — runs already in `recent_work/` from
previous sessions are not listed. (The CheMeleon foundation path is
currently inactive — see [CheMeleon foundation model](#chemeleon-foundation-model)
— so no `chemeleon_mp.pt` cache is produced or moved.)

## What it can do

- **Discover a target** — when you only have a protein name or a disease (not a
  ChEMBL ID), `search_targets` lists proteins associated with a disease (Open
  Targets), `search_uniprot` resolves a protein/gene name to UniProt accessions
  (optionally human-only), and `list_bioactives` maps UniProt IDs to ChEMBL
  target IDs with their activity counts so you can pick the richest target.
- **Prepare data** — read a local CSV, or fetch from ChEMBL by target ID
  (either given directly or found via the discovery tools).
  SMILES/target columns are autodetected (handles `SMILES`, `smiles`,
  `canonical_smi`, etc., and `IC50`/`pIC50`/`EC50`/`Ki`/`Kd`/`Lmax`/`λ_max`/
  `potency`/…). A log10 transform is autodetected when the target range is
  ≥ 1000 (e.g. nM IC50); inference inverts it automatically.
- **Featurize** — fingerprints & descriptors via scikit-fingerprints, then a
  Murcko-scaffold train/test split. 2D types run on the molecular graph; 3D
  types generate conformers first. Pass `fp_type` (case-insensitive) to
  `featurize_fingerprints`.

  | `fp_type`        | family       | description                                            |
  |------------------|--------------|--------------------------------------------------------|
  | `ECFP`           | fingerprint  | Morgan/ECFP circular fingerprint (counted)             |
  | `Atom_Pair`      | fingerprint  | topological atom-pair fingerprint                       |
  | `MACCS`          | fingerprint  | 166-bit MACCS structural keys                           |
  | `PubChem`        | fingerprint  | PubChem 881-bit fingerprint                            |
  | `Functional_Groups` | fingerprint | functional-group presence bits                       |
  | `RDKitFingerprint` | fingerprint | RDKit path fingerprint                                 |
  | `Mordred`        | descriptor   | Mordred 2D/3D descriptor set (NaNs imputed downstream) |
  | `RDKit_2D`       | descriptor   | RDKit's native ~217 continuous 2D descriptors           |
  | `EState`         | descriptor   | 79 Kier–Hall electrotopological-state descriptors      |
  | `E3FP`           | 3D fingerprint | 3D fingerprint from generated conformers             |
  | `Autocorr`       | 3D descriptor | 3D autocorrelation descriptors                        |
  | `MORSE`          | 3D descriptor | 3D Molecule Representation of Structures based on Electron diffraction |
  | `RDF`            | 3D descriptor | radial distribution function descriptors              |
  | *(molecular graph)* | graph     | not an `fp_type` — `train_chemprop` builds the graph directly from `data.csv` (SMILES → atom/bond graph via chemprop) |

- **Train** — pick one model per run (same `run_id`). Fingerprint/descriptor
  models take the featurized matrix; Chemprop reads `data.csv` directly and
  trains on molecular graphs (no `featurize_fingerprints` call needed).

  | tool              | model                          | input              | notes                                                   |
  |-------------------|--------------------------------|--------------------|---------------------------------------------------------|
  | `train_model`     | Random Forest                  | fingerprints/descriptors | `model_type=random_forest`, `n_estimators` tunable |
  | `train_model`     | LightGBM                       | fingerprints/descriptors | `model_type=lightgbm`                               |
  | `train_model`     | SVR                            | fingerprints/descriptors | `model_type=svr`, literature-tuned poly kernel     |
  | `train_mlp`       | PyTorch wide-and-deep MLP      | fingerprints      | SGD/lr/weight-decay/batch tuned internally             |
  | `train_chemprop`  | Chemprop MPNN                  | molecular graphs  | from-scratch MPNN only (CheMeleon foundation wired but inactive — see below) |
- **Evaluate & infer** — metrics on the held-out split; predictions for new
  SMILES in original units.

## Requirements

- **Python 3.14** (pinned in `pyproject.toml`). The cp312 torch wheel segfaults
  inside chemprop/lightning training; the cp314 wheel trains on Apple MPS
  without issue, so the project requires 3.14.
- [uv](https://docs.astral.sh/uv/) for environment management.
- An Ollama endpoint (cloud or local) with a chat model that supports tool
  calls, e.g. `glm-5.2`, `gemma3:27b`, `qwen2.5`.

## Install

Clone from GitHub, then create the environment and sync:

```bash
git clone https://github.com/MauricioCafiero/CheMLAgent.git
cd CheMLAgent
uv venv --python 3.14
uv sync --all-extras      # base + mlp + chemprop + chembl
# or pick extras:  uv sync --extra mlp --extra chemprop --extra chembl
```

Extras:

| extra       | brings in                              |
|-------------|----------------------------------------|
| `mlp`       | torch, matplotlib (PyTorch MLP)        |
| `chemprop`  | chemprop, lightning, torch (MPNN)      |
| `chembl`    | chembl_webresource_client (ChEMBL fetch)|
| `all`       | all of the above                       |
| `dev`       | pytest                                 |

Configure the Ollama client via environment variables:

```bash
export OLLAMA_HOST=https://ollama.com   # or http://localhost:11434 for local
export OLLAMA_API_KEY=...               # only if your host requires it
```

## Usage

Interactive REPL:

```bash
uv run python -m chemlagent.agent --print --model glm-5.2
```

`--print` shows the model's thinking, tool calls, and results; `--model` selects
the chat model (defaults to the first of `DEFAULT_MODELS` in `agent.py`).

### REPL keywords

The REPL also takes deterministic keywords that skip the chat model and act
directly on saved runs (the banner lists these at startup):

| keyword                                       | action                                            |
|-----------------------------------------------|---------------------------------------------------|
| `quit` / `exit`                               | exit; move products to `recent_work/`, print paths |
| `/models`                                     | list saved runs, then pick one (row number or `run_id`) to load |
| `/predict <SMILES ...>`                       | inference for inline SMILES with the loaded model |
| `/predict --csv <f> [--smiles-col C] [--out o.csv]` | batch: load SMILES from a CSV, predict with the loaded model |
| `/predict <run_id> <SMILES ...>`              | load that run, then predict (becomes the loaded model) |
| `/help`                                       | show the keywords                                 |

The flow is **load-then-predict**: `/models` lists the saved runs in
`recent_work/` and loads the one you pick; it stays loaded (the prompt shows
`[run_id]` when one is active) so later `/predict` calls reuse it with inline
SMILES or `--csv` — no `run_id` needed. `/predict <run_id> …` loads a run and
predicts in one step. These reuse the deterministic reload code
(`chemlagent.reload`), so no Ollama call is involved. Anything that isn't a
keyword is sent to the model as your request.

### Example prompt

> Build a QSAR model for the human MAO-B protein. I don't have a ChEMBL ID —
> discover one: search UniProt for `MAOB` (human only), then list the bioactives
> to find the ChEMBL target with the most IC50s. Prepare a 600-row dataset
> (`run_id=maob`), featurize with ECFP, and train a random forest. Report the
> ChEMBL ID and test R².

This drives the full discovery → prepare → featurize → train chain. On a live
run it resolves MAO-B → UniProt P27338 → CHEMBL2039 (5,751 IC50s), trains on 600
rows, and reports R² ≈ 0.18 on the held-out scaffold split (overfits at 600
rows; `limit=0` for all 5,751 generalizes better).

Or, with a local CSV (no discovery / fetch):

> Use the local CSV `621-azo.csv` (SMILES column `SMILES`, target column `Lmax`).
> Prepare it with `run_id=azo`, train a chemprop model (`epochs=15`),
> then predict λ_max for `c1ccc(/N=N/c2ccccc2)cc1` and
> `C[N]1N=NC(=N1)N=NC2=CC=CC=C2`. Report test R², MAE, and the predictions in nm.

On the azo dataset this yields ~R² 0.90, ~MAE 15 nm, and predicts azobenzene at
~320 nm (matching its experimental π→π\* band). *(These numbers were observed
when the CheMeleon foundation path was still active; with the from-scratch
MPNN that `train_chemprop` now uses, expect somewhat lower R² at the same
epoch count.)*

## Tools

Each is a plain function in `src/chemlagent/tools.py` with a numpydoc docstring
that doubles as the Ollama tool schema.

| tool                     | args (besides `run_id`)                                   |
|--------------------------|-----------------------------------------------------------|
| `search_targets`         | `disease_names`                                           |
| `search_uniprot`         | `protein_names`; `human_only` (default `False`)           |
| `list_bioactives`        | `uniprot_ids`; `activity_type` (default `IC50`)           |
| `prepare_chembl_csv`     | `chembl_id` *or* `input_csv`; `limit`, `units`, `activity_type` |
| `featurize_fingerprints` | `fp_type` (default `ECFP`), `test_size`                   |
| `train_model`            | `model_type` (`random_forest`/`lightgbm`/`svr`), `n_estimators` |
| `evaluate_model`         | —                                                         |
| `run_inference`          | `smiles_list`                                             |
| `train_mlp`              | `epochs` (default 2500, the tuned value)                  |
| `run_inference_mlp`      | `smiles_list`                                             |
| `train_chemprop`         | `epochs` (default 30), `batch_size`, `accelerator`       |
| `run_inference_chemprop` | `smiles_list`                                             |

Arguments are deliberately minimal to keep tool calls well-formed. Hyperparameters
that aren't exposed use tuned internal defaults (e.g. the SVR poly kernel, the
MLP's SGD/lr/weight-decay/batch config, chemprop's Noam-style LR schedule).

### CheMeleon foundation model

`train_chemprop` is wired to initialize its message-passing block from the
[CheMeleon](https://arxiv.org/abs/2506.15792) pretrained weights (downloaded
once to `runs/chemeleon_mp.pt`) and fine-tune a fresh regression head —
transfer learning that converges in a few epochs. **This path is currently
inactive, however.** The pretrained message-passer is large (`d_h=2048`,
`depth=6`, ~8.7M parameters) and blows up memory on this machine, so on
2026-07-24 the `foundation` argument was removed from `train_chemprop` and it
now trains only chemprop's standard from-scratch MPNN (`d_h=300`, `depth=3`,
~30× smaller). The foundation loading machinery
(`chemprop_model._load_foundation_mp`, the `from_foundation` / `foundation_path`
constructor path, and the `_FOUNDATION_CACHE = runs/chemeleon_mp.pt` constant
in `tools.py`) is preserved intact so the path can be re-enabled later by
re-adding a `foundation` flag that branches to `chemprop_model(from_foundation="chemeleon", foundation_path=_FOUNDATION_CACHE)`. Until then, no
`chemeleon_mp.pt` cache is downloaded or written, and the manifest's
`is_foundation` is always `False`.

## Where input CSVs come from

`prepare_chembl_csv` takes a path in its `input_csv` argument and reads it
**as given, relative to your current working directory** (it does not search
anywhere). When you launch the agent from the project root —
`uv run python -m chemlagent.agent` — that CWD *is* the repo root, so a CSV
dropped there (e.g. the bundled `621-azo.csv`, `CHEMBL220_bioactives.csv`) is
reachable by bare filename: `input_csv="621-azo.csv"`. Absolute paths work too,
for CSVs living elsewhere. If `input_csv` is omitted, pass a `chembl_id` to
fetch from ChEMBL instead.

## What an inference call needs

Inference needs **only a `run_id` plus a list of SMILES strings — no CSV**.
The model + featurizer (and, for the MLP, the preprocessing stats) are reloaded
from the saved run artifacts, the new SMILES are featurized, and predictions
come back in original units with any log transform inverted. The SMILES are
whatever you type in the prompt (e.g. *"predict for c1ccc(/N=N/c2ccccc2)cc1
and CC(=O)Oc1ccccc1C(=O)O"*); there is no agent tool that reads a CSV of
SMILES for batch inference. For batch prediction from a CSV, use the
deterministic reload CLI below (`predict --csv`).

## Reloading saved models (deterministic, no Ollama)

`src/chemlagent/reload.py` lets you reuse a run's products straight from
`recent_work/<run_id>/` without going through the agent. It dispatches on the
manifest's `model_type` (random_forest / lightgbm / svr / mlp / chemprop) and
applies the manifest's log-transform inversion, so predictions are in original
units.

```bash
# list every saved run with its metrics
uv run python -m chemlagent.reload list

# predict for SMILES given on the command line
uv run python -m chemlagent.reload predict azo "c1ccc(/N=N/c2ccccc2)cc1"

# batch: read SMILES from a CSV, write predictions to a CSV
uv run python -m chemlagent.reload predict azo \
    --csv 621-azo.csv --smiles-col SMILES --out preds.csv
```

Paths resolve from `recent_work/<run_id>/` by the known filenames for each
model type, so it works whether or not the manifest's path fields were
rewritten at quit.

## Project layout

```
src/chemlagent/
  agent.py              Ollama dispatch loop + CLI
  tools.py              the 12 Ollama-callable tool functions (3 discovery + 9 pipeline)
  data.py               CSV prep (local + ChEMBL), column/log autodetect
  fingerprints.py       scikit-fingerprints wrappers + scaffold split
  models.py             RF / LightGBM / SVR + evaluate
  pytorch_mlp.py        wide-and-deep PyTorch MLP
  chemprop_model.py     Chemprop MPNN data + model classes
  descriptor_cleaning.py  impute / aggressive feature cleaners
  products.py            publish products to recent_work/ + list them on quit
  reload.py              deterministic model reload + inference from recent_work/
tests/test_smoke.py     end-to-end MVP round-trip on synthetic data
runs/<run_id>/          working run artifacts (gitignored, regenerable)
recent_work/<run_id>/   curated products copied out as stages complete (gitignored)
```

## Tests

```bash
uv run pytest
```

`test_smoke.py` exercises `prepare → featurize → train (RF) → evaluate` on
synthetic data, so the tools are verifiable without hitting Ollama.

## Notes

- **torch before lightgbm import order.** `tools.py` imports `torch` (guarded)
  before `chemlagent.fingerprints`/`chemlagent.models`. On Apple Silicon both
  torch and lightgbm bundle OpenMP; if lightgbm's loads first, torch ops
  segfault. Don't reorder those imports.
- No grid search is exposed (per scope); pass explicit hyperparameters or rely
  on the tuned defaults.
- `CLAUDE.md` holds the original project brief / agent instructions.

## Acknowledgments

CheMLAgent is a thin agentic layer on top of several excellent open-source
projects; the real chemistry and ML heavy lifting is theirs:

| package                                              | used for                                  |
|------------------------------------------------------|-------------------------------------------|
| [RDKit](https://www.rdkit.org/)                      | SMILES parsing, descriptors, Murcko scaffolds, conformer generation |
| [scikit-fingerprints](https://github.com/dfpl/scikit-fingerprints) | the fingerprint/descriptor estimators behind `featurize_fingerprints` |
| [scikit-learn](https://scikit-learn.org/)            | Random Forest, SVR, PCA, StandardScaler, metrics |
| [LightGBM](https://lightgbm.readthedocs.io/)         | gradient-boosted trees model               |
| [PyTorch](https://pytorch.org/)                      | the wide-and-deep MLP                     |
| [Chemprop](https://chemprop.org/)                   | message-passing neural network for molecular graphs |
| [Lightning](https://lightning.ai/)                  | Chemprop's training loop                   |
| [CheMeleon](https://arxiv.org/abs/2506.15792)        | pretrained MPNN foundation weights (wired in `chemprop_model` but currently inactive in `train_chemprop` due to memory — see above) |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) + `chembl_webresource_client` | bioactivity data and the `chembl_id` fetch mode |
| [Open Targets](https://www.opentargets.org/)        | disease → target associations (`search_targets`) |
| [UniProt](https://www.uniprot.org/)                 | protein/gene → accession resolution (`search_uniprot`) |
| [Ollama](https://ollama.com/)                       | chat model + tool-call dispatch           |
| [Rich](https://rich.readthedocs.io/)                | terminal UI (banners, tables, Markdown rendering) |
| [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) | tabular I/O and array math       |
| [uv](https://docs.astral.sh/uv/) / [pytest](https://docs.pytest.org/) | environment management / tests     |

Data fetched via ChEMBL, Open Targets, and UniProt is governed by their
respective licenses and terms of use.