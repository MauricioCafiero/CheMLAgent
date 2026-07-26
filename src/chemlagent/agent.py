"""Ollama-driven CLI for CheMLAgent.

A thin agentic loop modeled on ``src/agent_template.py``: it hands the
chemistry tools in :mod:`chemlagent.tools` to an Ollama chat model, dispatches
the model's tool calls, and feeds the JSON results back until the model has no
more calls to make. Ollama infers each tool's JSON schema from the function
signature + docstring, so the tools are passed as raw function objects.

Run::

    uv run python -m chemlagent.agent --print --model glm-5.2

The Ollama host and API key are read from OLLAMA_HOST (default
https://ollama.com) and OLLAMA_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ollama import Client as ollama_client

from chemlagent import tools as T
from chemlagent.markdown_render import to_unicode_markdown

console = Console(width=80)

# CheMLAgent figlet banner (slant font), rendered at startup. Hardcoded so the
# CLI needs no figlet dependency. Raw string so the backslashes in the art are
# literal (no escape processing); the leading newline is stripped at render.
_BANNER_LOGO = r"""
   ________         __  _____    ___                    __
  / ____/ /_  ___  /  |/  / /   /   | ____ ____  ____  / /_
 / /   / __ \/ _ \/ /|_/ / /   / /| |/ __ `/ _ \/ __ \/ __/
/ /___/ / / /  __/ /  / / /___/ ___ / /_/ /  __/ / / / /_
\____/_/ /_/\___/_/  /_/_____/_/  |_\__, /\___/_/ /_/\__/
                                   /____/
"""

# Tools exposed to the model. Ollama builds the tool schemas from these
# functions' signatures + docstrings, so keep those informative (see tools.py).
TOOLS = [
    T.search_uniprot,
    T.list_bioactives,
    T.search_targets,
    T.prepare_chembl_csv,
    T.featurize_fingerprints,
    T.train_model,
    T.grid_search,
    T.evaluate_model,
    T.run_inference,
    T.train_mlp,
    T.run_inference_mlp,
    T.train_chemprop,
    T.run_inference_chemprop,
]

AVAILABLE_FUNCTIONS = {fn.__name__: fn for fn in TOOLS}

DEFAULT_MODELS = ["gemma4:31b", "glm-5.2", "qwen3.5:397b"]

SYS_MESSAGE = f"""\
You are CheMLAgent, an assistant for building and training machine-learning \
models for chemistry (QSAR / bioactivity prediction). You have access to the \
following tools: {', '.join(fn.__name__ for fn in TOOLS)}.

To answer a modeling request, drive this pipeline and thread `run_id` between \
calls so artifacts stay linked:

0. DISCOVERY (only if the user gives a protein name or disease instead of a \
ChEMBL ID — if they already give a CHEMBLxxx ID or a CSV, skip to step 1):
   - search_targets — disease name -> ranked gene symbols (approved target \
names). Use when the user names a disease (e.g. "breast cancer").
   - search_uniprot — protein name / gene symbol -> UniProt accession IDs \
(set human_only=True for human targets).
   - list_bioactives — UniProt IDs -> ChEMBL target IDs + their activity \
counts. Pick the ChEMBL ID with the largest count, then pass it to step 1.
   Typical chain: search_targets -> search_uniprot -> list_bioactives -> \
prepare_chembl_csv(chembl_id=...).
1. prepare_chembl_csv  — get a normalized (SMILES, target) dataset. Use \
input_csv=<path> for a user-supplied CSV, or chembl_id=<ChEMBL target ID> to \
fetch from ChEMBL (the ID may come from step 0). Choose a run_id (e.g. the \
target name or a short label). SMILES/target columns and whether to \
log-transform the target are autodetected.
2. featurize_fingerprints — featurize the dataset with a fingerprint type \
(e.g. ECFP, Mordred, PubChem, MACCS) and scaffold-split it. Required before \
train_model / train_mlp. Use the SAME run_id.
3. Train ONE model on that run (same run_id):
   - train_model — random_forest, lightgbm, or svr (on the fingerprints).
   - train_mlp — PyTorch MLP (on the fingerprints).
   - train_chemprop — Chemprop MPNN on molecular graphs (reads data.csv \
directly; trains a small from-scratch MPNN). No \
featurize_fingerprints call needed for chemprop.
   Report the returned r2 / mae.
   - grid_search — optional: 5-fold CV grid search over a SMALL grid for ONE of \
random_forest / lightgbm / svr (caps: 3 values per hyperparameter, 6 combos \
total), then saves the best fit like train_model. Tunable params: \
random_forest {{n_estimators, max_depth, min_samples_leaf, max_features}}, \
lightgbm {{n_estimators, num_leaves, learning_rate, min_child_samples}}, \
svr {{C, gamma, epsilon}}. Use it when the user asks to tune or optimize, or \
when a plain train_model r2 is poor; otherwise train_model is enough.
4. evaluate_model — evaluate the saved fingerprint model on the held-out test \
split (not used for chemprop, which reports test metrics at train time).
5. run_inference / run_inference_mlp / run_inference_chemprop — predict the \
target for new SMILES using the trained model. Use the SAME run_id. \
Predictions come back in original units (e.g. IC50 nM or nm) with any log \
transform inverted automatically.

Always reuse the same run_id across the stages of one task. Report metrics \
(R^2, MAE) clearly to the user, and explain in plain terms what was built. \
If a tool returns an error, read the message and adjust (e.g. pick a different \
fp_type or model_type). Do not invent tool names or arguments.
"""


def _make_client() -> ollama_client:
    host = os.getenv("OLLAMA_HOST", "https://ollama.com")
    headers = {}
    key = os.getenv("OLLAMA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return ollama_client(host=host, headers=headers)


def chat_turn(prompt: str, model: str, messages: list,
              client: ollama_client, print_flag: bool,
              session_runs: set[str] | None = None) -> str:
    """Run one user turn to completion, executing any tool calls, and return
    the model's final text content. ``session_runs`` collects every ``run_id``
    the model passes to a tool this session so the on-quit report can list only
    what was produced this session, not leftover runs from previous sessions."""
    messages.append({"role": "user", "content": prompt})

    while True:
        response = client.chat(
            model=model, messages=messages, tools=TOOLS, think=True,
        )
        messages.append(response.message)
        if print_flag:
            console.print("--------------------------------------------------------")
            console.print(f"[dim]Thinking:[/dim] {response.message.thinking}")
            console.print("--------------------------------------------------------")
            console.print(f"[dim]Content:[/dim] {response.message.content}")

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                name = tc.function.name
                args = tc.function.arguments
                if isinstance(args, dict) and session_runs is not None:
                    rid = args.get("run_id")
                    if rid:
                        session_runs.add(str(rid))
                if print_flag:
                    console.print(f"[cyan]Calling {name}[/cyan] with {args}")
                fn = AVAILABLE_FUNCTIONS.get(name)
                if fn is None:
                    result = f"Error: unknown tool {name!r}."
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:  # surface errors to the model
                        result = f"Error in {name}: {exc!r}"
                if print_flag:
                    console.print(f"[green]Result:[/green] {result}")
                messages.append({"role": "tool", "tool_name": name,
                                 "content": str(result)})
        else:
            break

    return messages[-1]["content"] or ""


def _print_recent_work(session_runs: set[str] | None = None) -> None:
    """On quit: move each run's products into recent_work/, clean up the
    intermediates left in runs/, then remind the user where this session's
    products are. Only run_ids used during this session are reported, so old
    runs already in recent_work/ from previous sessions are not listed."""
    from chemlagent.products import (
        RECENT_WORK_DIR, list_recent_work, relocate_all_runs)

    relocate_all_runs()
    files = list_recent_work()
    if not files:
        return
    # Keep only files whose run_id was produced this session. rel path under
    # recent_work/ is <run_id>/<file>, so the first component is the run_id.
    base = os.path.abspath(RECENT_WORK_DIR)
    wanted = session_runs or set()
    shown = []
    for path in files:
        rel = os.path.relpath(path, base)
        rid = rel.split(os.sep)[0]
        if rid in wanted:
            shown.append(path)
    if not shown:
        return
    console.print(f"\n[bold green]Your work is saved in[/bold green] "
                  f"[cyan]{base}/[/cyan]")
    for path in sorted(shown):
        console.print(f"  • {path}")


def _help_keywords() -> None:
    """Print the deterministic REPL keywords (the ones that skip the model)."""
    console.print(Panel(
        "[bold]Deterministic keywords[/bold] [dim](skip the chat model):[/dim]\n\n"
        "  [cyan]/models[/cyan]   "
        "[dim]list saved runs, then pick one to load for inference[/dim]\n"
        "  [cyan]/predict[/cyan]  "
        "[dim]run inference with the loaded model:[/dim]\n"
        "      [dim]<SMILES ...>                  inline SMILES[/dim]\n"
        "      [dim]--csv <file> [--smiles-col C] [--out out.csv]   "
        "batch from a CSV[/dim]\n"
        "      [dim]<run_id> <SMILES ...>         load that run, then predict[/dim]\n"
        "  [cyan]/help[/cyan]    [dim]show these keywords[/dim]\n"
        "  [cyan]quit[/cyan]/[cyan]exit[/cyan]  "
        "[dim]exit; moves products to recent_work/ and prints their paths[/dim]\n\n"
        "[dim]A <run_id> is the label a run was prepared/trained under -- the "
        "folder under recent_work/. /models lists the runs you have and loads "
        "the one you pick; it stays loaded so later /predict calls reuse it "
        "(the prompt shows [run_id] when one is active).[/dim]\n\n"
        "[dim]Anything else is sent to the model as your request.[/dim]",
        box=box.ROUNDED, border_style="cyan", padding=(1, 2), expand=False))


def _is_run_id(token: str) -> bool:
    """True if ``token`` is an existing saved run folder under recent_work/."""
    from chemlagent.products import recent_work_path
    return os.path.isdir(os.path.join(recent_work_path(), token))


def _print_models_table(models: list[dict]) -> None:
    """Print the saved-runs table with a 1-based row index for selection."""
    table = Table(title="Saved models in recent_work/", show_lines=False)
    for col in ("#", "run_id", "model_type", "fp_type", "is_foundation",
                "r2", "mae", "n_train", "n_test", "log_transformed"):
        table.add_column(col)
    for i, m in enumerate(models, 1):
        table.add_row(
            str(i), str(m["run_id"]), str(m["model_type"]),
            str(m.get("fp_type") or "-"),
            str(m.get("is_foundation") if m.get("is_foundation") is not None
                else "-"),
            f"{m['r2']:.3f}" if m.get("r2") is not None else "-",
            f"{m['mae']:.2f}" if m.get("mae") is not None else "-",
            str(m.get("n_train") or "-"), str(m.get("n_test") or "-"),
            str(m.get("log_transformed")))
    console.print(table)


def _do_models(state: dict) -> None:
    """List saved runs, then let the user load one into ``state['loaded']``."""
    from chemlagent.reload import list_available_models, LoadedModel

    models = list_available_models()
    if not models:
        console.print("[dim]No saved models in recent_work/. "
                      "Train a model first, then quit to publish it.[/dim]")
        return
    _print_models_table(models)
    try:
        choice = console.input(
            "\n[bold cyan]Load which? (number or run_id, Enter to skip) > [/bold cyan]")
    except (EOFError, KeyboardInterrupt):
        console.print()
        return
    choice = choice.strip()
    if not choice:
        return
    run_id = None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(models):
            run_id = models[idx - 1]["run_id"]
    elif _is_run_id(choice):
        run_id = choice
    if run_id is None:
        console.print(f"[red]No match for[/red] {choice!r}. "
                      "Use a row number or a run_id from the list.")
        return
    try:
        lm = LoadedModel.load(run_id)
    except Exception as exc:
        console.print(f"[red]load error:[/red] {exc!r}")
        return
    state["loaded"] = lm
    state["loaded_id"] = run_id
    console.print(f"[green]Loaded[/green] [cyan]{run_id}[/cyan] "
                  f"({lm.model_type}). Now use [cyan]/predict[/cyan] "
                  f"with SMILES or [cyan]--csv[/cyan].")


def _parse_predict_args(args: list[str]):
    """Split /predict args into (csv, smiles_col, out, positional_smiles)."""
    csv = out = None
    smiles_col = "SMILES"
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--csv" and i + 1 < len(args):
            csv = args[i + 1]; i += 2
        elif a == "--smiles-col" and i + 1 < len(args):
            smiles_col = args[i + 1]; i += 2
        elif a == "--out" and i + 1 < len(args):
            out = args[i + 1]; i += 2
        else:
            positional.append(a); i += 1
    return csv, smiles_col, out, positional


def _do_predict(state: dict, args: list[str]) -> None:
    """Run inference with the loaded model (optionally loading a run_id first)."""
    from chemlagent.reload import LoadedModel

    # Optional leading run_id: load it (and keep it loaded), then predict.
    if args and _is_run_id(args[0]):
        run_id = args[0]
        try:
            lm = LoadedModel.load(run_id)
        except Exception as exc:
            console.print(f"[red]load error:[/red] {exc!r}")
            return
        state["loaded"] = lm
        state["loaded_id"] = run_id
        console.print(f"[green]Loaded[/green] [cyan]{run_id}[/cyan] ({lm.model_type}).")
        args = args[1:]
    else:
        lm = state.get("loaded")
        if lm is None:
            console.print("[red]No model loaded.[/red] Load one with "
                          "[cyan]/models[/cyan], or use "
                          "[cyan]/predict <run_id> ...[/cyan].")
            return

    csv, smiles_col, out, positional = _parse_predict_args(args)
    smiles = list(positional)
    if csv:
        try:
            df = pd.read_csv(csv)
        except Exception as exc:
            console.print(f"[red]csv error:[/red] {exc!r}")
            return
        col = (smiles_col if smiles_col in df.columns else next(
            (c for c in df.columns if "smiles" in c.lower()), smiles_col))
        smiles = df[col].astype(str).tolist()
    if not smiles:
        console.print("[red]No SMILES.[/red] Give SMILES inline or "
                      "[cyan]--csv <file>[/cyan].")
        return
    try:
        preds = lm.predict(smiles)
    except Exception as exc:
        console.print(f"[red]predict error:[/red] {exc!r}")
        return
    out_df = pd.DataFrame({"SMILES": smiles, "prediction": preds})
    if out:
        out_df.to_csv(out, index=False)
        console.print(f"[green]wrote[/green] {len(out_df)} predictions "
                      f"to [cyan]{out}[/cyan]")
    else:
        console.print(out_df.to_string(index=False))


def _handle_keyword(prompt: str, state: dict) -> bool:
    """Run a deterministic REPL keyword (no Ollama call).

    Returns True if ``prompt`` was recognized and handled (skip the chat model),
    False to fall through to the model. ``state`` holds the currently loaded
    model (``state['loaded']`` / ``state['loaded_id']``) so /predict can reuse
    it across calls. The inference keywords are slash-prefixed so a natural-
    language request starting with "predict" is not hijacked; quit/exit are
    handled by the caller.
    """
    text = prompt.strip()
    if not text:
        return False
    cmd = text.split()[0].lower()

    if cmd in ("/help", "help", "?"):
        _help_keywords()
        return True
    if cmd in ("/models", "/reload", "/list"):
        _do_models(state)
        return True
    if cmd == "/predict":
        _do_predict(state, text.split()[1:])
        return True
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CheMLAgent — an Ollama-driven chemistry ML assistant.")
    parser.add_argument("--print", action="store_true",
                        help="verbose mode: print thinking, tool calls, results")
    parser.add_argument("--model", default=None,
                        help="Ollama chat model to use; defaults to the first "
                             "entry in DEFAULT_MODELS")
    args = parser.parse_args(argv)

    # Broadcast --print to every tool module, mirroring src/agent_template.py
    # (`modrag_protein_functions.print_flag = print_flag`). Each module gates its
    # debug/progress prints on this flag; tool banners always print. Default
    # False = tools are silent unless --print is given.
    from chemlagent import data, pytorch_mlp, models, fingerprints, chemprop_model
    for mod in (T, data, pytorch_mlp, models, fingerprints, chemprop_model):
        mod.print_flag = args.print

    model = args.model or DEFAULT_MODELS[0]
    client = _make_client()
    messages = [{"role": "system", "content": SYS_MESSAGE}]

    console.print(Panel(
        f"[bold cyan]{_BANNER_LOGO.lstrip(chr(10))}[/bold cyan]\n\n"
        "[bold magenta]An Ollama-driven chemistry ML assistant.[/bold magenta]\n"
        "[dim]Prepare a ChEMBL/local CSV · featurize SMILES · train a model · "
        "evaluate · run inference.[/dim]\n\n"
        "[bold]Keywords:[/bold] [cyan]/models[/cyan] [dim]list + load a run[/dim] · "
        "[cyan]/predict[/cyan] [dim]infer with loaded model (SMILES or --csv)[/dim] · "
        "[cyan]/help[/cyan] · [cyan]quit[/cyan] [dim]exit[/dim]",
        box=box.ROUNDED, border_style="cyan", padding=(1, 2), expand=False))
    console.print()

    state: dict = {"loaded": None, "loaded_id": None}
    session_runs: set[str] = set()
    while True:
        loaded_id = state.get("loaded_id")
        tag = f" [{loaded_id}]" if loaded_id else ""
        try:
            prompt = console.input(
                f"[bold cyan]What can I help with?{tag} > [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            _print_recent_work(session_runs)
            break
        if prompt.strip().lower() in ("quit", "exit"):
            console.print("Goodbye!")
            _print_recent_work(session_runs)
            break
        if not prompt.strip():
            continue
        if _handle_keyword(prompt, state):
            continue

        start = time.time()
        try:
            content = chat_turn(prompt, model, messages, client, args.print,
                                session_runs)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc!r}")
            continue
        elapsed = (time.time() - start) / 60
        console.print(f"\n[bold magenta]Response ({elapsed:.2f}m) >[/bold magenta]")
        console.print(Markdown(to_unicode_markdown(content)))
        console.print()


if __name__ == "__main__":
    sys.exit(main())