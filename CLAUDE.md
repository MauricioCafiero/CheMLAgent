#  CheMLAgent: an agentic system for building and training ML models for chemistry

This agent will:
- parse a csv file (prepare a ChEMBL CSV and apply log to a target if needed)
- featurize SMILES strings using either:
 * RDKit or Modred descriptors
 * fingerprints (Pubchem, Morgan or others)
 * molecular graphs (via chemprop)
- apply PCA to descriptors or fingerprints or otherwise clean features
- split datasets according to scaffolds (Murko, etc)
- scale data if needed
- create a model with:
 * Random forest
 * Light GBM
 * SVR
 * MLP (pytorch)
 * message passing neural network (for molecular graphs -- via chemprop)
- train models and evaluate
- save models (.pt files, or perhaps pickle for sklearn models)
- run inference on a saved model

There are functions and classes for most of this already in src/

- there are functions in modrag_protein_functions.py that are helpers to find a chembl csv if needed
- the functionality to clean the ChEMBL csv files is in the function get_bioactives but should be copied to an independent function that focuses on csv prep.

There is a teamplate for the agentic part in scr/

## current priorities

- examine existing code
- set up an environment using python venv or uv (not conda)
- convert classes into functions that are callable from the ollama api.
- make dockstrings into informative docktrings that the ollama api can use
- wire the tools into the ollama code. 

## launching long-running background jobs (e.g. grid scans)

This machine auto-suspends after 1 hour idle, even on AC power (GNOME
`sleep-inactive-ac-timeout=3600`, type `suspend`) -- this killed a
background MLP grid scan mid-run more than once (see
`INVESTIGATION.md`). Before launching anything expected to run
unattended for more than ~1 hour, set a sleep/idle inhibitor first:

```bash
systemd-inhibit --what=sleep:idle --who="CheMLAgent" \
    --why="Running background ML grid scans" --mode=block sleep infinity &
disown
```

Verify it's active with `systemd-inhibit --list` (look for a `CheMLAgent`
row). Release it once the job finishes (or the user says they're stepping
away) by killing that `sleep infinity` process -- don't leave it running
indefinitely, since it blocks suspend for the whole session, not just the
job.

The grid scan script (`big_datasets/scripts/mlp_grid_scan.py`) writes
results incrementally and resumes from the last completed row if a run
gets interrupted regardless, so a missed inhibitor isn't catastrophic --
just a lost few hours' progress and a manual resume.
