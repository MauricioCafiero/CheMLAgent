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
