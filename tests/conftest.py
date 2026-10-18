"""Shared test fixtures: the whole test suite runs on REAL data.

Every test module that needs a run featurizes a small slice of the repo's
real ChEMBL bioactivity CSV (CHEMBL220_bioactives.csv) through the actual
pipeline (prepare_chembl_csv -> featurize_fingerprints, ECFP, Murcko scaffold
split), once per module, from a per-module tmp CWD so runs/recent_work stay
isolated and are cleaned up with the tmp dir. The ``seeded_run`` fixture name
is what the tool test modules request.
"""

import os

import pytest

from chemlagent import tools as T

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))
_REAL_CSV = os.path.join(_REPO_ROOT, "CHEMBL220_bioactives.csv")


@pytest.fixture(scope="session")
def real_csv():
    """Absolute path to the repo's real ChEMBL bioactivity CSV."""
    if not os.path.exists(_REAL_CSV):
        pytest.skip("CHEMBL220_bioactives.csv not present in the repo")
    return _REAL_CSV


@pytest.fixture(scope="module")
def seeded_run(tmp_path_factory, real_csv):
    """A run over real data, featurized ONCE per test module.

    CWD is switched to a per-module tmp dir for the fixture's whole lifetime
    (restored on teardown), so runs/<run_id> and recent_work land there and
    die with the tmp dir. Yields the run_id ('real').
    """
    d = tmp_path_factory.mktemp("real_run")
    cwd = os.getcwd()
    os.chdir(d)
    run_id = "real"
    try:
        prep = T.prepare_chembl_csv(input_csv=real_csv, run_id=run_id,
                                    limit=200)
        assert prep["n_rows"] > 0
        T.featurize_fingerprints(run_id=run_id, fp_type="ECFP", test_size=0.2)
        yield run_id
    finally:
        os.chdir(cwd)