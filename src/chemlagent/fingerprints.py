from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
from skfp.preprocessing import MolFromSmilesTransformer
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from skfp.fingerprints import RDKitFingerprint 
from skfp.fingerprints import AtomPairFingerprint
from skfp.fingerprints import MordredFingerprint #, RDKit2DDescriptorsFingerprint
from skfp.fingerprints import MACCSFingerprint, PubChemFingerprint
from skfp.fingerprints import EStateFingerprint
from skfp.fingerprints import FunctionalGroupsFingerprint
from skfp.preprocessing import ConformerGenerator
from skfp.fingerprints import (
    AutocorrFingerprint,
    E3FPFingerprint,
    MORSEFingerprint,
    RDFFingerprint,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from chemlagent.spectral_fp import SpectralFingerprint

# Broadcast print flag: the agent sets chemlagent.fingerprints.print_flag =
# <args.print> at startup (see agent.py, mirroring src/agent_template.py). The
# featurizer has no banner; these feature-size / CV-result prints are debug
# output and are gated here. Default False = debug-silent unless --print.
print_flag = False

# RDKit embedding time limit (s per embedding call) for every ConformerGenerator
# here. Without it a molecule RDKit cannot embed burns its full attempt budget
# before failing: up to 22 min each on CHEMBL220 (4 such molecules), which was
# ~all of the 3D featurization time. Measured: such a molecule now fails in
# ~30 s while the largest molecules that do embed (58-77 heavy atoms) still
# succeed in 1-7 s.
EMBED_TIMEOUT = 10


class SmilesFeaturizer:
  """A picklable SMILES -> feature-matrix adapter around a fitted skfp
  fingerprint estimator.

  skfp fingerprint estimators' ``.transform`` consumes RDKit Mol objects (the
  output of ``MolFromSmilesTransformer``), and 3D fingerprint types additionally
  need conformer generation. This wrapper mirrors ``get_fingerprints.transform``
  so that anything holding a saved featurizer can featurize new SMILES with a
  plain ``.transform(smiles_list)`` call -- used by the inference tools
  (sklearn models and the PyTorch MLP, whose ``predict_single_value`` expects
  this interface). It is what ``featurize_fingerprints`` pickles.
  """

  def __init__(self, fp, is_3d: bool, many_conf: bool = True,
               num_confs: int = 5, n_jobs: int = -1):
    self.fp = fp
    self.is_3d = is_3d
    self.many_conf = many_conf
    self.num_confs = num_confs
    self.n_jobs = n_jobs

  def transform(self, smiles_list):
    mols = MolFromSmilesTransformer().transform(clean_smiles(smiles_list))
    if self.is_3d:
      if self.many_conf:
        conf_gen = ConformerGenerator(timeout=EMBED_TIMEOUT, num_conformers=self.num_confs,
                                      optimize_force_field="UFF",
                                      n_jobs=self.n_jobs)
      else:
        conf_gen = ConformerGenerator(timeout=EMBED_TIMEOUT, n_jobs=self.n_jobs)
      mols = conf_gen.transform(mols)
    return self.fp.transform(mols)


class RDKit2DDescriptors:
  """Compute RDKit's native full 2D descriptor set (`Descriptors._descList`,
  ~217 continuous physicochemical/topological/electronic descriptors) -- the
  RDKit-builtin continuous counterpart to Mordred (which wraps the separate
  Mordred library). Mimics the skfp `.transform(mols) -> np.ndarray` interface so
  it plugs into `get_fingerprints` unchanged. Per-descriptor/per-molecule errors
  are emitted as NaN and handled downstream by `descriptor_cleaning` imputation
  (the same way Mordred's NaNs are handled). `fit` is a no-op -- the descriptor
  set is fixed. (skfp's own `RDKit2DDescriptorsFingerprint` is not available in
  this version, so this thin wrapper fills that slot.)
  """
  def __init__(self, n_jobs=-1):
    self.n_jobs = n_jobs

  def fit(self, X, y=None):
    return self

  def transform(self, mols):
    desc = list(Descriptors._descList)
    out = np.empty((len(mols), len(desc)), dtype=np.float64)
    for i, mol in enumerate(mols):
      for j, (_, fn) in enumerate(desc):
        try:
          out[i, j] = float(fn(mol))
        except Exception:
          out[i, j] = np.nan
    return out


def scaffold_train_test_split(molecules, targets, test_size=0.2, random_state=None):
  """
  Split molecules into train and test sets based on Murcko scaffold.
  Ensures that molecules with the same scaffold go to the same set.
  
  Parameters:
  -----------
  molecules : list of RDKit mol objects or SMILES strings
      Input molecules to split
  targets : array-like
      Target values corresponding to molecules
  test_size : float, default=0.2
      Proportion of data to include in test set
  random_state : int, optional
      Random seed for reproducibility
  
  Returns:
  --------
  train_mols, test_mols, train_targets, test_targets
  """
  if random_state is not None:
    np.random.seed(random_state)
  
  targets = np.asarray(targets)
  
  # Convert SMILES to molecules if needed
  if isinstance(molecules[0], str):
    mols = [Chem.MolFromSmiles(smi) for smi in molecules]
  else:
    mols = molecules
  
  # Get scaffolds for each molecule
  scaffolds = {}
  for idx, mol in enumerate(mols):
    if mol is not None:
      scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(mol))
      if scaffold not in scaffolds:
        scaffolds[scaffold] = []
      scaffolds[scaffold].append(idx)
  
  # Randomly assign scaffolds to train/test
  scaffold_list = list(scaffolds.keys())
  np.random.shuffle(scaffold_list)
  
  # Calculate split point
  n_test_scaffolds = max(1, int(len(scaffold_list) * test_size))
  test_scaffolds = set(scaffold_list[:n_test_scaffolds])
  
  # Collect indices for train and test
  train_indices = []
  test_indices = []
  
  for idx, mol in enumerate(mols):
    if mol is not None:
      scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(mol))
      if scaffold in test_scaffolds:
        test_indices.append(idx)
      else:
        train_indices.append(idx)
  
  # Split the data
  train_mols = [molecules[i] for i in train_indices]
  test_mols = [molecules[i] for i in test_indices]
  train_targets = targets[train_indices]
  test_targets = targets[test_indices]
  
  return train_mols, test_mols, train_targets, test_targets


def clean_smiles(smiles_list: list[str]):
  '''
  Strip counter-ions / solvent from salt SMILES by keeping the largest
  fragment (RDKit LargestFragmentChooser). Catches every salt form, including
  the neutral HCl/HBr written as '.Cl' / '.Br', BF4-, mesylate, TFA and water
  that a fixed ion list misses. Single-fragment SMILES are returned unchanged;
  unparseable SMILES are passed through for the featurizer to handle.
  '''
  chooser = rdMolStandardize.LargestFragmentChooser()
  cleaned = []
  # BlockLogs: the chooser logs several INFO lines per salt molecule.
  with rdBase.BlockLogs():
    for smiles in smiles_list:
      if '.' in smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
          smiles = Chem.MolToSmiles(chooser.choose(mol))
      cleaned.append(smiles)
  return cleaned

def _embed_or_none(gen, mol):
  '''Conformer-embed one molecule; None if RDKit cannot embed it.'''
  try:
    return gen.transform([mol])[0]
  except ValueError as exc:
    if "Could not generate conformer" not in str(exc):
      raise
    return None


def _embed_chunk(gen, mols):
  '''Batch-embed a chunk. skfp aborts the batch at the first molecule it
  cannot embed and names it in the error; that molecule becomes None (it is
  not retried -- failing is the expensive part) and the rest of the chunk is
  embedded again.'''
  try:
    return list(gen.transform(mols))
  except ValueError as exc:
    msg = str(exc)
    if "Could not generate conformer for " not in msg:
      raise
    bad = msg.split("Could not generate conformer for ", 1)[1].split(":", 1)[0].strip()
    hit = [i for i, m in enumerate(mols) if Chem.MolToSmiles(m) == bad]
    if not hit:  # SMILES did not round-trip: fall back to one at a time
      return [_embed_or_none(gen, m) for m in mols]
    i = hit[0]
    rest = _embed_chunk(gen, mols[:i] + mols[i + 1:])
    return rest[:i] + [None] + rest[i:]


class get_fingerprints():
  '''
  Finerprints available include: ECFP, Atom_Pair, Mordred, RDKit_2D, MACCS, PubChem, 
  Functional_Groups, RDKitFingerprint, Spectral_2D, E3FP, Autocorr, MORSE, RDF, Spectral.
  E3FP, Autocorr, MORSE, RDF and Spectral are 3D fingerprints and require conformer
  generation. The rest are 2D fingerprints and do not require conformer generation.
  '''
  def __init__(self, smiles_list: list[str], target_list: list[float], transform_flag: bool, n_jobs=-1):
    '''

    '''
    self.smiles_list = clean_smiles(smiles_list)
    self.target_list = target_list
    self.transform_flag = transform_flag
    self.n_jobs = n_jobs

    if transform_flag:
      self.y = np.log10(target_list)
    else:
      self.y = target_list
    
    mols_from_smiles = MolFromSmilesTransformer()
    self.mols_raw = mols_from_smiles.transform(self.smiles_list)

    self.types_3d = ['E3FP', 'Autocorr', 'MORSE', 'RDF', 'Spectral']
    self.types_2d = ['ECFP', 'Atom_Pair', 'Mordred', 'RDKit_2D', 'MACCS', 'PubChem', 'EState',
                     'Spectral_2D']
    self.fp_hash = {
        'ECFP': (ECFPFingerprint(n_jobs=self.n_jobs),{'fp_size': [1024, 2048], 'radius': [2, 4, 6]}),
        'Atom_Pair': (AtomPairFingerprint(n_jobs=self.n_jobs),{'fp_size': [1024, 2048], 'min_distance': [1, 2, 3], 'max_distance': [20,30,40]}),
        'Mordred': (MordredFingerprint(n_jobs=self.n_jobs),{'use_3D': [True, False]}),
        # RDKit's native ~217 continuous 2D descriptors (Descriptors._descList),
        # via the thin RDKit2DDescriptors wrapper above -- the RDKit-builtin
        # continuous counterpart to Mordred. NaNs imputed downstream.
        'RDKit_2D': (RDKit2DDescriptors(n_jobs=self.n_jobs),{}),
        'MACCS': (MACCSFingerprint(n_jobs=self.n_jobs),{}),
        'PubChem': (PubChemFingerprint(n_jobs=self.n_jobs),{}),
        # EState: 79 continuous Kier-Hall electrotopological-state descriptors
        # (atom electronic environment) -- a 2D continuous-alternative to
        # Mordred, directly relevant to push-pull chromophore lambda_max.
        'EState': (EStateFingerprint(n_jobs=self.n_jobs),{}),
        'Functional_Groups': (FunctionalGroupsFingerprint(n_jobs=self.n_jobs),{}),
        'RDKitFingerprint': (RDKitFingerprint(n_jobs=self.n_jobs),{'fp_size': [1024, 2048], 'max_path': [5, 7, 9]}),
        'E3FP': (E3FPFingerprint(n_jobs=self.n_jobs),{'fp_size': [1024, 2048], 'radius_multiplier': [1.1, 1.1718, 3.0]}),
        'Autocorr': (AutocorrFingerprint(n_jobs=self.n_jobs),{'use_3D': [True, False]}),
        'MORSE': (MORSEFingerprint(n_jobs=self.n_jobs),{}),
        'RDF': (RDFFingerprint(n_jobs=self.n_jobs),{}),
        # Spectral fingerprints (Kulik group, Chem. Sci. 2026; see spectral_fp.py):
        # 64 graph-Laplacian eigenvalues over 4 physics channels. 'Spectral' uses
        # conformer coordinates, 'Spectral_2D' bond-path distances (no conformers).
        'Spectral': (SpectralFingerprint(use_3D=True, n_jobs=self.n_jobs),{}),
        'Spectral_2D': (SpectralFingerprint(use_3D=False, n_jobs=self.n_jobs),{}),
    }

  def create(self, fp_type: str = 'Mordred', many_conf = True, num_confs = 5, test_size = 0.2,
             split_seed: int = 132):
    '''
    '''
    self.fp_type = fp_type
    self.many_conf = many_conf
    self.num_confs = num_confs

    y = np.asarray(self.y)
    if self.fp_type in self.types_3d:
      self.mols, y = self._embed_training_mols(y)
    else:
      self.mols = self.mols_raw
    
    self.fp = self.fp_hash[self.fp_type][0]

    self.mols_train, self.mols_test, self.y_train, self.y_test = scaffold_train_test_split(
        self.mols, y, test_size=test_size, random_state=split_seed)
  
    self.X_train = self.fp.transform(self.mols_train)
    self.X_test = self.fp.transform(self.mols_test)

    if print_flag:
      print(f'Fingerprints created. The feature array size per molecule is: {self.X_train.shape[1]}')

    return self.X_train, self.X_test, self.y_train, self.y_test

  def _embed_training_mols(self, y):
    '''Generate conformers for the training set. A molecule RDKit cannot embed
    (after skfp's ETKDG -> KDG -> plain-DG fallbacks; e.g. some strained cage
    compounds) is dropped together with its target, with a printed note,
    instead of aborting the whole featurization. Returns (mols, y).

    Molecules are embedded in parallel chunks with skfp's normal batch call;
    only a chunk that hits an unembeddable molecule is redone one molecule at
    a time (skfp's errors="ignore"/"filter" crash with num_conformers > 1).
    Conformers do not depend on the split or on which 3D fp_type is used, so
    they are cached on this object per (many_conf, num_confs): further
    create() calls (other split seeds, other 3D types) reuse them.'''
    key = (self.many_conf, self.num_confs)
    cache = self.__dict__.setdefault('_conformer_cache', {})
    if key not in cache:
      kw = dict(num_conformers=self.num_confs, optimize_force_field="UFF") if self.many_conf else {}
      gen = ConformerGenerator(timeout=EMBED_TIMEOUT, n_jobs=1, **kw)
      n_chunks = 4 * effective_n_jobs(self.n_jobs)
      chunks = [c for c in np.array_split(np.arange(len(self.mols_raw)), n_chunks) if len(c)]
      out = Parallel(n_jobs=self.n_jobs)(
          delayed(_embed_chunk)(gen, [self.mols_raw[i] for i in c]) for c in chunks)
      mols = [m for chunk in out for m in chunk]
      ok = np.array([m is not None for m in mols])
      self.dropped_no_conformer = [Chem.MolToSmiles(m0)
                                   for m0, k in zip(self.mols_raw, ok) if not k]
      if self.dropped_no_conformer:
        print(f"Dropped {len(self.dropped_no_conformer)} molecule(s) with no 3D "
              f"conformer: {self.dropped_no_conformer}")
      cache[key] = ([m for m, k in zip(mols, ok) if k], ok)
    mols, ok = cache[key]
    return mols, y[ok]

  def transform(self, smiles_list: list[str]):
    '''
    '''
    self.new_smiles_list = clean_smiles(smiles_list)

    mols_from_smiles = MolFromSmilesTransformer()
    self.new_mols_raw = mols_from_smiles.transform(self.new_smiles_list)

    if self.fp_type in self.types_3d:
      if self.many_conf:
        conformer_gen = ConformerGenerator(timeout=EMBED_TIMEOUT, num_conformers=self.num_confs, optimize_force_field="UFF", n_jobs=self.n_jobs)
      else:
        conformer_gen = ConformerGenerator(timeout=EMBED_TIMEOUT, n_jobs=self.n_jobs)
      
      self.new_mols = conformer_gen.transform(self.new_mols_raw)
    else:
      self.new_mols = self.new_mols_raw
    
    self.new_X = self.fp.transform(self.new_mols)

    if print_flag:
      print(f'Fingerprints created. The feature array size per molecule is: {self.new_X.shape[1]}')

    return self.new_X
  
  def fp_gridsearch(self, fp_type: str, model, model_grid, many_conf = True, num_confs = 5):
    '''
    '''
    self.fp_type = fp_type
    self.model = model
    self.model_grid = model_grid
    self.fp_grid = self.fp_hash[self.fp_type][1]
    self.many_conf = many_conf
    self.num_confs = num_confs

    fp = self.fp_hash[self.fp_type][0]

    gs_cv = GridSearchCV(
        estimator=self.model,
        param_grid=self.model_grid,
    )

    fp_cv = FingerprintEstimatorGridSearch(fp, self.fp_grid, gs_cv)

    if self.fp_type in self.types_3d:
      self.mols, y = self._embed_training_mols(np.asarray(self.y))
      self.mols_train, self.mols_test, self.y_train, self.y_test = scaffold_train_test_split(
          self.mols, y, test_size=0.2)
      fp_cv.fit(self.mols_train, self.y_train)
      y_pred = fp_cv.predict(self.mols_test)
      r2 = r2_score(self.y_test, y_pred)
    else:
      self.smiles_train, self.smiles_test, self.y_train, self.y_test = scaffold_train_test_split(
          self.smiles_list, self.y, test_size=0.2)
      fp_cv.fit(self.smiles_train, self.y_train)
      y_pred = fp_cv.predict(self.smiles_test)
      r2 = r2_score(self.y_test, y_pred)

    if print_flag:
      print(f"R2 score for best estimator: {r2:.3f}")
      print(f"Best FP parameters: {fp_cv.best_fp_params_}")
      print(f"CV results: {fp_cv.cv_results_}")

    return r2
  
  