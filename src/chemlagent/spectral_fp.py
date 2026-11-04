"""Spectral fingerprints from graph-Laplacian eigenvalues (Toney, Farnood,
Darouich & Kulik, Chem. Sci. 2026, DOI 10.1039/d6sc06309h).

Ported from the MIT-licensed reference implementation
https://github.com/hjkgrp/SpectralScore (spectralscore/spectrum.py,
get_spectrum + get_eigenvalues, default normalize=True path; element table from
spectralscore/globalvars.py). Only the fingerprint math is taken -- the repo's
file readers (which import molSimplify), clustering and similarity code are
not needed, since chemlagent already has RDKit Mols (with conformers for 3D).

Each molecule is a complete graph over all atoms (explicit H included). Four
physics-inspired edge-weight channels -- electrostatics, dispersion, covalent
bonding, sterics -- each give an adjacency matrix A (Frobenius-normalised);
from each Laplacian L = D - A we keep the Fiedler eigenvalue plus the
n_eigs-1 largest, zero-padded for small molecules. Result: 4 * n_eigs
continuous features (64 at the default n_eigs=16), ordered [elec, disp, cov,
steric] as in the reference code.

Distances: 3D uses the conformer coordinates (the lowest-energy conformer
skfp's ConformerGenerator marks with the 'conf_id' prop). 2D (use_3D=False)
uses the reference "2D scope": along the shortest bond path (up to 5 bonds,
ties broken by smaller radius sum) the distance is the sum of covalent radii
of every atom on the path; pairs further apart get 1e6 (no interaction).
"""
import numpy as np
from rdkit import Chem
from scipy.linalg import eigvalsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

EPS = 1e-12
MAX_HOPS = 5        # reference 2D scope: bond paths longer than this don't interact
FAR = 1e6           # distance assigned to non-interacting pairs (reference default)

# symbol: (nuclear charge, valence electrons, polarizability [a.u.],
#          Pauling electronegativity, 1st ionization potential [eV],
#          covalent radius [A], van der Waals radius [A])
# Generated verbatim from SpectralScore globalvars.py (valence = sp + d + f).
ELEMENT_PROPS = {
    "H": (1, 1, 4.50711, 2.2, 13.6, 0.37, 1.2),
    "D": (1, 1, 4.50711, 2.2, 13.6, 0.37, 1.2),
    "He": (2, 2, 1.38375, 0, 24.59, 0.46, 1.43),
    "Li": (3, 1, 164.1125, 0.98, 5.39, 1.33, 2.12),
    "Be": (4, 2, 37.74, 1.57, 9.32, 1.02, 1.98),
    "B": (5, 3, 20.5, 2.04, 8.3, 0.85, 1.91),
    "C": (6, 4, 11.3, 2.55, 11.26, 0.77, 1.77),
    "N": (7, 5, 7.4, 3.04, 14.53, 0.75, 1.66),
    "O": (8, 6, 5.3, 3.44, 13.62, 0.73, 1.5),
    "F": (9, 7, 3.74, 3.98, 17.42, 0.71, 1.46),
    "Ne": (10, 8, 2.66, 0, 21.56, 0.67, 1.58),
    "Na": (11, 1, 162.7, 0.93, 5.14, 1.55, 2.5),
    "Mg": (12, 2, 71.2, 1.31, 7.65, 1.39, 2.51),
    "Al": (13, 3, 57.8, 1.61, 5.99, 1.26, 2.25),
    "Si": (14, 4, 37.3, 1.9, 8.15, 1.16, 2.19),
    "P": (15, 5, 25, 2.19, 10.49, 1.06, 1.9),
    "S": (16, 6, 19.4, 2.58, 10.36, 1.02, 1.89),
    "Cl": (17, 7, 14.6, 3.16, 12.97, 0.99, 1.82),
    "Ar": (18, 8, 11.083, 0, 15.76, 0.96, 1.83),
    "K": (19, 1, 289.7, 0.82, 4.34, 1.96, 2.73),
    "Ca": (20, 2, 160.8, 1.0, 6.11, 1.71, 2.62),
    "Sc": (21, 3, 97, 1.36, 6.56, 1.7, 2.58),
    "Ti": (22, 4, 100, 1.54, 6.83, 1.36, 2.46),
    "V": (23, 5, 87, 1.63, 6.75, 1.22, 2.42),
    "Cr": (24, 6, 83, 1.66, 6.77, 1.27, 2.45),
    "Mn": (25, 7, 68, 1.55, 7.43, 1.39, 2.45),
    "Fe": (26, 8, 62, 1.83, 7.9, 1.25, 2.44),
    "Co": (27, 9, 55, 1.88, 7.88, 1.26, 2.4),
    "Ni": (28, 10, 49, 1.91, 7.64, 1.21, 2.4),
    "Cu": (29, 11, 46.5, 1.9, 7.73, 1.38, 2.38),
    "Zn": (30, 12, 38.67, 1.65, 9.39, 1.31, 2.39),
    "Ga": (31, 13, 50, 1.81, 6, 1.24, 2.32),
    "Ge": (32, 14, 40, 2.01, 7.9, 1.21, 2.29),
    "As": (33, 15, 30, 2.18, 9.79, 1.21, 1.88),
    "Se": (34, 16, 28.9, 2.55, 9.75, 1.16, 1.82),
    "Br": (35, 17, 21, 2.96, 11.81, 1.14, 1.86),
    "Kr": (36, 18, 16.78, 0, 14, 1.17, 2.25),
    "Rb": (37, 1, 319.8, 0.82, 4.18, 2.1, 3.21),
    "Sr": (38, 2, 197.2, 0.95, 5.69, 1.85, 2.84),
    "Y": (39, 3, 162, 1.22, 6.22, 1.63, 2.75),
    "Zr": (40, 4, 112, 1.33, 6.63, 1.54, 2.52),
    "Nb": (41, 5, 98, 1.6, 6.76, 1.47, 2.56),
    "Mo": (42, 6, 87, 2.16, 7.09, 1.38, 2.45),
    "Tc": (43, 7, 79, 2.1, 7.12, 1.56, 2.44),
    "Ru": (44, 8, 72, 2.2, 7.36, 1.25, 2.46),
    "Rh": (45, 9, 66, 2.28, 7.46, 1.25, 2.44),
    "Pd": (46, 10, 26.14, 2.2, 8.34, 1.2, 2.15),
    "Ag": (47, 11, 55, 1.93, 7.58, 1.28, 2.53),
    "Cd": (48, 12, 46, 1.69, 8.99, 1.48, 2.49),
    "In": (49, 13, 65, 1.78, 5.79, 1.42, 2.43),
    "Sn": (50, 14, 53, 1.96, 7.34, 1.4, 2.42),
    "Sb": (51, 15, 43, 2.05, 8.61, 1.4, 2.47),
    "Te": (52, 16, 38, 2.1, 9.01, 1.38, 1.99),
    "I": (53, 17, 32.9, 2.66, 10.45, 1.4, 2.04),
    "Xe": (54, 18, 27.32, 0, 12.13, 1.31, 2.06),
    "Cs": (55, 1, 400.9, 0.79, 3.89, 2.32, 3.48),
    "Ba": (56, 2, 272, 0.89, 5.21, 1.96, 3.03),
    "La": (57, 3, 215, 1.1, 5.58, 1.69, 2.98),
    "Ce": (58, 4, 205, 1.12, 5.54, 1.63, 2.88),
    "Pr": (59, 5, 216, 1.13, 5.47, 1.76, 2.92),
    "Nd": (60, 6, 208, 1.14, 5.53, 1.74, 2.95),
    "Sm": (62, 8, 192, 1.17, 5.64, 1.72, 2.9),
    "Eu": (63, 9, 184, 1.2, 5.67, 1.68, 2.87),
    "Gd": (64, 10, 158, 1.2, 6.15, 1.69, 2.83),
    "Tb": (65, 11, 170, 1.1, 5.86, 1.68, 2.79),
    "Dy": (66, 12, 163, 1.22, 5.94, 1.67, 2.87),
    "Ho": (67, 13, 156, 1.23, 6.02, 1.66, 2.81),
    "Er": (68, 14, 150, 1.24, 6.11, 1.65, 2.83),
    "Tm": (69, 15, 144, 1.25, 6.18, 1.64, 2.79),
    "Yb": (70, 16, 139, 1.1, 6.25, 1.7, 2.8),
    "Lu": (71, 17, 137, 1.27, 5.43, 1.62, 2.74),
    "Hf": (72, 18, 103, 1.3, 6.83, 1.5, 2.63),
    "Ta": (73, 19, 74, 1.5, 7.55, 1.38, 2.53),
    "W": (74, 20, 68, 2.36, 7.86, 1.46, 2.57),
    "Re": (75, 21, 62, 1.9, 7.83, 1.59, 2.49),
    "Os": (76, 22, 57, 2.2, 8.44, 1.28, 2.48),
    "Ir": (77, 23, 54, 2.2, 8.97, 1.37, 2.41),
    "Pt": (78, 24, 48, 2.28, 8.96, 1.23, 2.29),
    "Au": (79, 25, 36, 2.54, 9.23, 1.24, 2.32),
    "Hg": (80, 26, 33.91, 2.0, 10.44, 1.49, 2.45),
    "Tl": (81, 27, 50, 1.62, 6.11, 1.44, 2.47),
    "Pb": (82, 28, 47, 2.33, 7.42, 1.44, 2.6),
    "Bi": (83, 29, 48, 2.02, 7.29, 1.51, 2.54),
    "Po": (84, 30, 44, 2.0, 8.42, 1.4, 2.93),
    "At": (85, 31, 42, 2.2, 9.32, 1.5, 2.88),
    "Rn": (86, 32, 35, 0, 10.75, 1.5, 2.71),
    "Fr": (87, 1, 317.8, 0.7, 4.07, 3.48, 2.82),
    "Ra": (88, 2, 246, 0.9, 5.28, 2.01, 2.81),
    "Ac": (89, 3, 203, 1.1, 5.38, 1.86, 2.8),
    "Th": (90, 4, 217, 1.3, 6.31, 1.75, 2.93),
    "Pa": (91, 5, 154, 1.5, 5.89, 2.0, 2.88),
    "U": (92, 6, 129, 1.38, 6.19, 1.7, 2.71),
    "Np": (93, 7, 151, 1.36, 6.27, 1.9, 2.82),
    "Pu": (94, 8, 132, 1.28, 6.03, 1.75, 2.81),
    "Am": (95, 9, 131, 1.3, 5.97, 1.8, 2.83),
    "Cm": (96, 10, 144, 1.3, 5.99, 1.69, 3.05),
    "Bk": (97, 11, 125, 1.3, 6.2, 1.68, 3.4),
    "Cf": (98, 12, 122, 1.3, 6.28, 1.68, 3.05),
    "Es": (99, 13, 118, 1.3, 6.37, 1.65, 2.7),
    "Fm": (100, 14, 113, 1.3, 6.5, 1.67, 2.7),
}


class SpectralFingerprint:
  """Spectral (graph-Laplacian eigenvalue) fingerprint with the skfp
  ``.transform(mols) -> np.ndarray`` interface, so it plugs into
  ``get_fingerprints`` / ``SmilesFeaturizer`` like the other featurizers.

  Args:
    n_eigs: eigenvalues kept per channel (feature length = 4 * n_eigs).
    use_3D: True -> conformer coordinates (mols must carry conformers);
      False -> bond-path (2D) distances, no conformers needed.
    n_jobs: accepted for interface parity; featurization is fast serially.
  """

  def __init__(self, n_eigs: int = 16, use_3D: bool = True, n_jobs: int = -1):
    self.n_eigs = n_eigs
    self.use_3D = use_3D
    self.n_jobs = n_jobs

  def fit(self, X, y=None):
    return self

  def transform(self, mols):
    return np.array([self._fingerprint(m) for m in mols], dtype=np.float32)

  def _fingerprint(self, mol):
    if not self.use_3D:
      mol = Chem.AddHs(mol)
    props = np.array([ELEMENT_PROPS[a.GetSymbol()] for a in mol.GetAtoms()],
                     dtype=np.float64)
    Z, val, pol, chi, ip, covr, vdw = props.T

    if self.use_3D:
      conf_id = mol.GetIntProp("conf_id") if mol.HasProp("conf_id") else -1
      dist = squareform(pdist(mol.GetConformer(conf_id).GetPositions()))
    else:
      # min-radius-sum path among the fewest-bond paths: every edge costs a
      # large constant (so hop count dominates) plus the mean radius of its
      # ends. Summed over a path that is BIG*hops + interior radii + half of
      # each endpoint radius; adding the other halves gives the reference
      # "sum of covalent radii of all atoms on the path".
      BIG = 100.0
      i, j = np.nonzero(Chem.GetAdjacencyMatrix(mol))
      w = shortest_path(csr_matrix((BIG + (covr[i] + covr[j]) / 2, (i, j)),
                                   shape=(len(Z), len(Z))))
      hops = Chem.GetDistanceMatrix(mol)
      dist = w - BIG * hops + (covr[:, None] + covr[None, :]) / 2
      dist[(hops > MAX_HOPS) | ~np.isfinite(w)] = FAR

    # Pairwise channel weights (reference get_spectrum); distances are made
    # dimensionless by the pair's covalent-radius sum.
    ds = dist / np.maximum(covr[:, None] + covr[None, :], EPS)
    vdw_sum = vdw[:, None] + vdw[None, :]
    channels = [
        np.outer(Z, Z) / (ds + EPS),                                    # electrostatics
        np.outer(ip, ip) / np.maximum(ip[:, None] + ip[None, :], EPS)
        * np.outer(pol, pol) / np.maximum(ds, EPS) ** 6,                # dispersion
        np.outer(val, val) * np.exp(-ds * np.abs(chi[:, None] - chi[None, :])),  # covalent
        np.maximum(0, (vdw_sum - dist) / (vdw_sum + EPS)),              # sterics
    ]

    spectra = []
    for A in channels:
      np.fill_diagonal(A, 0.0)
      frob = np.linalg.norm(A)
      if frob > EPS:
        A = A / frob
      eigs = np.sort(eigvalsh(np.diag(A.sum(axis=1)) - A))
      spec = np.concatenate([eigs[1:2], eigs[2:][::-1]])[:self.n_eigs]
      spectra.append(np.pad(spec, (0, self.n_eigs - len(spec))))
    return np.concatenate(spectra)
