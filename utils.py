"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem.Scaffolds import MurckoScaffold
# import torch.nn as nn
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
import numpy as np
import hyp
import sys
import os

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# import sascorer


def get_fingerprint(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    if smiles is None:
        return np.zeros((hyp.fingerprint_length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((hyp.fingerprint_length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, hyp.fingerprint_radius, hyp.fingerprint_length
    )

    arr = np.zeros((1,))

    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.

  Note that this is not a count of valence electrons, but a count of the
  maximum number of bonds each element will make. For example, passing
  atom_types ['C', 'H', 'O'] will return [4, 1, 2].

  Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

  Returns:
    List of integer atom valences.
  """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
    ]
