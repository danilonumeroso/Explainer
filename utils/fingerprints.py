import numpy as np
import torch
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import AllChem
from utils.molecules import mol_from_smiles, mol_to_smiles

class Fingerprint:
    def __init__(self, fingerprint, fp_length):
        self.fp = fingerprint
        self.fp_len = fp_length

    def is_valid(self):
        return self.fingerprint is None

    def numpy(self):
        np_ = np.zeros((1,))
        ConvertToNumpyArray(self.fp, np_)
        return np_

    def tensor(self):
        return torch.as_tensor(self.numpy())


def morgan_bit_fingerprint(molecule, fp_len, fp_rad):
    m = molecule
    if isinstance(molecule, str):
        molecule = mol_from_smiles(molecule)

    if molecule is None:
        print(m)
        input("NOOOOOOOOOOOOOOOOONE")
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, fp_rad, fp_len)
    return Fingerprint(fp, fp_len)


def morgan_count_fingerprint(molecule, fp_len, fp_rad, bitInfo=None):
    if isinstance(molecule, str):
        molecule = mol_from_smiles(molecule)

    fp = AllChem.GetHashedMorganFingerprint(molecule, fp_rad, fp_len, bitInfo=bitInfo)
    return Fingerprint(fp, fp_len)
