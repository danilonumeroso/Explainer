"""Tools for manipulating graphs and converting from atom and pair features."""

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from torch_geometric.data import Data

from config.explainer import Args, Elements, Edges, EdgesToRDKit

import numpy as np
import sys
import os
import torch

Hyperparams = Args()

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))


def morgan_fingerprint(smiles):
    if smiles is None:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        Hyperparams.fingerprint_radius,
        Hyperparams.fingerprint_length
    )


def numpy_morgan_fingerprint(smiles):
    fingerprint = morgan_fingerprint(smiles)

    if fingerprint is None:
        return np.zeros((Hyperparams.fingerprint_length,))

    arr = np.zeros((1,))

    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)

    return arr


def atom_valences(atom_types):
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def pyg_to_smiles(pyg_mol):
    mol = Chem.RWMol()

    X = pyg_mol.x.numpy().tolist()
    X = [
        Chem.Atom(Elements(x.index(1)).name)
        for x in X
    ]

    E = pyg_mol.edge_index.t()

    for x in X:
        mol.AddAtom(x)

    if pyg_mol.edge_attr is None:
        for u, v in E:
            u = u.item()
            v = v.item()
            if mol.GetBondBetweenAtoms(u, v):
                continue
            mol.AddBond(u, v, Chem.BondType.SINGLE)
    else:
        for (u, v), attr in zip(E, pyg_mol.edge_attr):
            u = u.item()
            v = v.item()
            attr = attr.numpy().tolist()
            attr = EdgesToRDKit(attr.index(1))

            if mol.GetBondBetweenAtoms(u, v):
                continue

            mol.AddBond(u, v, attr)

    return Chem.MolToSmiles(mol)


def mol_to_pyg(molecule):
    X = torch.nn.functional.one_hot(
        torch.tensor([
            Elements[atom.GetSymbol()].value
            for atom in molecule.GetAtoms()
        ]),
        num_classes=50
    ).float()

    E = torch.tensor([
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        for bond in molecule.GetBonds()
    ] + [
        [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        for bond in molecule.GetBonds()
    ]).t()

    edge_attr = torch.nn.functional.one_hot(
        torch.tensor([
            Edges(bond.GetBondType())
            for bond in molecule.GetBonds()
        ] + [
            Edges(bond.GetBondType())
            for bond in molecule.GetBonds()
        ]),
        num_classes=4
    ).float()

    pyg_mol = Data(x=X, edge_index=E, edge_attr=edge_attr)
    pyg_mol.batch = torch.zeros(X.shape[0]).long()

    return pyg_mol


def get_encoder(name="Encoder"):

    if name == "Encoder":
        from models.encoder.Encoder import Encoder

        encoder = Encoder(50, 128, 2)
        encoder.load_state_dict(
            torch.load("ckpt/Encoder.pth")
        )
        encoder.eval()
        return encoder
    elif name == "EncoderV2":
        from models.encoder.EncoderV2 import EncoderV2

        encoder = EncoderV2(50, 4, 2)
        encoder.load_state_dict(
            torch.load("ckpt/Encoder.pth")
        )
        encoder.eval()
        return encoder
