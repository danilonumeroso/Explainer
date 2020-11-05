from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from torch_geometric.data import Data
from config.explainer import Args, Elements\
    , Edges, EdgesToRDKit, Path

from torch_geometric.datasets.molecule_net import x_map, e_map
import numpy as np
import sys
import os
import torch
import torch_geometric

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))


def morgan_fingerprint(smiles, fp_length, fp_radius):
    if smiles is None:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        fp_radius,
        fp_length
    )


def numpy_morgan_fingerprint(smiles, fp_length, fp_radius):
    fingerprint = morgan_fingerprint(smiles, fp_length, fp_radius)

    if fingerprint is None:
        return np.zeros((fp_length,))

    arr = np.zeros((1,))

    DataStructs.ConvertToNumpyArray(fingerprint, arr)

    return arr


def atom_valences(atom_types):
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def pyg_to_mol(pyg_mol):
    mol = Chem.RWMol()

    X = pyg_mol.x.numpy().tolist()
    X = [
        Chem.Atom(Elements(x.index(1)).name)
        for x in X
    ]

    E = pyg_mol.edge_index.t()

    for x in X:
        mol.AddAtom(x)

    for (u, v), attr in zip(E, pyg_mol.edge_attr):
        u = u.item()
        v = v.item()
        attr = attr.numpy().tolist()
        attr = EdgesToRDKit(attr.index(1))

        if mol.GetBondBetweenAtoms(u, v):
            continue

        mol.AddBond(u, v, attr)

    return mol

def esol_pyg_to_mol(pyg_mol):
    mol = Chem.RWMol()

    X = pyg_mol.x.numpy().tolist()
    X = [
        Chem.Atom(int(x[0]))
        for x in X
    ]

    E = pyg_mol.edge_index.t()

    for x in X:
        mol.AddAtom(x)

    for (u, v), attr in zip(E, pyg_mol.edge_attr):
        u = u.item()
        v = v.item()
        attr = attr.numpy().tolist()
        attr = attr[0]

        if mol.GetBondBetweenAtoms(u, v):
            continue


        mol.AddBond(u, v, Chem.BondType.values[attr])

    return mol

def pyg_to_smiles(pyg_mol):
    return Chem.MolToSmiles(
        pyg_to_mol(pyg_mol)
    )

def esol_pyg_to_smiles(pyg_mol):
    return Chem.MolToSmiles(
        esol_pyg_to_mol(pyg_mol)
    )

def check_molecule_validity(mol, transform):
    if type(mol) == torch_geometric.data.Data:
        mol = transform(mol)

    return Chem.SanitizeMol(mol, catchErrors=True) == Chem.SANITIZE_NONE

def mol_to_pyg(molecule):
    X = torch.nn.functional.one_hot(
        torch.tensor([
            Elements[atom.GetSymbol()].value
            for atom in molecule.GetAtoms()
        ]),
        num_classes=53
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

def mol_to_esol_pyg(mol):
    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(
            str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

        x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(x.shape[0]).long()

    return data


def get_dgn(dataset, experiment):
    from models.encoder.GCNN import GCNN
    import json

    BasePath = 'runs/' + dataset.lower() + "/" + experiment

    params = None
    with open(BasePath + '/hyperparams.json') as file:
        params = json.load(file)

    if dataset.lower() == "tox21":
        m = GCNN(params['num_input'], params['num_hidden'], params['num_output'])
        m.load_state_dict(
            torch.load(
                BasePath + "/ckpt/GCNN.pth",
                map_location=torch.device('cpu')
            )
        )
        m.eval()
        return m

    elif dataset.lower() == "esol":
        m = GCNN(params['num_input'], params['num_hidden'], params['num_output'])
        m.load_state_dict(
            torch.load(
                BasePath + "/ckpt/GCNN.pth",
                map_location=torch.device('cpu')
            ),

        )
        m.eval()
        return m


def molecule_encoding(model, smile):
    mol = mol_to_pyg(Chem.MolFromSmiles(smile))
    _, encoding = model(mol.x, mol.edge_index, mol.batch)
    return encoding.squeeze()
