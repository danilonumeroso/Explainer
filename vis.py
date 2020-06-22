import argparse as ap
import sys

parser = ap.ArgumentParser(description='Visualisation script')

parser.add_argument('--file', required=True)
parser.add_argument('--encoder', default='Encoder')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--figure', dest='figure', action='store_true')
args = parser.parse_args()

sys.argv = sys.argv[:1]

import torch
import json
import utils
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.datasets import TUDataset
from config.explainer import Path
from config import filter
from config.explainer import Elements
from models import GNNExplainerAdapter
import matplotlib.pyplot as plt
import numpy as np

Encoder   = utils.get_encoder(args.encoder)
Explainer = GNNExplainerAdapter(Encoder, epochs=args.epochs)


def inv_nll(log_v):
    import math
    return math.exp(log_v)


def mol_details(smiles, description="Molecule"):
    molecule = utils.mol_to_pyg(
        Chem.MolFromSmiles(smiles)
    )

    molecule.batch = torch.zeros(
        molecule.x.shape[0]
    ).long()

    cert, _ = Encoder(molecule.x, molecule.edge_index, molecule.batch)
    cert = cert.detach().squeeze()

    print(f"{description} {smiles}")
    print(f"Class 0 certainty {inv_nll(cert[0].item()):.3f}")
    print(f"Class 1 certainty {inv_nll(cert[1].item()):.3f}")

    return molecule, smiles


def show(smiles):
    if not args.figure:
        return

    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToMPL(mol, centerIt=False)
    plt.show()


def gnn_explainer(sfx, mol):
    if not args.figure:
        return

    mol, smiles = mol
    _, edge_mask = Explainer.explain_graph(mol.x, mol.edge_index)

    labels = {
        i: Elements(e).name
        for i, e in enumerate([x.tolist().index(1) for x in mol.x.numpy()])
    }

    Explainer.visualize_subgraph(mol.edge_index, edge_mask,
                                 len(mol.x), labels=labels)
    plt.show()


with open(args.file, 'r') as f:
    counterfacts = json.load(f)

mols = []

show(counterfacts['original'])

gnn_explainer(
    str(counterfacts['index']) + "_ORIGINAL",
    mol_details(counterfacts['original'], "[ORIGINAL] ")
)

for mol in counterfacts['counterfacts']:
    mols.append(mol_details(mol['smiles']))
    show(mol['smiles'])

for mol in mols:
    gnn_explainer(str(counterfacts['index']), mol)
