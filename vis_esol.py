import argparse as ap
import sys
import torch
import json
import utils
import re
import os

from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.datasets import TUDataset, MoleculeNet
from config.explainer import Path
from config import filter as filter_
from config.explainer import Elements
from models import GNNExplainerESOL
import matplotlib.pyplot as plt
from utils import get_split, get_dgn

parser = ap.ArgumentParser(description='Visualisation script')

parser.add_argument('--sample', default="0")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--experiment', default='test')
parser.add_argument('--figure', dest='figure', action='store_true')
parser.add_argument('--indexes', nargs='+', type=int,
                    default=[0,1,2,3,4,5])
args = parser.parse_args()

sys.argv = sys.argv[:1]
SAVE_DIR = f"runs/esol/{args.experiment}/counterfacts"

SAMPLE = args.sample

Encoder = get_dgn("esol", args.experiment)
Explainer = GNNExplainerESOL(Encoder, epochs=args.epochs)


dataset = get_split('esol', 'test', args.experiment)

with open(SAVE_DIR + '/' + SAMPLE + '.json', 'r') as f:
    counterfacts = json.load(f)

Original = {
    'mol': utils.mol_to_esol_pyg(
        Chem.MolFromSmiles(counterfacts['original'])
    )
}

assert dataset[int(SAMPLE)].smiles == counterfacts['original']

Target = dataset[int(SAMPLE)].y.detach()[0]

Original['pred'], Original['enc'] = Encoder(Original['mol'].x,
                                            Original['mol'].edge_index,
                                            Original['mol'].batch)

Original['pred'] = Original['pred'].detach()[0]
Original['enc'] = Original['enc'].detach()

Distance = lambda x, y: F.l1_loss(x,y).detach().item()

def rescale(value, min_, max_):
    return (value - min_) / (max_ - min_)

def mol_details(smiles, cf=None, description="Molecule"):
    molecule = utils.mol_to_esol_pyg(
        Chem.MolFromSmiles(smiles)
    )

    molecule.batch = torch.zeros(
        molecule.x.shape[0]
    ).long()

    pred, encoding = Encoder(molecule.x, molecule.edge_index, molecule.batch)
    pred = pred.detach().squeeze()
    encoding = encoding.detach()

    sim = F.cosine_similarity(encoding, Original['enc']).item()

    print(f"{description} {smiles}")
    print(f"OriginalPrediction: {Original['pred'].item():.6f}")
    print(f"CounterPrediction: {pred:.6f}")
    print(f"Target: {Target.item():.6f}")

    if cf is not None:
        print(f"Loss: {cf['loss']:.6f}")
        print(f"Gain: {cf['gain']:.6f}")
        print(f"Sim: {cf['sim']:.6f}")

    return molecule, smiles

def show(i, smiles):
    if not args.figure:
        return

    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, SAVE_DIR + SAMPLE + "." + str(i) + ".mol.svg")

def gnn_explainer(i, sfx, mol):
    if not args.figure:
        return

    mol, smiles = mol
    node_mask, edge_mask = Explainer.explain_graph(mol.x, mol.edge_index)

    rdkit_mol = Chem.MolFromSmiles(smiles)

    labels = {
        i: s
        for i, s in enumerate([x.GetSymbol() for x in rdkit_mol.GetAtoms()])
    }


    plt.imshow(node_mask.numpy(), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    filename = SAVE_DIR + SAMPLE + "." + str(i) + ".nodemask.png"
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    plt.close()

    Explainer.visualize_subgraph(mol.edge_index, edge_mask,
                                 len(mol.x), labels=labels)

    plt.axis('off')
    filename = SAVE_DIR + SAMPLE + "." + str(i) + ".expl.svg"
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    plt.close()


def process_original_molecule():
    if 0 not in args.indexes:
        print("Orig skipped")
        return

    show(0, counterfacts['original'])

    gnn_explainer(
        0,
        str(counterfacts['index']) + "_ORIGINAL",
        mol_details(counterfacts['original'], description="[ORIGINAL] ")
    )

def process_counterfactuals():
    for i, cf in enumerate(counterfacts['counterfacts']):
        if i+1 not in args.indexes:
            print(str(i+1) + "-th skipped")
            continue

        md = mol_details(cf['smiles'], cf)
        show(i+1, cf['smiles'])
        gnn_explainer(i+1, str(counterfacts['index']), md)

process_original_molecule()
process_counterfactuals()
