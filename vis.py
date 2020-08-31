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
from torch_geometric.datasets import TUDataset
from config.explainer import Path
from config import filter as filter_
from config.explainer import Elements
from models import GNNExplainerAdapter
import matplotlib.pyplot as plt
# import numpy as np

parser = ap.ArgumentParser(description='Visualisation script')

parser.add_argument('--file', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--encoder', default='Encoder')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--figure', dest='figure', action='store_true')
parser.add_argument('--indexes', nargs='+', type=int,
                    default=[0,1,2,3,4,5])
args = parser.parse_args()

sys.argv = sys.argv[:1]

SAMPLE = re.findall("\d+\.json", args.file)[0]
SAMPLE = re.findall("\d+", SAMPLE)[0]
SAVE_DIR = "counterfacts/drawings/" + args.dataset + "/" + SAMPLE + "/"

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

Encoder = utils.get_encoder(args.dataset, args.encoder)
Explainer = GNNExplainerAdapter(Encoder, epochs=args.epochs)

dataset = None

if args.dataset == "Tox21":
    dataset = TUDataset(
        Path.data('Balanced-Tox21'),
        name='Tox21_AhR_training',
        pre_filter=filter_
    )

elif args.dataset == "Others":
    dataset = None

with open(args.file, 'r') as f:
    counterfacts = json.load(f)

def rescale(value, min_, max_):
    return (value - min_) / (max_ - min_)


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

    cert, encoding = Encoder(molecule.x, molecule.edge_index, molecule.batch)
    cert = cert.detach().squeeze()

    print(f"{description} {smiles}")
    print(f"Class 0 certainty {inv_nll(cert[0].item()):.3f}")
    print(f"Class 1 certainty {inv_nll(cert[1].item()):.3f}")

    return molecule, smiles


def sim(smiles):
    molecule = utils.mol_to_pyg(
        Chem.MolFromSmiles(smiles)
    )

    orig = utils.mol_to_pyg(
        Chem.MolFromSmiles(counterfacts['original'])
    )

    molecule.batch = torch.zeros(
        molecule.x.shape[0]
    ).long()

    orig.batch = torch.zeros(
        orig.x.shape[0]
    ).long()

    _, encoding = Encoder(molecule.x, molecule.edge_index, molecule.batch)
    _, og_encoding = Encoder(orig.x, orig.edge_index, orig.batch)

    S = []
    for mol in dataset:
        if molecule.y == mol.y.item():
            continue

        mol.batch = torch.zeros(
            mol.x.shape[0]
        ).long()
        _, enc_adv = Encoder(mol.x, mol.edge_index, mol.batch)

        S.append(F.cosine_similarity(encoding, enc_adv).item())

    max_ = 1
    min_ = sum(S) / len(S)
    # min_ = min(S)
    sim = F.cosine_similarity(encoding, og_encoding).item()

    print("Embedding similarity: ", rescale(sim, min_, max_))


def show(i, smiles):
    # if not args.figure:
    #     return

    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, SAVE_DIR + SAMPLE + "." + str(i) + ".mol.svg")

    # plt.show()


def gnn_explainer(i, sfx, mol):
    # if not args.figure:
    #     return

    mol, smiles = mol
    _, edge_mask = Explainer.explain_graph(mol.x, mol.edge_index)

    labels = {
        i: Elements(e).name
        for i, e in enumerate([x.tolist().index(1) for x in mol.x.numpy()])
    }

    Explainer.visualize_subgraph(mol.edge_index, edge_mask,
                                 len(mol.x), labels=labels)

    plt.savefig(SAVE_DIR + SAMPLE + "." + str(i) + ".expl.png")
    plt.close()


def process_original_molecule():
    if 0 not in args.indexes:
        print("Orig skipped")
        return

    show(0, counterfacts['original'])

    gnn_explainer(
        0,
        str(counterfacts['index']) + "_ORIGINAL",
        mol_details(counterfacts['original'], "[ORIGINAL] ")
    )

def process_counterfactuals():
    for i, mol in enumerate(counterfacts['counterfacts']):
        if i+1 not in args.indexes:
            print(str(i+1) + "-th skipped")
            continue

        md = mol_details(mol['smiles'])
        sim(mol['smiles'])
        show(i+1, mol['smiles'])
        gnn_explainer(i+1, str(counterfacts['index']), md)

process_original_molecule()
process_counterfactuals()
