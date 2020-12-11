import argparse
import json
import matplotlib.pyplot as plt

from rdkit.Chem import Draw
from models import GNNExplainer_
from utils import get_split, get_dgn, mol_to_tox21_pyg, mol_to_esol_pyg, explain_loss, mol_from_smiles

parser = argparse.ArgumentParser(description='Visualisation script')

parser.add_argument('--dataset')
parser.add_argument('--sample', default="0")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--experiment', default='test')
parser.add_argument('--figure', dest='figure', action='store_true')
args = parser.parse_args()

SAVE_DIR = f"runs/{args.dataset}/{args.experiment}/counterfacts/"

SAMPLE = args.sample
og = None

with open(SAVE_DIR + '/' + SAMPLE + '.json' , 'r') as f:
    cfs = json.load(f)
    cfs = list(filter(lambda mol: mol['marker'] in ['og', 'cf'], cfs))


GCNN = get_dgn(args.dataset, args.experiment)
explainer = GNNExplainer_(model=GCNN, prediction_loss=explain_loss[args.dataset.lower()], epochs=args.epochs)
dataset = get_split(args.dataset, 'test', args.experiment)

Y = dataset[int(SAMPLE)].y.item()

for idx, mol in enumerate(cfs):
    print('[' + mol['marker'].upper() + ']', mol['smiles'])
    print('Prediction:', mol['prediction']['output'])
    print('Similarity:', mol['reward_sim'] if 'reward_sim' in mol else '-')

    if args.figure:
        Draw.MolToFile(mol_from_smiles(mol['smiles']),
                       SAVE_DIR + SAMPLE + "." + str(idx) + ".mol.svg")

if args.figure:
    transform = mol_to_tox21_pyg if args.dataset.lower() == 'tox21' else mol_to_esol_pyg
    for mol in cfs[1:]:
        data = transform(mol['smiles'])
        node_feat_mask, edge_mask = explainer.explain_undirected_graph(data.x, data.edge_index, prediction=mol['prediction']['for_explanation'])

        labels = {} # TODO: extract labels the right way
        #     labels = {
        #         i: Elements(e).name
        #         for i, e in enumerate([x.tolist().index(1) for x in mol.x.numpy()])
        #     }

        explainer.visualize_subgraph(mol.edge_index, edge_mask,
                                 len(mol.x), labels=labels)

        plt.axis('off')
        plt.savefig(SAVE_DIR + SAMPLE + "." + str(i) + ".expl.svg",
                    bbox_inches='tight',
                    transparent=True)
        plt.close()
