import json
import matplotlib.pyplot as plt
import typer
import torch
import os
import numpy as np
import networkx as nx

from tqdm import tqdm as tq
from geomloss import SamplesLoss
from config.explainer import Elements
from torch.nn import Sequential, Linear
from torch.nn import functional as F
from torch_geometric.utils import accuracy, precision, recall, f1_score
from pathlib import Path
from rdkit.Chem import Draw
from models import GNNExplainer_
from models.encoder import GCNN
from utils import get_split, get_dgn, mol_to_tox21_pyg, mol_to_esol_pyg, mol_from_smiles, morgan_count_fingerprint

app = typer.Typer(add_completion=False)


def check_path(output_path: Path):
    if not output_path.exists():
        typer.confirm("Output path does not exist, do you want to create it?", abort=True)
        output_path.mkdir(parents=True)


def read_graphs(dataset_path: Path):
    labels = {}
    nx_graphs = {}
    for name in os.listdir(str(dataset_path)):
        if not name.endswith('gexf'):
            continue
        idx, label = name.split('.')[-3:-1]
        idx, label = int(idx), int(label)
        nx_graphs[idx] = nx.read_gexf(dataset_path / name)
        labels[idx] = label
    print('Found %d samples' % len(nx_graphs))
    return nx_graphs, labels

@app.command(name='info')
def info(data_path: Path):
    cfs = []
    with open(data_path, 'r') as f:
        cfs = json.load(f)
        cfs = list(filter(lambda mol: mol['marker'] in ['og', 'cf'], cfs))

    for idx, mol in enumerate(cfs):
        print('[' + mol['marker'].upper() + ']', mol['smiles'])
        print('Prediction:', mol['prediction']['output'])
        print('Similarity:', mol['reward_sim'] if 'reward_sim' in mol else '-')

@app.command(name='GNNExplainer')
def gnn_explainer(dataset_name: str, experiment_name: str,
                  epochs: int, data_path: Path, output_dir: Path):

    check_path(output_dir)

    dataset_name = dataset_name.lower()

    def loss_for_classification(p1, p2):
        p1 = F.softmax(p1, dim=-1).detach().squeeze()
        return p1[1 - p2]

    def loss_for_regression(p1, p2):
        return F.l1_loss(p1, p2)

    cfs = []
    with open(data_path, 'r') as f:
        cfs = json.load(f)
        cfs = list(filter(lambda mol: mol['marker'] in ['og', 'cf'], cfs))

    if dataset_name in ['tox21', 'cycliq', 'cycliq-multi']:
        loss = loss_for_classification
        transform = mol_to_tox21_pyg if dataset_name == 'tox21' else None
    elif dataset_name in ['esol']:
        loss = loss_for_regression
        transform = mol_to_esol_pyg

    GCNN = get_dgn(dataset_name, experiment_name)
    explainer = GNNExplainer_(model=GCNN, prediction_loss=loss, epochs=epochs)
    dataset = get_split(dataset_name, 'test', experiment_name)

    for i, mol in enumerate(cfs):
        data = transform(mol['smiles'])
        node_feat_mask, edge_mask = explainer.explain_undirected_graph(data.x, data.edge_index, prediction=mol['prediction']['for_explanation'])


        labels = {} # TODO: extract labels the right way
        if dataset_name == 'tox21':
            labels = {
                i: Elements(e).name
                for i, e in enumerate([x.tolist().index(1) for x in data.x.numpy()])
            }
        elif dataset_name == 'esol':
            rdkit_mol = Chem.MolFromSmiles(smiles)

            labels = {
                i: s
                for i, s in enumerate([x.GetSymbol() for x in rdkit_mol.GetAtoms()])
            }

        explainer.visualize_subgraph(data.edge_index, edge_mask,
                                 len(data.x), labels=labels)


        plt.axis('off')
        plt.savefig(f"{output_dir}/{i}.expl.svg",
                    bbox_inches='tight',
                    transparent=True)
        plt.close()


@app.command(name='linear')
def linear_model(data_path: Path, num_input: int,
                 num_output: int, epochs: int):
    data = json.load(open(data_path, 'r'))
    X = torch.stack([
        morgan_count_fingerprint(d['smiles'], num_input, 2).tensor()
        for d in data
    ]).float()

    Y = torch.stack([
        torch.tensor(d['prediction']['class'])
        for d in data
    ])

    print("X = ", X.numpy())
    print("Y = ", Y.numpy())

    interpretable_model = Sequential(
        Linear(num_input, num_output)
    )

    optimizer = torch.optim.SGD(interpretable_model.parameters(), lr=1e-2)

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = interpretable_model(X)
        loss = F.nll_loss(F.log_softmax(out, dim=-1), Y)
        yp = out.max(dim=1)[1]

        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"Loss: {loss.item():.4f}")
            print(f"Accuracy: {accuracy(Y, yp):.4f}")
            print(f"Precision: {precision(Y, yp, 2).mean().item():.4f}")
            print(f"Recall: {recall(Y, yp, 2).mean().item():.4f}")
            print(f"F1 Score: {f1_score(Y, yp, 2).mean().item():.4f}")

        coeff = interpretable_model[0].weight.abs().max(dim=0)[0].detach().numpy()

    for i, value in enumerate(coeff):
        print(f"Feature {i} = {value}")


@app.command(name='contrast')
def contrast(dataset_path: Path,
             embedding_path: Path,
             output_path: Path,
             loss_str: str = typer.Option('-+s', '--loss'),
             similar_size: int = typer.Option(10),
             distance_str: str = typer.Option('ot', '--distance')):

    check_path(output_path)
    nx_graphs, labels = read_graphs(dataset_path)
    torch.set_num_threads(1)
    graph_embs = {}
    for name in os.listdir(str(embedding_path)):
        if not name.endswith('npy'):
            continue
        graph_num = int(name.split('.')[0])
        embs = np.load(str(embedding_path / name))
        last_idx = len(nx_graphs[graph_num].nodes)
        embs = embs[:last_idx, :]
        graph_embs[graph_num] = embs

    def closest(graph_num, dist, size=1, neg_label=None):
        cur_label = labels[graph_num]
        pos_dists = []
        neg_dists = []
        for i in graph_embs:
            if i == graph_num:
                continue
            #         if pred_labels[i] != dataset[i][1]: # ignore those not predicted correct
            #             continue
            d = dist(graph_num, i)
            if labels[i] != cur_label:
                if neg_label is None or labels[i] == neg_label:
                    neg_dists.append((d, i))
            else:
                pos_dists.append((d, i))
        pos_dists = sorted(pos_dists)
        neg_dists = sorted(neg_dists)
        pos_indices = [i for d, i in pos_dists]
        neg_indices = [i for d, i in neg_dists]

        return pos_indices[:size], neg_indices[:size]

    def loss_verbose(loss_str):
        res = ''
        if '-' in loss_str:
            res = res + '+ loss_neg '
        if '+' in loss_str:
            res = res + '- loss_pos '
        if 's' in loss_str:
            res = res + '+ loss_self '
        return res

    print('Using %s for loss function' % loss_verbose(loss_str))

    if distance_str == 'ot':
        distance = SamplesLoss("sinkhorn", p=1, blur=.01)
    elif distance_str == 'avg':
        distance = lambda x, y: torch.dist(x.mean(axis=0), y.mean(axis=0))

    def graph_distance(g1_num, g2_num):
        k = (min(g1_num, g2_num), max(g1_num, g2_num))
        g1_embs = graph_embs[g1_num]
        g2_embs = graph_embs[g2_num]
        return distance(torch.Tensor(g1_embs), torch.Tensor(g2_embs)).item()

    def explain(graph_num):
        cur_embs = torch.Tensor(graph_embs[graph_num])

        distance = SamplesLoss("sinkhorn", p=1, blur=.01)

        positive_ids, negative_ids = closest(graph_num, graph_distance, size=similar_size)

        positive_embs = [torch.Tensor(graph_embs[i]) for i in positive_ids]
        negative_embs = [torch.Tensor(graph_embs[i]) for i in negative_ids]

        mask = torch.nn.Parameter(torch.zeros(len(cur_embs)))

        learning_rate = 1e-1
        optimizer = torch.optim.Adam([mask], lr=learning_rate)

        if distance_str == 'ot':
            def mydist(mask, embs):
                return distance(mask.softmax(0), cur_embs,
                                distance.generate_weights(embs), embs)
        else:
            def mydist(mask, embs):
                return torch.dist((cur_embs * mask.softmax(0).reshape(-1, 1)).sum(axis=0), embs.mean(axis=0))
        # tq = tqdm(range(50))
        history = []
        for t in range(50):
            loss_pos = torch.mean(torch.stack([mydist(mask, x) for x in positive_embs]))
            loss_neg = torch.mean(torch.stack([mydist(mask, x) for x in negative_embs]))
            loss_self = mydist(mask, cur_embs)

            loss = 0
            if '-' in loss_str:
                loss = loss + loss_neg
            if '+' in loss_str:
                loss = loss - loss_pos
            if 's' in loss_str:
                loss = loss + loss_self

            hist_item = dict(loss_neg=loss_neg.item(), loss_self=loss_self.item(), loss_pos=loss_pos.item(),
                             loss=loss.item())
            history.append(hist_item)
            # tq.set_postfix(**hist_item)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        node_importance = list(1 - mask.softmax(0).detach().numpy().ravel())
        N = nx_graphs[graph_num].number_of_nodes()
        masked_adj = np.zeros((N, N))
        for u, v in nx_graphs[graph_num].edges():
            u = int(u)
            v = int(v)
            masked_adj[u, v] = masked_adj[v, u] = node_importance[u] + node_importance[v]
        return masked_adj

    for gid in tq(graph_embs):
        masked_adj = explain(gid)
        np.save(output_path / ('%s.npy' % gid), masked_adj)

if __name__ == "__main__":
    app()
