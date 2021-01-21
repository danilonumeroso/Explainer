import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer
from torch_geometric.utils import to_networkx
from torch_geometric.nn import MessagePassing

from tqdm import tqdm
from math import sqrt

EPS = 1e-15


class GNNExplainer_(GNNExplainer):

    def __init__(self, prediction_loss, **kwargs):
        super(GNNExplainer_, self).__init__(**kwargs)
        self.prediction_loss = prediction_loss

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn((1,F)) * 0.1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        self.edge_mask = torch.nn.Parameter(torch.randn(E//2) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask.repeat(2)

    def __modify_edge_mask__(self, edge_mask):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__edge_mask__ = edge_mask.repeat(2)

    def __loss__(self, p1, p2):

        loss = self.prediction_loss(p1, p2)
        m = self.edge_mask.sigmoid()

        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.node_feat_mask is not None:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.mean()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def explain_undirected_graph(self, x, edge_index, **kwargs):
        print(self.coeffs)

        self.model.eval()
        self.__clear_masks__()

        batch = torch.zeros(
                x.shape[0]
        ).long()

        # Get the initial prediction.
        prediction = kwargs['prediction']
        node_feat_mask_enable = False if 'node_feats' not in kwargs else kwargs['node_feats']

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        parameters = [self.edge_mask]

        if node_feat_mask_enable:
            parameters.append(self.node_feat_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explaining')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            if self.node_feat_mask is not None:
                h = x * self.node_feat_mask.sigmoid()

            explaining_prediction, _ = self.model(x=x, edge_index=edge_index, batch=batch)
            loss = self.__loss__(explaining_prediction, prediction)
            loss.backward()
            optimizer.step()
            self.__modify_edge_mask__(self.edge_mask)

            if self.log:  # pragma: no cover
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.2f}")

        if self.log:  # pragma: no cover
            pbar.close()

        if self.node_feat_mask is not None:
            node_feat_mask = self.node_feat_mask.detach().sigmoid()

        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        print("Final Pred:", explaining_prediction)
        return node_feat_mask, edge_mask.repeat(2)

    def visualize_subgraph(self, edge_index, edge_mask, num_nodes,
                           threshold=None, **kwargs):

        assert edge_mask.size(0) == edge_index.size(1)

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        print(edge_mask)

        data = Data(edge_index=edge_index, att=edge_mask).to('cpu')
        data.num_nodes = num_nodes
        G = to_networkx(data, edge_attrs=['att'])

        # kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        node_size = kwargs.get('node_size') or 800
        # kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        SCALE = 2
        pos = nx.rescale_layout_dict(nx.kamada_kawai_layout(G), scale=SCALE)

        ax = plt.gca()
        ax.set_xlim((-SCALE - 0.1, SCALE + 0.1))
        ax.set_ylim((-SCALE - 0.1, SCALE + 0.1))

        for source, target, data in G.to_undirected().edges(data=True):
            ax.annotate(
                '',
                xy=pos[target],
                xycoords='data',
                xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_size) / 2.0,
                    shrinkB=sqrt(node_size) / 2.0,
                    connectionstyle="arc3,rad=0.00",
                ))

        nx.draw_networkx_labels(G, pos, **kwargs)
        return ax, G
