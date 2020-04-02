import os
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
import numpy as np
import pandas as pd
import torch

class Tox21_AHR(InMemoryDataset):

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/'

    def __init__(
        self,
        root,
        scope='graph', # graph | node | edge
        transform=None,
        pre_transform=None
    ):
        self.scope = scope
        super(Tox21_AHR, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __repr__(self):
        return "Tox21_AHR"

    @property
    def raw_dir_names(self):
        return [
            'Tox21_AhR_training',
            'Tox21_AhR_testing',
            'Tox21_AhR_evaluation'
        ]

    @property
    def processed_file_names(self):
        return 'Tox21_AhR.pt'

    @property
    def raw_file_names(self):
        suffix = [
            'graph_indicator',
            'A',
            'edge_labels',
            'graph_labels',
            'node_labels'
        ]

        return [
            f'{pfx}/{pfx}_{sfx}.txt'
            for pfx in self.raw_dir_names
            for sfx in suffix
        ]

    def download(self):
        for raw_name in self.raw_dir_names:
            path = download_url(self.url + raw_name + '.zip', self.raw_dir)

            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):

        raw_graph_indic, raw_A, raw_edge_labels, raw_graph_labels, raw_node_labels = self.raw_paths[:5]

        graph_indic  = np.squeeze(pd.read_csv(raw_graph_indic, header=None).values)
        node_labels  = np.squeeze(pd.read_csv(raw_node_labels, header=None).values)
        edge_labels  = np.squeeze(pd.read_csv(raw_edge_labels, header=None).values)
        graph_labels = np.squeeze(pd.read_csv(raw_graph_labels, header=None).values)
        edges        = pd.read_csv(raw_A, header=None).values

        graphs = [{
            'node_labels': [],
            'edge_labels': [],
            'graph_label': None,
            'edges': []
        } for _ in range(len(graph_labels))]

        for j, edge in enumerate(edges):
            i = graph_indic[edge[0] - 1] - 1
            graphs[i]['edges'].append(edge)

        for i, g in enumerate(graphs):
            try:
                min_node = min(np.array(g['edges']).ravel())

                g['edges'] = [
                    [e[0]-min_node, e[1]-min_node]
                    for e in g['edges']
                ]
            except:
                print(f"Sample {i} is not a graph (no edges found). Skipped.")

        for node, label in enumerate(node_labels):
            i = graph_indic[node] - 1
            graphs[i]['node_labels'].append(label)

        for arc, label in enumerate(edge_labels):
            i = graph_indic[edges[arc][0] - 1] - 1
            graphs[i]['edge_labels'].append(label)

        for i, label in enumerate(graph_labels):
            graphs[i]['graph_label'] = [label]

        data_list = []
        for i, g in enumerate(graphs):
            if len(g['edges']) == 0:
                continue

            g['node_labels'] = torch.tensor(g['node_labels'])
            g['edge_labels'] = torch.tensor(g['edge_labels'])
            g['graph_label'] = torch.tensor(g['graph_label'])
            g['edges'] = torch.tensor(np.transpose(g['edges']))

            X = torch.zeros(len(g['node_labels']), 50)
            data = Data(x=X, edge_index=g['edges'], y=g['node_labels'])

            data.num_nodes = len(g['node_labels'])

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
