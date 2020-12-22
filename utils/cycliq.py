import os
import os.path as osp
import glob

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.io.tu import split, read_file, cat
from torch_geometric.utils import remove_self_loops

class CYCLIQ(InMemoryDataset):

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        super(CYCLIQ, self).__init__(root, transform, pre_transform,
                                     pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        self.data, self.slices = read_cycliq_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

def read_cycliq_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    x = torch.ones((edge_index.max().item() + 1, 10))

    edge_attr = torch.ones((edge_index.size(1), 5))

    y = read_file(folder, prefix, 'graph_labels', torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    return data, slices
