import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from datasets import Tox21_AHR

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Tox21_AHR'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Tox21_AHR(path)
data = dataset.data

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(50, 16,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, 16,
                             normalize=not args.use_gdc)

        self.pred  = torch.nn.Linear(16, 1)

    def forward(self):
        input(data)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        input(x.shape)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        input(x.shape)
        x, _ = torch.max(x, dim=1)
        input(x.shape)
        x = self.pred(x)
        x = torch.sigmoid(x)
        input(x.shape)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     loss = F.binary_cross_entropy(model(), data.y.float())
#     loss.backward()
#     optimizer.step()

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    loss = F.binary_cross_entropy(model(), data.y.float())
    loss.backward()
    optimizer.step()
    print(loss.item())

# @torch.no_grad()
# def test():
#     model.eval()
#     logits, accs = model(), []
#     input (logits)
#     pred = logits.max(1)[1]

#     # acc = pred.eq(data.y).sum().item() /
#     # accs.append(acc)
#     return accs

# best_val_acc = test_acc = 0
# for epoch in range(1, 201):
#     train()
#     # train_acc, val_acc, tmp_test_acc = test()
#     # if val_acc > best_val_acc:
#         # best_val_acc = val_acc
#         # test_acc = tmp_test_acc
#     # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#     # print(log.format(epoch, train_acc, best_val_acc, test_acc))
#     train_acc = test()
