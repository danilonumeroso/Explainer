import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCNN(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output,
            dropout=0
    ):
        super(GCNN, self).__init__()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.conv1 = GraphConv(num_input, num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden)
        self.conv3 = GraphConv(num_hidden, num_hidden)

        self.lin1 = torch.nn.Linear(num_hidden*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_output)

        self.p = dropout

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = torch.zeros(x.shape[0]).long()

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        node_embs = x

        x = x1 + x2 + x3

        graph_emb = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return x, (node_embs.detach(), graph_emb.detach())
