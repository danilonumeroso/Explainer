import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class EncoderESOL(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output
    ):
        super(EncoderESOL, self).__init__()

        self.conv1 = GraphConv(num_input, num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden)
        self.conv3 = GraphConv(num_hidden, num_hidden)

        self.lin1 = torch.nn.Linear(num_hidden*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_output)

    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        encoding = x

        x = F.relu(self.lin1(x))
        F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return x, encoding.detach()
