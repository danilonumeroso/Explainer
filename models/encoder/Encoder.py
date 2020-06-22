import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Encoder(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output
    ):
        super(Encoder, self).__init__()

        self.conv1 = GraphConv(num_input, num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden)
        self.conv3 = GraphConv(num_hidden, num_hidden)

        self.lin1 = torch.nn.Linear(256, 128)
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
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))

        x = F.log_softmax(self.lin3(x), dim=-1)

        return x, encoding.detach()
