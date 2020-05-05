import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, CGConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class EncoderV2(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_edge_features,
            num_output
    ):
        super(EncoderV2, self).__init__()

        # self.conv1 = GraphConv(num_input, num_hidden)
        self.conv1 = CGConv(num_input, num_edge_features)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        # self.conv2 = GraphConv(num_hidden, num_hidden)
        self.conv2 = CGConv(num_input, num_edge_features)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        # self.conv3 = GraphConv(num_hidden, num_hidden)
        self.conv3 = CGConv(num_input, num_edge_features)
        # self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(100, num_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        encoding = x

        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin1(x), dim=-1)

        return x, encoding.detach()
