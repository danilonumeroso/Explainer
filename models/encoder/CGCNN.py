import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, CGConv
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmmp, global_add_pool as gap


class CGCNN(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_edge_features,
            num_output
    ):
        super(EncoderV2, self).__init__()

        self.conv1 = CGConv(num_input, num_edge_features)
        self.conv2 = CGConv(num_input, num_edge_features)
        self.conv3 = CGConv(num_input, num_edge_features)

        self.lin1 = torch.nn.Linear(27, 126)
        self.lin2 = torch.nn.Linear(126, 64)
        self.lin3 = torch.nn.Linear(64, num_output)

    def forward(self, x, edge_index, edge_attr, batch):

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = torch.cat([gmmp(x, batch), gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x2 = torch.cat([gmmp(x, batch), gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmmp(x, batch), gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        encoding = x
        x = F.relu(self.lin1(x))
        F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return x, encoding.detach()
