import torch
from torch.nn import functional as F

class MolDQN(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_output
    ):
        super(MolDQN, self).__init__()
        self.layers = torch.nn.ModuleList([])

        hs = [1024, 512, 128, 32]

        N = len(hs)

        for i in range(N - 1):
            h, h_next  = hs[i], hs[i+1]
            dim_input  = num_input if i == 0 else h

            self.layers.append(
                torch.nn.Linear(dim_input, h_next)
            )
        self.out = torch.nn.Linear(hs[-1], num_output)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.out(x)

        return x
