import torch.nn as nn

class MolDQN(nn.Module):
    def __init__(
            self,
            input_length,
            output_length
    ):
        super(MolDQN, self).__init__()

        self.linear1 = nn.Linear(input_length, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 32)
        self.linear5 = nn.Linear(32, output_length)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = self.linear5(x)

        return x
