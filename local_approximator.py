import json
import torch
import numpy as np

from utils import morgan_bit_fingerprint
from torch.nn import Sequential, functional as F, Linear

sample = "runs/tox21/test/counterfacts/0.json"
num_input = 4096
num_output = 2

data = json.load(open(sample, "r"))
morgan_bit_fingerprint(data[0]['smiles'], num_input, 2)

X = torch.stack([
    morgan_bit_fingerprint(d['smiles'], num_input, 2).tensor()
    for d in data
]).float()

Y = torch.stack([
    torch.tensor(d['pred_class'])
    for d in data
])

print(X)
print(Y)

interpretable_model = Sequential(
    Linear(num_input, num_output)
)

optimizer = torch.optim.SGD(interpretable_model.parameters(), lr=1e-2)

for epoch in range(200):
    optimizer.zero_grad()

    out = interpretable_model(X)
    loss = F.nll_loss(F.log_softmax(out, dim=-1), Y)
    out = out.max(dim=1)[1]

    loss.backward()
    optimizer.step()

    print(loss)