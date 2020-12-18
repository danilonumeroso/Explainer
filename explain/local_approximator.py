import json
import torch
import numpy as np

from utils import morgan_bit_fingerprint, morgan_count_fingerprint
from torch_geometric.utils import accuracy, precision, recall, f1_score
from torch.nn import Sequential, functional as F, Linear

sample = "runs/tox21/Adam_LR_0.001_BS_120_HS_256/counterfacts/3.json"
num_input = 40
num_output = 2

data = json.load(open(sample, "r"))

X = torch.stack([
    morgan_count_fingerprint(d['smiles'], num_input, 2).tensor()
    for d in data
]).float()

Y = torch.stack([
    torch.tensor(d['prediction']['class'])
    for d in data
])

print("X = ", X.numpy())
print("Y = ", Y.numpy())

interpretable_model = Sequential(
    Linear(num_input, num_output)
)

optimizer = torch.optim.SGD(interpretable_model.parameters(), lr=1e-2)

for epoch in range(1000 + 1):
    optimizer.zero_grad()

    out = interpretable_model(X)
    loss = F.nll_loss(F.log_softmax(out, dim=-1), Y)
    yp = out.max(dim=1)[1]

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Loss: {loss.item():.4f}")
        print(f"Accuracy: {accuracy(Y, yp):.4f}")
        print(f"Precision: {precision(Y, yp, 2).mean().item():.4f}")
        print(f"Recall: {recall(Y, yp, 2).mean().item():.4f}")
        print(f"F1 Score: {f1_score(Y, yp, 2).mean().item():.4f}")

coeff = interpretable_model[0].weight.abs().max(dim=0)[0].detach().numpy()

for i, value in enumerate(coeff):
    print(f"Feature {i} = {value}")
