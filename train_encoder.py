import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models import Encoder

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Tox21')
dataset = TUDataset(path, name='Tox21_AhR_training')
dataset = dataset.shuffle()

n = len(dataset) // 10

test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=60)
train_loader = DataLoader(train_dataset, batch_size=60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Encoder(
    num_input=dataset.num_features,
    num_hidden=128,
    num_output=dataset.num_classes
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

best_acc = (0,0)
for epoch in range(1, 201):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    print(f'Epoch: {epoch}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, Test Acc: {test_acc:.5f}')

    if best_acc[1] < test_acc:
        best_acc = train_acc, test_acc
        torch.save(model.state_dict(), "ckpt/Encoder.pth")
        print("New best model saved!")
