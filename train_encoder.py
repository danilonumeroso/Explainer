import torch
import torch.nn.functional as F
import os.path as osp

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import precision, recall
from torch_geometric.utils import f1_score, accuracy

from models.encoder import Encoder
from config.encoder import Args, Path
from config import filter


Hyperparams = Args()

dataset = TUDataset(
    Path.data('Balanced-Tox21'),
    name='Tox21_AhR_training',
    pre_filter=filter
)

dataset = dataset.shuffle()

n = len(dataset) // Hyperparams.test_split

test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=Hyperparams.batch_size)
train_loader = DataLoader(train_dataset, batch_size=Hyperparams.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Encoder(
    num_input=dataset.num_features,
    num_hidden=Hyperparams.hidden_size,
    num_output=dataset.num_classes
).to(device)

# model = EncoderV2(
#     num_input=dataset.num_features,
#     num_edge_features=dataset.num_edge_features,
#     num_output=dataset.num_classes
# ).to(device)


optimizer = Hyperparams.optimizer(
    model.parameters(),
    lr=Hyperparams.lr
)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(output, data.y, weight=Hyperparams.weight.to(device))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    y = torch.tensor([]).long().to(device)
    yp = torch.tensor([]).long().to(device)

    for data in loader:
        data = data.to(device)

        pred, _ = model(data.x, data.edge_index, batch=data.batch)
        pred = pred.max(dim=1)[1]

        y = torch.cat([y, data.y])
        yp = torch.cat([yp, pred])

    k = dataset.num_classes

    return (
        accuracy(y, yp),
        precision(y, yp, k).mean().item(),
        recall(y, yp, k).mean().item(),
        f1_score(y, yp, k).mean().item()
    )


best_acc = (0, 0)
for epoch in range(Hyperparams.epochs):
    loss = train(epoch)
    train_acc, train_prec, train_rec, train_f1 = test(train_loader)
    test_acc, test_prec, test_rec, test_f1 = test(test_loader)

    print(f'Epoch: {epoch}, Loss: {loss:.5f}')

    print(f'Train -> Acc: {train_acc:.5f}  Rec: {train_rec:.5f}  \
    Prec: {train_prec:.5f}  F1: {train_f1:.5f}')

    print(f'Test -> Acc: {test_acc:.5f}  Rec: {test_rec:.5f}  \
    Prec: {test_prec:.5f}  F1: {test_f1:.5f}')

    if best_acc[1] < test_acc:
        best_acc = train_acc, test_acc
        torch.save(
            model.state_dict(),
            osp.join(Path.ckpt,
                     model.__class__.__name__ + ".pth")
        )
        print("New best model saved!")
