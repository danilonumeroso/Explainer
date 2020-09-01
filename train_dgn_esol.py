import torch
import torch.nn.functional as F
import os
import os.path as osp
import torchvision

from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
from torch_geometric.utils import precision, recall
from torch_geometric.utils import f1_score, accuracy
from torch.utils.tensorboard import SummaryWriter

from models.encoder import GCNN
from config.encoder import Args, Path
from utils import preprocess

Hyperparams = Args()

BasePath = './runs/esol/' + Hyperparams.experiment_name
if not osp.exists(BasePath):
    os.makedirs(BasePath + "/ckpt")

writer = SummaryWriter(BasePath)

train_loader, test_loader, *extra = preprocess('esol', Hyperparams)
train_ds, val_ds, num_features, num_classes = extra

len_train = len(train_ds)
len_val   = len(val_ds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCNN(
    num_input=num_features,
    num_hidden=Hyperparams.hidden_size,
    num_output=num_classes,
    dropout=0.1
).to(device)

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
        output, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(output, data.y)

        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train


def test(epoch, loader, n):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)

        pred, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(pred, data.y).detach()

        loss_all += data.num_graphs * loss.item()

    return loss_all / n


best_error = (+10000, +10000)
for epoch in range(Hyperparams.epochs):
    loss = train(epoch)
    writer.add_scalar('Loss/train', loss, epoch)
    train_error = test(epoch, train_loader, len_train)
    test_error = test(epoch, test_loader, len_val)

    writer.add_scalar('MSE/train', train_error, epoch)
    writer.add_scalar('MSE/test', test_error, epoch)

    print(f'Epoch: {epoch}, Loss: {loss:.5f}')

    print(f'Training Error: {train_error:.5f}')
    print(f'Test Error: {test_error:.5f}')

    if best_error[1] > test_error:
        best_error = train_error, test_error
        torch.save(
            model.state_dict(),
            osp.join(BasePath+'/ckpt/',
                     model.__class__.__name__ + ".pth")
        )
        print("New best model saved!")
