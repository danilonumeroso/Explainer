import torch
import torch.nn.functional as F
import os
import os.path as osp
import json

from torch_geometric.utils import precision, recall
from torch_geometric.utils import f1_score, accuracy
from torch.utils.tensorboard import SummaryWriter
from models.encoder import GCNN
from config.encoder import Args
from utils import preprocess, get_dgn

Hyperparams = Args()
torch.manual_seed(Hyperparams.seed)
BasePath = './runs/tox21/' + Hyperparams.experiment_name
if not osp.exists(BasePath):
    os.makedirs(BasePath + "/ckpt")
    os.makedirs(BasePath + "/plots")
    os.makedirs(BasePath + "/splits")
else:
    import shutil
    shutil.rmtree(BasePath + "/plots", ignore_errors=True)
    os.makedirs(BasePath + "/plots")

writer = SummaryWriter(BasePath + '/plots')

train_loader, val_loader, test_loader, *extra = preprocess('tox21', Hyperparams)
train_ds, val_ds, test_ds, num_features, num_classes = extra

len_train = len(train_ds)
len_val = len(val_ds)
len_test = len(test_ds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dropout = 0.4

model = GCNN(
    num_input=num_features,
    num_hidden=Hyperparams.hidden_size,
    num_output=num_classes,
    dropout=dropout
).to(device)

with open(BasePath + '/hyperparams.json', 'w') as outfile:
    json.dump({'num_input': num_features,
               'num_hidden': Hyperparams.hidden_size,
               'num_output': num_classes,
               'dropout': dropout,
               'seed': Hyperparams.seed}, outfile)

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
        loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y, weight=Hyperparams.weight.to(device))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train

def test(loader):
    model.eval()

    y = torch.tensor([]).long().to(device)
    yp = torch.tensor([]).long().to(device)

    loss_all = 0
    for data in loader:
        data = data.to(device)

        pred, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(F.log_softmax(pred, dim=-1), data.y, weight=Hyperparams.weight.to(device))
        pred = pred.max(dim=1)[1]

        y = torch.cat([y, data.y])
        yp = torch.cat([yp, pred])

        loss_all += data.num_graphs * loss.item()

    return (
        accuracy(y, yp),
        precision(y, yp, num_classes).mean().item(),
        recall(y, yp, num_classes).mean().item(),
        f1_score(y, yp, num_classes).mean().item(),
        loss_all
    )


best_acc = (0, 0)
for epoch in range(Hyperparams.epochs):
    loss = train(epoch)
    writer.add_scalar('Loss/train', loss, epoch)
    train_acc, train_prec, train_rec, train_f1, _ = test(train_loader)
    val_acc, val_prec, val_rec, val_f1, l = test(val_loader)

    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Loss/val', l / len_val, epoch)

    print(f'Epoch: {epoch}, Loss: {loss:.5f}')

    print(f'Train -> Acc: {train_acc:.5f}  Rec: {train_rec:.5f}  \
    Prec: {train_prec:.5f}  F1: {train_f1:.5f}')

    print(f'Val -> Acc: {val_acc:.5f}  Rec: {val_rec:.5f}  \
    Prec: {val_prec:.5f}  F1: {val_f1:.5f}')

    if best_acc[1] < val_acc:
        best_acc = train_acc, val_acc

        torch.save(
            model.state_dict(),
            osp.join(BasePath+'/ckpt/',
                     model.__class__.__name__ + ".pth")
        )
        print("New best model saved!")

        with open(BasePath + '/best_result.json', 'w') as outfile:
            json.dump({'train_acc': train_acc,
                       'val_acc': val_acc,
                       'train_rec': train_rec,
                       'val_rec': val_rec,
                       'train_f1': train_f1,
                       'val_f1': val_f1,
                       'train_prec': train_prec,
                       'val_prec': val_prec}, outfile)


model = get_dgn('tox21', Hyperparams.experiment_name)
val_acc, val_prec, val_rec, val_f1, l = test(test_loader)
print(f'TS -> Acc: {val_acc:.5f}  Rec: {val_rec:.5f}  \
Prec: {val_prec:.5f}  F1: {val_f1:.5f}')
