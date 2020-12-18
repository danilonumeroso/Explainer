import torch
import torch.nn.functional as F
import os.path as osp
import json

from torch_geometric.utils import precision, recall
from torch_geometric.utils import f1_score, accuracy
from torch.utils.tensorboard import SummaryWriter

def train_epoch_classifier(model, train_loader, len_train, optimizer, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(F.log_softmax(output, dim=-1), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len_train

def test_classifier(model, loader, device):
    model.eval()

    y = torch.tensor([]).long().to(device)
    yp = torch.tensor([]).long().to(device)

    loss_all = 0
    for data in loader:
        data = data.to(device)
        pred, _ = model(data.x, data.edge_index, batch=data.batch)
        loss = F.nll_loss(F.log_softmax(pred, dim=-1), data.y)
        pred = pred.max(dim=1)[1]

        y = torch.cat([y, data.y])
        yp = torch.cat([yp, pred])

        loss_all += data.num_graphs * loss.item()

    return (
        accuracy(y, yp),
        precision(y, yp, model.num_output).mean().item(),
        recall(y, yp, model.num_output).mean().item(),
        f1_score(y, yp, model.num_output).mean().item(),
        loss_all
    )

def train_cycle_classifier(task, train_loader, val_loader, test_loader, len_train, len_val, len_test,
                           model, optimizer, device, base_path, epochs):

    best_acc = (0, 0)
    writer = SummaryWriter(base_path + '/plots')

    for epoch in range(epochs):
        loss = train_epoch_classifier(model, train_loader, len_train, optimizer, device)
        writer.add_scalar('Loss/train', loss, epoch)
        train_acc, train_prec, train_rec, train_f1, _ = test_classifier(model, train_loader, device)
        val_acc, val_prec, val_rec, val_f1, l = test_classifier(model, val_loader, device)

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
                osp.join(base_path + '/ckpt/',
                         model.__class__.__name__ + ".pth")
            )
            print("New best model saved!")

            with open(base_path + '/best_result.json', 'w') as outfile:
                json.dump({'train_acc': train_acc,
                           'val_acc': val_acc,
                           'train_rec': train_rec,
                           'val_rec': val_rec,
                           'train_f1': train_f1,
                           'val_f1': val_f1,
                           'train_prec': train_prec,
                           'val_prec': val_prec}, outfile)


def train_epoch_regressor(model, train_loader, len_train, optimizer, device):
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


def test_regressor(model, loader, len_loader, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)

        pred, _ = model(data.x.float(), data.edge_index, batch=data.batch)

        loss = F.mse_loss(pred, data.y).detach()

        loss_all += data.num_graphs * loss.item()

    return loss_all / len_loader


def train_cycle_regressor(task, train_loader, val_loader, test_loader, len_train, len_val, len_test,
                          model, optimizer, device, base_path, epochs):

    best_acc = (0, 0)
    writer = SummaryWriter(base_path + '/plots')

    best_error = (+10000, +10000)
    for epoch in range(epochs):
        loss = train_epoch_regressor(model, train_loader, len_train, optimizer, device)
        writer.add_scalar('Loss/train', loss, epoch)
        train_error = test_regressor(model, train_loader, len_train, device)
        val_error = test_regressor(model, val_loader, len_val, device)

        writer.add_scalar('MSE/train', train_error, epoch)
        writer.add_scalar('MSE/test', val_error, epoch)

        print(f'Epoch: {epoch}, Loss: {loss:.5f}')

        print(f'Training Error: {train_error:.5f}')
        print(f'Val Error: {val_error:.5f}')

        if best_error[1] > val_error:
            best_error = train_error, val_error
            torch.save(
                model.state_dict(),
                osp.join(base_path + '/ckpt/',
                         model.__class__.__name__ + ".pth")
            )
            print("New best model saved!")

            with open(base_path + '/best_result.json', 'w') as outfile:
                json.dump({'train_error': train_error,
                           'val_error': val_error}, outfile)
