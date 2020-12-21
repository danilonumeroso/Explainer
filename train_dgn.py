import torch
import torch.nn.functional as F
import os
import os.path as osp
import json

from models.encoder import GCNN
from config.encoder import Args
from utils import preprocess, get_dgn, train_cycle_classifier, train_cycle_regressor

args = Args()

torch.manual_seed(args.seed)

BasePath = './runs/' + args.dataset  + '/' + args.experiment_name
if not osp.exists(BasePath):
    os.makedirs(BasePath + "/ckpt")
    os.makedirs(BasePath + "/plots")
    os.makedirs(BasePath + "/splits")
    os.makedirs(BasePath + "/meg_output")
else:
    import shutil
    shutil.rmtree(BasePath + "/plots", ignore_errors=True)
    os.makedirs(BasePath + "/plots")


train_loader, val_loader, test_loader, *extra = preprocess(args.dataset, args)
train_ds, val_ds, test_ds, num_features, num_classes = extra

len_train = len(train_ds)
len_val = len(val_ds)
len_test = len(test_ds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCNN(
    num_input=num_features,
    num_hidden=args.hidden_size,
    num_output=num_classes,
    dropout=args.dropout
).to(device)

with open(BasePath + '/hyperparams.json', 'w') as outfile:
    json.dump({'num_input': num_features,
               'num_hidden': args.hidden_size,
               'num_output': num_classes,
               'dropout': args.dropout,
               'seed': args.seed}, outfile)

optimizer = args.optimizer(
    model.parameters(),
    lr=args.lr
)


if args.dataset.lower() in ['tox21', 'cycliq', 'cycliq-multi']:
    train_cycle_classifier(task=args.dataset.lower(),
                           train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           len_train=len_train,
                           len_val=len_val,
                           len_test=len_test,
                           model=model,
                           optimizer=optimizer,
                           device=device,
                           base_path=BasePath,
                           epochs=args.epochs)

elif args.dataset.lower() in ['esol']:
    train_cycle_regressor(task=args.dataset.lower(),
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          len_train=len_train,
                          len_val=len_val,
                          len_test=len_test,
                          model=model,
                          optimizer=optimizer,
                          device=device,
                          base_path=BasePath,
                          epochs=args.epochs)
