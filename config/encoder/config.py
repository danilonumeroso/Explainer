from collections import namedtuple
from os import path

_Hyperparams = namedtuple(
    'Hyperparams',
    [
        'lr',
        'batch_size',
        'test_split',
        'hidden_size',
        'epochs',
        'optimizer',
        'weight',
        'experiment_name',
        'seed'
    ]
)

_Path = namedtuple(
    'Path',
    [
        'data',
        'ckpt'
    ]
)


def Args():
    import argparse as ap
    import torch

    parser = ap.ArgumentParser(description='Encoder Hyperparams')

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'Adamax'], default='Adam')
    parser.add_argument('--split', type=int, default=10)
    parser.add_argument('--weight', nargs='+', type=int, default=[1, 1])
    parser.add_argument('--experiment_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    optim = None
    if args.optimizer == 'SGD':
        optim = torch.optim.SGD
    elif args.optimizer == 'Adamax':
        optim = torch.optim.Adamax
    else:
        optim = torch.optim.Adam

    return _Hyperparams(
        lr=args.lr,
        hidden_size=args.hidden_size,
        optimizer=optim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_split=args.split,
        weight=torch.Tensor(args.weight),
        experiment_name=args.experiment_name,
        seed=args.seed
    )

_BasePath = path.normpath(path.join(
    path.dirname(path.realpath(__file__)),
    '..',
    '..'
))

Path = _Path(
    data=lambda x: path.join(_BasePath, 'data', x),
    ckpt=path.join(
        _BasePath,
        'ckpt'
    )
)
