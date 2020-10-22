import torch
from config.encoder import Args, Path
from torch_geometric.data import DataLoader
from torch.nn import functional as F


def preprocess(dataset_name, args):
    return _PREPROCESS[dataset_name.lower()](args)


def _preprocess_tox21(args):
    from torch_geometric.datasets import TUDataset
    notox_mol_to_retain = 950

    def pre_filter(sample):
        if sample.y == 1:
            return True # retain

        nonlocal notox_mol_to_retain
        if sample.y == 0 and notox_mol_to_retain > 0:
            notox_mol_to_retain = notox_mol_to_retain - 1
            return True

        return False

    dataset = TUDataset(
        Path.data('Balanced-Tox21'),
        name='Tox21_AhR_training',
        pre_filter=pre_filter
    )

    dataset = dataset.shuffle()
    n = len(dataset) // args.test_split
    train = dataset[n:]
    val = dataset[:n]

    return (
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val,   batch_size=args.batch_size),
        train,
        val,
        train.num_features,
        train.num_classes,
    )


def _preprocess_esol(args):
    from torch_geometric.datasets import MoleculeNet

    dataset = MoleculeNet(
        Path.data('MoleculeNet'),
        name='ESOL'
    )

    dataset = dataset.shuffle()
    n = len(dataset) // args.test_split

    train = dataset[n:]
    val = dataset[:n]

    return (
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val,   batch_size=args.batch_size),
        train,
        val,
        train.num_features,
        train.num_classes,
    )


_PREPROCESS = {
    'tox21': _preprocess_tox21,
    'esol': _preprocess_esol
}
