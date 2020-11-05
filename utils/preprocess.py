import torch
from config.encoder import Args, Path
from torch_geometric.data import DataLoader, InMemoryDataset
from torch.nn import functional as F
from utils.molecules import check_molecule_validity, pyg_to_mol, esol_pyg_to_mol
from torch_geometric.datasets import TUDataset, MoleculeNet
import random

def pad(sample, n_pad):
    sample.x = F.pad(sample.x, (0,n_pad), "constant", 0)
    return sample


def get_split(dataset_name, split, experiment):
    if dataset_name.lower() == 'tox21':
        ds = TUDataset('data/tox21',
                       name='Tox21_AhR_testing',
                       pre_transform=lambda sample: pad(sample, 2))

        ds.data, ds.slices = torch.load(f"runs/tox21/{experiment}/splits/{split}.pth")

        return ds

    elif dataset_name.lower() == 'esol':

        MoleculeNet.url = 'file://./data/esol-data.zip'
        ds = MoleculeNet(
            'data/esol',
            name='ESOL'
        )

        ds.data, ds.slices = torch.load(f"runs/esol/{experiment}/splits/{split}.pth")

        return ds


def preprocess(dataset_name, args):
    return _PREPROCESS[dataset_name.lower()](args)


def _preprocess_tox21(args):

    dataset_tr = TUDataset('data/tox21',
                        name='Tox21_AhR_training',
                        pre_transform=lambda sample: pad(sample, 3))

    dataset_vl = TUDataset('data/tox21',
                        name='Tox21_AhR_evaluation')

    dataset_ts = TUDataset('data/tox21',
                        name='Tox21_AhR_testing',
                        pre_transform=lambda sample: pad(sample, 2))

    data_list = (
        [dataset_tr.get(idx) for idx in range(len(dataset_tr))] +
        [dataset_vl.get(idx) for idx in range(len(dataset_vl))] +
        [dataset_ts.get(idx) for idx in range(len(dataset_ts))]
    )



    data_list = list(filter(lambda mol: check_molecule_validity(mol, pyg_to_mol), data_list))

    POSITIVES = list(filter(lambda x: x.y == 1, data_list))
    NEGATIVES = list(filter(lambda x: x.y == 0, data_list))
    N_POSITIVES = len(POSITIVES)
    N_NEGATIVES = N_POSITIVES
    NEGATIVES = NEGATIVES[:N_NEGATIVES]

    dataset_full = dataset_tr
    data_list = POSITIVES + NEGATIVES
    random.shuffle(data_list)

    n = len(data_list) // args.test_split
    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]

    train = dataset_tr
    val = dataset_vl
    test = dataset_ts

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices), f'runs/tox21/{args.experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices), f'runs/tox21/{args.experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices), f'runs/tox21/{args.experiment_name}/splits/test.pth')

    return (
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val,   batch_size=args.batch_size),
        DataLoader(test,   batch_size=args.batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )


def _preprocess_esol(args):
    MoleculeNet.url = 'file://./data/esol-data.zip'

    dataset = MoleculeNet(
        'data/esol',
        name='ESOL'
    )

    data_list = (
        [dataset.get(idx) for idx in range(len(dataset))]
    )

    # dataset_list = list(filter(lambda mol: check_molecule_validity(mol, esol_pyg_to_mol), dataset_list))

    random.shuffle(data_list)

    n = len(data_list) // args.test_split

    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]

    train = dataset
    val = dataset.copy()
    test = dataset.copy()

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices), f'runs/esol/{args.experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices), f'runs/esol/{args.experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices), f'runs/esol/{args.experiment_name}/splits/test.pth')


    return (
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val,   batch_size=args.batch_size),
        DataLoader(test,   batch_size=args.batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )

def _preprocess_alchemy(args):
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(
        'data/alchemy',
        name='alchemy_full'
    )


    data_list = (
        [dataset.get(idx) for idx in range(len(dataset))]
    )

    # dataset_list = list(filter(check_molecule_validity, dataset_list))

    random.shuffle(data_list)

    n = len(data_list) // args.test_split

    train_data = data_list[n:]
    val_data = data_list[:n]
    test_data = train_data[:n]

    train = dataset
    val = dataset.copy()
    test = dataset.copy()

    train.data, train.slices = train.collate(train_data)
    val.data, val.slices = train.collate(val_data)
    test.data, test.slices = train.collate(test_data)

    torch.save((train.data, train.slices), f'runs/alchemy/{args.experiment_name}/splits/train.pth')
    torch.save((val.data, val.slices), f'runs/alchemy/{args.experiment_name}/splits/val.pth')
    torch.save((test.data, test.slices), f'runs/alchemy/{args.experiment_name}/splits/test.pth')


    return (
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val,   batch_size=args.batch_size),
        DataLoader(test,   batch_size=args.batch_size),
        train,
        val,
        test,
        max(train.num_features, val.num_features, test.num_features),
        train.num_classes,
    )

_PREPROCESS = {
    'tox21': _preprocess_tox21,
    'esol': _preprocess_esol,
    'alchemy': _preprocess_alchemy,
}
