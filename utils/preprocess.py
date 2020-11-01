import torch
from config.encoder import Args, Path
from torch_geometric.data import DataLoader, InMemoryDataset
from torch.nn import functional as F
from utils.molecules import check_molecule_validity
from torch_geometric.datasets import TUDataset

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

    else:
        return None


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

    dataset_list = (
        [dataset_tr.get(idx) for idx in range(len(dataset_tr))] +
        [dataset_vl.get(idx) for idx in range(len(dataset_vl))] +
        [dataset_ts.get(idx) for idx in range(len(dataset_ts))]
    )

    dataset_list = list(filter(check_molecule_validity, dataset_list))

    POSITIVES = list(filter(lambda x: x.y == 1, dataset_list))
    NEGATIVES = list(filter(lambda x: x.y == 0, dataset_list))
    N_POSITIVES = len(POSITIVES)
    N_NEGATIVES = N_POSITIVES
    NEGATIVES = NEGATIVES[:N_NEGATIVES]

    dataset = dataset_tr

    dataset.data, dataset.slices = dataset.collate(POSITIVES + NEGATIVES)

    dataset = dataset.shuffle()
    n = len(dataset) // args.test_split
    train = dataset[n:]
    val = dataset[:n]
    test = train[:n]

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
