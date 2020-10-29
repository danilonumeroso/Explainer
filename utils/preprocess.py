import torch
from config.encoder import Args, Path
from torch_geometric.data import DataLoader, InMemoryDataset
from torch.nn import functional as F
from utils.molecules import check_molecule_validity

def preprocess(dataset_name, args):
    return _PREPROCESS[dataset_name.lower()](args)

def _preprocess_tox21(args):
    from torch_geometric.datasets import TUDataset
    notox_mol_to_retain = 950

    def pre_filter(dataset, percentaga_yes_no=0.5):

        N = len(list(filter(lambda x: x.y == 1, dataset)))

        if sample.y == 1:
            return True # retain

        nonlocal notox_mol_to_retain
        if sample.y == 0 and notox_mol_to_retain > 0:
            notox_mol_to_retain = notox_mol_to_retain - 1
            return True

        return False


    def pad(sample, n_pad):
        sample.x = F.pad(sample.x, (0,n_pad), "constant", 0)
        return sample

    dataset_tr = TUDataset('data/Tox21',
                        name='Tox21_AhR_training',
                        pre_transform=lambda sample: pad(sample, 3))

    dataset_vl = TUDataset('data/Tox21',
                        name='Tox21_AhR_evaluation')

    dataset_ts = TUDataset('data/Tox21',
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
    N_NEGATIVES = N_POSITIVES * 4
    NEGATIVES = NEGATIVES[:N_NEGATIVES]

    dataset = dataset_tr

    dataset.data, dataset.slices = dataset.collate(POSITIVES + NEGATIVES)

    dataset = dataset.shuffle()
    n = len(dataset) // args.test_split
    train = dataset[n:]
    val = dataset[:n]
    test = train[:n]

    return (
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val,   batch_size=args.batch_size),
        DataLoader(test,   batch_size=args.batch_size),
        train,
        val,
        test,
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
