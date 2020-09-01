from collections import namedtuple
from os import path


_Hyperparams = namedtuple(
    'Hyperparams',
    [
        'sample',
        'experiment',
        'seed',
        'test_split',
        'start_molecule',
        'eps_start',
        'eps_end',
        'optimizer',
        'polyak',
        'atom_types',
        'max_steps_per_episode',
        'allow_removal',
        'allow_no_modification',
        'allow_bonds_between_rings',
        'allowed_ring_sizes',
        'replay_buffer_size',
        'lr',
        'gamma',
        'fingerprint_radius',
        'fingerprint_length',
        'discount',
        'epochs',
        'batch_size',
        'num_updates_per_it',
        'update_interval',
        'num_counterfactuals'
    ]
)

_Path = namedtuple(
    'Path',
    [
        'data',
        'counterfacts',
        'drawings'
    ]
)

Hyperparams = None


def Args():

    if Hyperparams is not None:
        return Hyperparams

    import argparse as ap

    parser = ap.ArgumentParser(description='Explainer Hyperparams')

    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--experiment', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_split', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.00)
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'Adamax'],
                        default='Adam')
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--max_steps_per_episode', type=int, default=1)
    parser.add_argument('--allow_removal', type=bool, default=True)
    parser.add_argument('--allow_no_modification', type=bool, default=True)
    parser.add_argument('--allow_bonds_between_rings', type=bool, default=False)
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--fingerprint_radius', type=int, default=2)
    parser.add_argument('--fingerprint_length', type=int, default=4096)
    parser.add_argument('--discount', type=bool, default=0.9)
    parser.add_argument('--epochs', type=int, default=200000)
    parser.add_argument('--num_counterfactuals', type=int, default=5)

    args = parser.parse_args()

    return _Hyperparams(
        sample=args.sample,
        experiment=args.experiment,
        seed=args.seed,
        test_split=args.test_split,
        start_molecule=None,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        optimizer=args.optimizer,
        polyak=args.polyak,
        atom_types=["C", "O", "N"],
        max_steps_per_episode=args.max_steps_per_episode,
        allow_removal=args.allow_removal,
        allow_no_modification=args.allow_no_modification,
        allow_bonds_between_rings=args.allow_bonds_between_rings,
        allowed_ring_sizes=[5, 6],
        replay_buffer_size=args.replay_buffer_size,
        lr=args.lr,
        gamma=args.gamma,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_length=args.fingerprint_length,
        discount=args.discount,
        epochs=args.epochs,
        batch_size=1,
        num_updates_per_it=1,
        update_interval=1,
        num_counterfactuals=args.num_counterfactuals
    )


Log = print

_BasePath = path.normpath(path.join(
    path.dirname(path.realpath(__file__)),
    '..',
    '..'
))

Path = _Path(
    data=lambda x: path.join(_BasePath, 'data', x),
    counterfacts=lambda x, d: path.join(_BasePath, 'counterfacts', 'files', d, x),
    drawings=lambda x, d: path.join(_BasePath, 'counterfacts', 'drawings', d, x),
)
