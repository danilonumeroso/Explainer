from collections import namedtuple
from os import path


_Hyperparams = namedtuple(
    'Hyperparams',
    [
        'sample',
        'experiment',
        'dataset',
        'seed',
        'optimizer',
        'polyak',
        'max_steps_per_episode',
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
    parser.add_argument('--dataset', default='tox21')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'Adamax'],
                        default='Adam')
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--max_steps_per_episode', type=int, default=1)

    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--fingerprint_radius', type=int, default=2)
    parser.add_argument('--fingerprint_length', type=int, default=4096)
    parser.add_argument('--discount', type=bool, default=0.9)
    parser.add_argument('--epochs', type=int, default=200000)
    parser.add_argument('--num_counterfactuals', type=int, default=15)

    args = parser.parse_args()

    return _Hyperparams(
        sample=args.sample,
        experiment=args.experiment,
        seed=args.seed,
        dataset=args.dataset,
        optimizer=args.optimizer,
        polyak=args.polyak,
        max_steps_per_episode=args.max_steps_per_episode,
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
