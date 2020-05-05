from collections import namedtuple
from os import path


_Hyperparams = namedtuple(
    'Hyperparams',
    [
        'start_molecule',
        'eps_start',
        'eps_end',
        'eps_decay',
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
        'update_interval'
    ]
)

_Path = namedtuple(
    'Path',
    [
        'data'
    ]
)

Hyperparams = None


def Args():

    if Hyperparams is not None:
        return Hyperparams

    import argparse as ap

    parser = ap.ArgumentParser(description='Explainer Hyperparams')

    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=int, default=2000)
    parser.add_argument('--optimizer',
                        choices=['Adam', 'SGD', 'Adamax'],
                        default='Adam')
    parser.add_argument('--polyak', type=float, default=0.995)
    # parser.add_argument('--atom_types', type=float, default=0.995) [C, O, N]
    parser.add_argument('--max_steps_per_episode', type=int, default=1)
    parser.add_argument('--allow_removal', type=bool, default=True)
    parser.add_argument('--allow_no_modification', type=bool, default=True)
    parser.add_argument('--allow_bonds_between_rings', type=bool, default=False)
    # parser.add_argument('--allowed_ring_sizes', type=bool, default=0.995) [3,4,5,6]
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--fingerprint_radius', type=int, default=3)
    parser.add_argument('--fingerprint_length', type=int, default=2048)
    parser.add_argument('--discount', type=bool, default=0.9)
    parser.add_argument('--epochs', type=bool, default=200000)

    args = parser.parse_args()

    return _Hyperparams(
        start_molecule=None,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        optimizer=args.optimizer,
        polyak=args.polyak,
        atom_types=["C", "O", "N"],
        max_steps_per_episode=args.max_steps_per_episode,
        allow_removal=args.allow_removal,
        allow_no_modification=args.allow_no_modification,
        allow_bonds_between_rings=args.allow_bonds_between_rings,
        allowed_ring_sizes=[3, 4, 5, 6],
        replay_buffer_size=args.replay_buffer_size,
        lr=args.lr,
        gamma=args.gamma,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_length=args.fingerprint_length,
        discount=args.discount,
        epochs=args.epochs,
        batch_size=128,
        num_updates_per_it=1,
        update_interval=20
    )


Log = print

_BasePath = path.normpath(path.join(
    path.dirname(path.realpath(__file__)),
    '..',
    '..'
))

Path = _Path(
    data=lambda x: path.join(_BasePath, 'data', x)
)
