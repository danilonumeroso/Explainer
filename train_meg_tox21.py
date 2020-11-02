import torch
import utils
import numpy as np
import torchvision
import json

from torch_geometric.datasets import TUDataset
from models.explainer import CounterfactualTox21, Agent, CounterfactualESOL
from config.explainer import Args, Path, Log, Elements
from torch.utils.tensorboard import SummaryWriter
from utils import preprocess, molecule_encoding, get_split
from rdkit import Chem


def main():
    Hyperparams = Args()
    BasePath = './runs/tox21/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')
    episodes = 0

    dataset = get_split('tox21', 'test', Hyperparams.experiment)

    molecule = dataset[Hyperparams.sample]
    molecule.batch = torch.zeros(
        molecule.x.shape[0]
    ).long()

    Log(f'Molecule: {utils.pyg_to_smiles(molecule)}')

    utils.TopKCounterfactualsTox21.init(
        utils.pyg_to_smiles(molecule),
        Hyperparams.sample
    )

    atoms_ = [
        Elements(e).name
        for e in np.unique(
            [x.tolist().index(1) for x in molecule.x.numpy()]
        )
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = utils.get_dgn("Tox21", Hyperparams.experiment)

    environment = CounterfactualTox21(
        init_mol=utils.pyg_to_smiles(molecule),
        mol_fp=utils.morgan_fingerprint(
            utils.pyg_to_smiles(molecule),
            Hyperparams.fingerprint_length,
            Hyperparams.fingerprint_radius
        ),
        discount_factor=Hyperparams.discount,
        atom_types=set(atoms_),
        allow_removal=Hyperparams.allow_removal,
        allow_no_modification=Hyperparams.allow_no_modification,
        allow_bonds_between_rings=Hyperparams.allow_bonds_between_rings,
        allowed_ring_sizes=set(Hyperparams.allowed_ring_sizes),
        max_steps=Hyperparams.max_steps_per_episode,
        base_molecule=molecule,
        counterfactual_class=(1 - molecule.y.item()),
        weight_sim=0.2,
        encoder=base_model
    )

    agent = Agent(Hyperparams.fingerprint_length + 1, 1, device)

    environment.initialize()

    eps_threshold = 1.0
    batch_losses = []

    for it in range(Hyperparams.epochs):
        steps_left = Hyperparams.max_steps_per_episode - environment.num_steps_taken

        valid_actions = list(environment.get_valid_actions())

        observations = np.vstack(
            [
                np.append(
                    utils.numpy_morgan_fingerprint(
                        smile,
                        Hyperparams.fingerprint_length,
                        Hyperparams.fingerprint_radius
                    ),
                    steps_left
                )
                for smile in valid_actions
            ]
        )

        observations = torch.as_tensor(observations).float()

        a = agent.action_step(observations, eps_threshold)
        action = valid_actions[a]
        result = environment.step(action)

        action_fingerprint = np.append(
            utils.numpy_morgan_fingerprint(
                action,
                Hyperparams.fingerprint_length,
                Hyperparams.fingerprint_radius
            ),
            steps_left
        )

        _, reward, done = result
        reward, pred, sim = reward

        writer.add_scalar('Tox21/Reward', reward, it)
        writer.add_scalar('Tox21/Prediction', pred, it)
        writer.add_scalar('Tox21/Similarity', sim, it)

        steps_left = Hyperparams.max_steps_per_episode - environment.num_steps_taken

        action_fingerprints = np.vstack(
            [
                np.append(
                    utils.numpy_morgan_fingerprint(
                        act,
                        Hyperparams.fingerprint_length,
                        Hyperparams.fingerprint_radius
                    ),
                    steps_left,
                )
                for act in environment.get_valid_actions()
            ]
        )

        agent.replay_buffer.push(
            torch.as_tensor(action_fingerprint).float(),
            reward,
            torch.as_tensor(action_fingerprints).float(),
            float(result.terminated)
        )

        if it % Hyperparams.update_interval == 0 and len(agent.replay_buffer) >= Hyperparams.batch_size:
            for update in range(Hyperparams.num_updates_per_it):
                loss = agent.train_step(
                    Hyperparams.batch_size,
                    Hyperparams.gamma,
                    Hyperparams.polyak
                )
                loss = loss.item()
                batch_losses.append(loss)

        if done:
            final_reward = reward
            if episodes != 0 and episodes % 2 == 0:
                Log(f'Episode {episodes}::Final Molecule Reward: {final_reward:.6f} (pred: {pred:.6f}, sim: {sim:.6f})')
                Log(f'Episode {episodes}::Final Molecule: {action}')

            utils.TopKCounterfactualsTox21.insert({
                'smiles': action,
                'score': final_reward
            })

            episodes += 1
            eps_threshold *= 0.9985
            batch_losses = []
            environment.initialize()

if __name__ == '__main__':
    main()
