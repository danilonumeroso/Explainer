import torch
import utils
import numpy as np
import torchvision
import json
import os
import utils

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models.explainer import CounterfactualTox21, Agent, CounterfactualESOL
from config.explainer import Args, Path, Log, Elements
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem


def main():
    Hyperparams = Args()
    BasePath = './runs/tox21/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')
    episodes = 0

    dataset = utils.get_split('tox21', 'test', Hyperparams.experiment)
    dl = DataLoader(dataset, batch_size=None)
    set_ = dataset[torch.randint(0, len(dataset), (25,))]

    original_molecule = dataset[Hyperparams.sample]
    model_to_explain = utils.get_dgn("Tox21", Hyperparams.experiment)

    pred_class, original_encoding = model_to_explain(original_molecule.x,
                                                     original_molecule.edge_index)

    pred_class = pred_class.max(dim=1)[1]

    counterfactual_class = (1 - original_molecule.y.item())
    assert pred_class != counterfactual_class

    smiles = utils.pyg_to_smiles(original_molecule)
    Log(f'Molecule: {smiles}')

    utils.TopKCounterfactualsTox21.init(
        smiles,
        Hyperparams.sample,
        BasePath + '/counterfacts'
    )

    atoms_ = [
        Elements(e).name
        for e in np.unique(
            [x.tolist().index(1) for x in original_molecule.x.numpy()]
        )
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S = [
        model_to_explain(mol.x, mol.edge_index)[1]
        for mol in filter(lambda x: x.y.item() == counterfactual_class, dataset)
    ]
    S = [utils.cosine_similarity(encoding, original_encoding) for encoding in S]

    environment = CounterfactualTox21(
        init_mol=smiles,
        discount_factor=Hyperparams.discount,
        atom_types=set(atoms_),
        allow_removal=True,
        allow_no_modification=False,
        allow_bonds_between_rings=True,
        allowed_ring_sizes=set(Hyperparams.allowed_ring_sizes),
        max_steps=Hyperparams.max_steps_per_episode,
        model_to_explain=model_to_explain,
        counterfactual_class=counterfactual_class,
        weight_sim=0.2,
        similarity_measure="combined",
        similarity_set=S
    )

    agent = Agent(Hyperparams.fingerprint_length + 1, 1, device)

    environment.initialize()

    eps = 1.0
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

        a = agent.action_step(observations, eps)
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
            episodes += 1
            Log(f'Episode {episodes}::Final Molecule Reward: {final_reward:.6f} (pred: {pred:.6f}, sim: {sim:.6f})')
            Log(f'Episode {episodes}::Final Molecule: {action}')

            utils.TopKCounterfactualsTox21.insert({
                'smiles': action,
                'score': final_reward
            })

            eps *= 0.9985
            batch_losses = []
            environment.initialize()

if __name__ == '__main__':
    main()
