import torch
import utils
import numpy as np
import torchvision
import json
import os
import utils

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models.explainer import CF_Tox21, NCF_Tox21, Agent
from config.explainer import Args, Path, Log, Elements
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem
from utils import SortedQueue, morgan_bit_fingerprint
from torch.nn import functional as F

def main():
    Hyperparams = Args()
    BasePath = './runs/tox21/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')

    dataset = utils.get_split('tox21', 'test', Hyperparams.experiment)

    original_molecule = dataset[Hyperparams.sample]
    model_to_explain = utils.get_dgn("Tox21", Hyperparams.experiment)

    out, original_encoding = model_to_explain(original_molecule.x,
                                              original_molecule.edge_index)

    logits = F.softmax(out, dim=-1).detach().squeeze()
    pred_class = logits.argmax().item()

    assert pred_class == original_molecule.y.item()

    smiles = utils.pyg_to_smiles(original_molecule)
    print(f'Molecule: {smiles}')

    atoms_ = [
        Elements(e).name
        for e in np.unique(
            [x.tolist().index(1) for x in original_molecule.x.numpy()]
        )
    ]

    params = {
        # General-purpose params
        'init_mol': smiles,
        'discount_factor': Hyperparams.discount,
        'atom_types': set(atoms_),
        'allow_removal': True,
        'allow_no_modification': False,
        'allow_bonds_between_rings': True,
        'allowed_ring_sizes': set(Hyperparams.allowed_ring_sizes),
        'max_steps': Hyperparams.max_steps_per_episode,
        # Task-specific params
        'original_molecule': original_molecule,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.2,
        'similarity_measure': 'combined'
    }

    N = 20
    cf_queue = SortedQueue(N, sort_predicate=lambda mol: mol['score'])
    cf_env = CF_Tox21(**params)
    cf_env.initialize()

    ncf_queue = SortedQueue(N, sort_predicate=lambda mol: mol['score'])
    ncf_env = NCF_Tox21(**params)
    ncf_env.initialize()

    meg_train(writer, cf_env, cf_queue, marker="cf")
    meg_train(writer, ncf_env, ncf_queue, marker="ncf")

    overall_queue = []
    overall_queue.append({
        'marker': 'og',
        'smiles': smiles,
        'pred_class': original_molecule.y.item(),
        'certainty': logits[original_molecule.y.item()].item()
    })
    overall_queue.extend(cf_queue.data_)
    overall_queue.extend(ncf_queue.data_)

    with open(BasePath + f"/counterfacts/{Hyperparams.sample}.json", "w") as outf:
        json.dump(overall_queue, outf, indent=2)


def meg_train(writer, environment, queue, marker):
    Hyperparams = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(Hyperparams.fingerprint_length + 1, 1, device)

    eps = 1.0
    batch_losses = []
    episodes = 0

    for it in range(Hyperparams.epochs):

        steps_left = Hyperparams.max_steps_per_episode - environment.num_steps_taken
        valid_actions = list(environment.get_valid_actions())

        observations = np.vstack(
            [
                np.append(
                    morgan_bit_fingerprint(
                        smiles,
                        Hyperparams.fingerprint_length,
                        Hyperparams.fingerprint_radius
                    ).numpy(),
                    steps_left
                )
                for smiles in valid_actions
            ]
        )

        observations = torch.as_tensor(observations).float()
        a = agent.action_step(observations, eps)
        action = valid_actions[a]
        result = environment.step(action)

        action_fingerprint = np.append(
            morgan_bit_fingerprint(
                action,
                Hyperparams.fingerprint_length,
                Hyperparams.fingerprint_radius
            ).numpy(),
            steps_left
        )

        _, env_out, done = result
        reward, pred, sim, pred_class = env_out

        writer.add_scalar('Tox21/Reward', reward, it)
        writer.add_scalar('Tox21/Prediction', pred, it)
        writer.add_scalar('Tox21/Similarity', sim, it)

        steps_left = Hyperparams.max_steps_per_episode - environment.num_steps_taken

        action_fingerprints = np.vstack(
            [
                np.append(
                    morgan_bit_fingerprint(
                        act,
                        Hyperparams.fingerprint_length,
                        Hyperparams.fingerprint_radius
                    ).numpy(),
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

            print(f'Episode {episodes}::Final Molecule Reward: {final_reward:.6f} (pred: {pred:.6f}, sim: {sim:.6f})')
            print(f'Episode {episodes}::Final Molecule: {action}')

            queue.insert({
                'marker': marker,
                'smiles': action,
                'pred_class': pred_class,
                'score': final_reward,
                'certainty': pred,
                'similarity': sim
            })

            eps *= 0.9985
            batch_losses = []
            environment.initialize()


if __name__ == '__main__':
    main()
