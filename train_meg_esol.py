import torch
import utils
import numpy as np

from torch_geometric.datasets import MoleculeNet
from torch.utils.tensorboard import SummaryWriter
from models.explainer import Agent, CounterfactualESOL
from config.explainer import Args, Path, Log, Elements
from rdkit import Chem
from utils import preprocess, get_split


def main():
    Hyperparams = Args()
    BasePath = './runs/esol/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')
    episodes = 0

    dataset = get_split('esol', 'test', Hyperparams.experiment)

    molecule = dataset[Hyperparams.sample]
    molecule.batch = torch.zeros(
        molecule.x.shape[0]
    ).long()


    Log(f'Molecule: {molecule.smiles}')

    utils.TopKCounterfactualsESOL.init(
        molecule.smiles,
        Hyperparams.sample
    )

    mol_ = Chem.MolFromSmiles(molecule.smiles)

    atoms_ = np.unique(
        [x.GetSymbol() for x in mol_.GetAtoms()]
    )

    Hyperparams.atom_types.sort()
    atoms_.sort()

    if not np.array_equal(atoms_, Hyperparams.atom_types):
        Log("[Warn] Hyperparams.atom_types differ from the" +
            " actual atoms composing the real molecule.")

        Hyperparams = Hyperparams._replace(atom_types=atoms_)
        Log("Fixed", Hyperparams.atom_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_to_explain = utils.get_dgn("esol", Hyperparams.experiment)

    environment = CounterfactualESOL(
        init_mol=molecule.smiles,
        target=molecule.y,
        discount_factor=Hyperparams.discount,
        atom_types=set(Hyperparams.atom_types),
        allow_removal=Hyperparams.allow_removal,
        allow_no_modification=Hyperparams.allow_no_modification,
        allow_bonds_between_rings=Hyperparams.allow_bonds_between_rings,
        allowed_ring_sizes=set(Hyperparams.allowed_ring_sizes),
        max_steps=Hyperparams.max_steps_per_episode,
        base_molecule=molecule,
        model_to_explain=model_to_explain,
        weights=[0.8, 0.2]
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
            steps_left,
        )

        _, reward, done = result
        reward, loss_, gain, sim = reward

        writer.add_scalar('ESOL/Reward', reward, it)
        writer.add_scalar('ESOL/Distance', loss_, it)
        writer.add_scalar('ESOL/Similarity', sim, it)

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

        if it % Hyperparams.update_interval == 0 and agent.replay_buffer.__len__() >= Hyperparams.batch_size:
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
                Log(f'Episode {episodes}::Final Molecule Reward: {final_reward:.6f} (loss: {loss_:.6f}, gain: {gain:.6f}, sim: {sim:.6f})')
                Log(f'Episose {episodes}::Mean Loss: {np.array(batch_losses).mean()}')
                Log(f'Episode {episodes}::Final Molecule: {action}')

            utils.TopKCounterfactualsESOL.insert({
                'smiles': action,
                'score': final_reward,
                'loss': loss_,
                'gain': gain,
                'sim': sim
            })

            episodes += 1
            eps_threshold *= 0.9985
            batch_losses = []
            environment.initialize()

if __name__ == '__main__':
    main()
