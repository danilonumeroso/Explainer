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
    original_molecule = dataset[Hyperparams.sample]
    original_molecule.x = original_molecule.x.float()
    model_to_explain = utils.get_dgn("esol", Hyperparams.experiment)

    pred_solub, original_encoding = model_to_explain(original_molecule.x,
                                                     original_molecule.edge_index)

    Log(f'Molecule: {original_molecule.smiles}')

    utils.TopKCounterfactualsESOL.init(
        original_molecule.smiles,
        Hyperparams.sample,
        BasePath + "/counterfacts"
    )

    atoms_ = np.unique(
        [x.GetSymbol() for x in Chem.MolFromSmiles(original_molecule.smiles).GetAtoms()]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S = [
        model_to_explain(mol.x.float(), mol.edge_index)[1]
        for mol in dataset
    ]
    S = [utils.cosine_similarity(encoding, original_encoding) for encoding in S]

    environment = CounterfactualESOL(
        model_to_explain=model_to_explain,
        original_molecule=original_molecule,
        target=original_molecule.y,
        original_prediction=pred_solub,
        weight_sim=0.2,
        similarity_measure="combined",
        similarity_set=S
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
            Log(f'Episode {episodes}::Final Molecule Reward: {final_reward:.6f} (loss: {loss_:.6f}, gain: {gain:.6f}, sim: {sim:.6f})')
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
