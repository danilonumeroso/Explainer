import torch
import utils
import numpy as np
import torchvision
import json

from torch_geometric.datasets import TUDataset
from models.explainer import Counterfactual, Agent, CounterfactualESOL
from config.explainer import Args, Path, Log, Elements
from torch.utils.tensorboard import SummaryWriter
from utils import preprocess
from rdkit import Chem


def main():
    Hyperparams = Args()
    BasePath = './runs/tox21/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')
    episodes = 0

    with open(BasePath + '/hyperparams.json') as file:
        params = json.load(file)
        torch.manual_seed(params['seed'])

    *_, val, _, _  = preprocess('tox21', Hyperparams)


    torch.manual_seed(torch.initial_seed())

    molecule = val[Hyperparams.sample]
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

    Hyperparams.atom_types.sort()
    atoms_.sort()

    if not np.array_equal(atoms_, Hyperparams.atom_types):
        Log("[Warn] Hyperparams.atom_types differ from the" +
            " actual atoms composing the original molecule.")

        Hyperparams = Hyperparams._replace(atom_types=atoms_)
        Log("Fixed", Hyperparams.atom_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = utils.get_encoder("Tox21", Hyperparams.experiment)

    environment = Counterfactual(
        init_mol=utils.pyg_to_smiles(molecule),
        mol_fp=utils.morgan_fingerprint(
            utils.pyg_to_smiles(molecule),
            Hyperparams.fingerprint_length,
            Hyperparams.fingerprint_radius
        ),
        discount_factor=Hyperparams.discount,
        atom_types=set(Hyperparams.atom_types),
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

    agent = Agent(base_model.num_hidden, 1, device)

    environment.initialize()

    eps_threshold = 1.0
    batch_losses = []


    def smiles_to_pyg(smile):
        mol = utils.mol_to_pyg(Chem.MolFromSmiles(smile))
        _, encoding = base_model(mol.x, mol.edge_index, mol.batch)
        return encoding.squeeze()

    for it in range(Hyperparams.epochs):
        steps_left = Hyperparams.max_steps_per_episode - environment.num_steps_taken

        valid_actions = list(environment.get_valid_actions())

        # Append each valid action to steps_left and store in observations.


        observations = torch.stack(
            [
                # np.append(
                # utils.numpy_morgan_fingerprint(
                #     act,
                #     Hyperparams.fingerprint_length,
                #     Hyperparams.fingerprint_radius
                # )
                #     steps_left,
                # )
                smiles_to_pyg(act)
                for act in valid_actions
            ]
        )  # (num_actions, fingerprint_length)

        # input (observations)
        # for a in valid_actions:
        #     utils.mol_to_pyg(Chem.

        # for act in valid_actions:
        #     input(act)

        a = agent.get_action(observations, eps_threshold)

        # Find out the new state (we store the new state in "action" here.
        # Bit confusing but taken from original implementation)
        action = valid_actions[a]
        # Take a step based on the action
        result = environment.step(action)

        # action_fingerprint = np.append(
        #     utils.numpy_morgan_fingerprint(
        #         action,
        #         Hyperparams.fingerprint_length,
        #         Hyperparams.fingerprint_radius
        #     ),
        #     steps_left,
        # )

        action_fingerprint = smiles_to_pyg(action)
        # utils.numpy_morgan_fingerprint(
        #         action,
        #         Hyperparams.fingerprint_length,
        #         Hyperparams.fingerprint_radius
        #     )
        # # input(action_fingerprint)

        next_state, reward, done = result
        reward, pred, sim = reward

        writer.add_scalar('Tox21/Reward', reward, it)
        writer.add_scalar('Tox21/Prediction', pred, it)
        writer.add_scalar('Tox21/Similarity', sim, it)

        # Compute number of steps left
        steps_left = Hyperparams.max_steps_per_episode - environment.num_steps_taken

        # Append steps_left to the new state and store in next_state
        next_state = smiles_to_pyg(next_state)

        # utils.numpy_morgan_fingerprint(
        #     next_state,
        #     Hyperparams.fingerprint_length,
        #     Hyperparams.fingerprint_radius
        # )
        # (fingerprint_length)

        action_fingerprints = torch.stack(
            [
                # np.append(
                smiles_to_pyg(act)
                # utils.numpy_morgan_fingerprint(
                #     act,
                #     Hyperparams.fingerprint_length,
                #     Hyperparams.fingerprint_radius
                # )
                    # steps_left,
                # )
                for act in environment.get_valid_actions()
            ]
        )  # (num_actions, fingerprint_length + 1)

        # input(action_fingerprints)

        # Update replay buffer (state: (fingerprint_length + 1), action: _,
        # reward: (), next_state: (num_actions, fingerprint_length + 1),
        # done: ()

        agent.replay_buffer.push(
            action_fingerprint,
            reward,
            action_fingerprints,
            float(result.terminated)
        )

        # agent.replay_buffer.add(
        #     obs_t=action_fingerprint,  # (fingerprint_length + 1)
        #     action=0,  # No use
        #     reward=reward,
        #     obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
        #     done=float(result.terminated),
        # )

        if it % Hyperparams.update_interval == 0 and agent.replay_buffer.__len__() >= Hyperparams.batch_size:
            for update in range(Hyperparams.num_updates_per_it):
                loss = agent.update_params(
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
