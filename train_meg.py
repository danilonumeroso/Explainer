import torch
import utils
import numpy as np
import torchvision
import json
import os
import utils
import networkx as nx

from models.explainer import CF_Tox21, NCF_Tox21, Agent, CF_Esol, NCF_Esol
from config.explainer import Args
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem
from utils import SortedQueue, morgan_bit_fingerprint, get_split, get_dgn, mol_to_smiles, x_map_tox21
from torch.nn import functional as F
from torch_geometric.utils import to_networkx

def tox21(general_params):
    Hyperparams = Args()
    BasePath = './runs/tox21/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')

    dataset = get_split('tox21', 'test', Hyperparams.experiment)

    original_molecule = dataset[Hyperparams.sample]
    model_to_explain = get_dgn("tox21", Hyperparams.experiment)

    out, original_encoding = model_to_explain(original_molecule.x,
                                              original_molecule.edge_index)

    logits = F.softmax(out, dim=-1).detach().squeeze()
    pred_class = logits.argmax().item()

    assert pred_class == original_molecule.y.item()

    smiles = mol_to_smiles(original_molecule)

    print(f'Molecule: {smiles}')

    atoms_ = [
        x_map_tox21(e).name
        for e in np.unique(
            [x.tolist().index(1) for x in original_molecule.x.numpy()]
        )
    ]

    params = {
        # General-purpose params
        **general_params,
        'init_mol': smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'original_molecule': original_molecule,
        'model_to_explain': model_to_explain,
        'weight_sim': 0.2,
        'similarity_measure': 'combined'
    }

    N = 20
    cf_queue = SortedQueue(N, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Tox21(**params)
    cf_env.initialize()

    ncf_queue = SortedQueue(N, sort_predicate=lambda mol: mol['reward'])
    ncf_env = NCF_Tox21(**params)
    ncf_env.initialize()

    meg_train(writer, cf_env, cf_queue, marker="cf", tb_name="tox21")
    meg_train(writer, ncf_env, ncf_queue, marker="ncf", tb_name="tox_21")

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': smiles,
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'bin_classification',
            'output': logits.numpy().tolist(),
            'for_explanation': original_molecule.y.item(),
            'class': original_molecule.y.item()
        }
    })
    overall_queue.extend(cf_queue.data_)
    overall_queue.extend(ncf_queue.data_)

    save_results(BasePath, overall_queue)

def esol(general_params):
    Hyperparams = Args()
    BasePath = './runs/esol/' + Hyperparams.experiment
    writer = SummaryWriter(BasePath + '/plots')

    dataset = get_split('esol', 'test', Hyperparams.experiment)
    original_molecule = dataset[Hyperparams.sample]
    original_molecule.x = original_molecule.x.float()
    model_to_explain = get_dgn("esol", Hyperparams.experiment)

    og_prediction, original_encoding = model_to_explain(original_molecule.x, original_molecule.edge_index)
    print(f'Molecule: {original_molecule.smiles}')

    atoms_ = np.unique(
        [x.GetSymbol() for x in Chem.MolFromSmiles(original_molecule.smiles).GetAtoms()]
    )

    params = {
        # General-purpose params
        **general_params,
        'init_mol': original_molecule.smiles,
        'atom_types': set(atoms_),
        # Task-specific params
        'model_to_explain': model_to_explain,
        'original_molecule': original_molecule,
        'weight_sim': 0.2,
        'similarity_measure': 'combined'
    }

    N = 20
    cf_queue = SortedQueue(N, sort_predicate=lambda mol: mol['reward'])
    cf_env = CF_Esol(**params)
    cf_env.initialize()

    ncf_queue = SortedQueue(N, sort_predicate=lambda mol: mol['reward'])
    ncf_env = NCF_Esol(**params)
    ncf_env.initialize()

    meg_train(writer, cf_env, cf_queue, marker="cf", tb_name="esol")
    meg_train(writer, ncf_env, ncf_queue, marker="ncf", tb_name="esol")

    overall_queue = []
    overall_queue.append({
        'pyg': original_molecule,
        'marker': 'og',
        'smiles': original_molecule.smiles,
        'encoding': original_encoding.numpy(),
        'prediction': {
            'type': 'regression',
            'output': og_prediction.squeeze().detach().numpy().tolist(),
            'for_explanation': og_prediction.squeeze().detach().numpy().tolist()
        }
    })
    overall_queue.extend(cf_queue.data_)
    overall_queue.extend(ncf_queue.data_)

    save_results(BasePath, overall_queue)

def meg_train(writer, environment, queue, marker, tb_name):
    Hyperparams = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(Hyperparams.fingerprint_length + 1, 1, device)

    eps = 1.0
    batch_losses = []
    episode = 0

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

        _, out, done = result

        writer.add_scalar(f'{tb_name}/reward', out['reward'], it)
        writer.add_scalar(f'{tb_name}/prediction', out['reward_pred'], it)
        writer.add_scalar(f'{tb_name}/similarity', out['reward_sim'], it)

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
            out['reward'],
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
            episode += 1

            print(f'Episode {episode}::Final Molecule Reward: {out["reward"]:.6f} (pred: {out["reward_pred"]:.6f}, sim: {out["reward_sim"]:.6f})')
            print(f'Episode {episode}::Final Molecule: {action}')
            queue.insert({
                'marker': marker,
                'smiles': action,
                **out
            })

            eps *= 0.9995
            batch_losses = []
            environment.initialize()


def save_results(base_path, queue, quantitative=True):
    output_dir = base_path + f"/meg_output/{Hyperparams.sample}"
    embedding_dir = output_dir + "/embeddings"
    gexf_dir = output_dir + "/gexf_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(embedding_dir)
        os.makedirs(gexf_dir)

    for i, molecule in enumerate(queue):
        np.save(embedding_dir + f"/{i}", molecule.pop('encoding'))
        pyg = molecule.pop('pyg')
        if quantitative:
            g = to_networkx(pyg, to_undirected=True)
            nx.write_gexf(g, f"{gexf_dir}/{i}.{molecule['prediction']['class']}.gexf")

    with open(output_dir + "/data.json", "w") as outf:
        json.dump(queue, outf, indent=2)

if __name__ == '__main__':
    Hyperparams = Args()
    params = {
        # General-purpose params
        'discount_factor': Hyperparams.discount,
        'allow_removal': True,
        'allow_no_modification': False,
        'allow_bonds_between_rings': True,
        'allowed_ring_sizes': set(Hyperparams.allowed_ring_sizes),
        'max_steps': Hyperparams.max_steps_per_episode,
    }


    tox21(params) if Hyperparams.dataset.lower() == 'tox21' else esol(params)
