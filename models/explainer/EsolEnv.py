import torch
import torch.nn.functional as F
import numpy as np

from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule
from utils import get_similarity, mol_to_smiles, mol_from_smiles, pyg_to_mol_esol, mol_to_esol_pyg

class CF_Esol(Molecule):

    def __init__(
            self,
            model_to_explain,
            original_molecule,
            discount_factor,
            fp_len,
            fp_rad,
            similarity_set=None,
            weight_sim=0.5,
            similarity_measure="tanimoto",
            **kwargs
    ):
        super(CF_Esol, self).__init__(**kwargs)

        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.weight_sim = weight_sim
        self.target = original_molecule.y
        self.orig_pred, _ = model_to_explain(original_molecule.x, original_molecule.edge_index)
        self.distance  = lambda x,y: F.l1_loss(x,y).detach()
        self.base_loss = self.distance(self.orig_pred, self.target).item()
        self.gain = lambda p: torch.sign(self.distance(p, self.orig_pred)).item()

        self.similarity, self.make_encoding, \
            self.original_encoding = get_similarity(similarity_measure,
                                                    lambda x: mol_to_smiles(pyg_to_mol_esol(x)),
                                                    model_to_explain,
                                                    original_molecule,
                                                    fp_len,
                                                    fp_rad)

    def _reward(self):

        molecule = mol_from_smiles(self._state)
        molecule = mol_to_esol_pyg(molecule)

        pred, encoding = self.model_to_explain(molecule.x,
                                        molecule.edge_index)

        sim = self.similarity(self.make_encoding(molecule), self.original_encoding)


        loss = self.distance(pred, self.orig_pred).item()

        gain = self.gain(pred)

        reward = gain * loss * (1 - self.weight_sim) + sim * self.weight_sim

        return {
            'pyg': molecule,
            'reward': reward * self.discount_factor ** (self.max_steps - self.num_steps_taken),
            'reward_pred': loss,
            'reward_gain': gain,
            'reward_sim': sim,
            'encoding': encoding.numpy(),
            'prediction': {
                'type': 'regression',
                'output': pred.squeeze().detach().numpy().tolist(),
                'for_explanation': pred.squeeze().detach().numpy().tolist()
            }
        }


class NCF_Esol(CF_Esol):

    def __init__(
            self,
            **kwargs
    ):
        super(NCF_Esol, self).__init__(**kwargs)
        self.distance  = lambda x,y: -F.l1_loss(x,y).detach()
        self.gain = lambda p: 1
