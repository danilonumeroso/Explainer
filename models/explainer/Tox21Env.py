import utils
import torch

from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule
from config.explainer import Args
from torch.nn import functional as F
from utils import get_similarity, mol_to_smiles, mol_from_smiles, mol_to_tox21_pyg


class CF_Tox21(Molecule):
    def __init__(
            self,
            model_to_explain,
            original_molecule,
            discount_factor,
            similarity_set=None,
            weight_sim=0.5,
            similarity_measure="tanimoto",
            **kwargs
    ):
        super(CF_Tox21, self).__init__(**kwargs)

        Hyperparams = Args()

        self.fp_length = Hyperparams.fingerprint_length
        self.fp_radius = Hyperparams.fingerprint_radius
        self.class_to_optimise = 1 - original_molecule.y.item()
        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.weight_sim = weight_sim


        self.similarity, self.make_encoding, \
            self.original_encoding = get_similarity(similarity_measure,
                                                    mol_to_smiles,
                                                    model_to_explain,
                                                    original_molecule,
                                                    self.fp_length,
                                                    self.fp_radius)

    def _reward(self):
        molecule = mol_from_smiles(self._state)
        molecule = mol_to_tox21_pyg(molecule)

        out, encoding = self.model_to_explain(molecule.x, molecule.edge_index)
        out = F.softmax(out, dim=-1).squeeze().detach()

        sim_score = self.similarity(self.make_encoding(molecule), self.original_encoding)
        pred_score = out[self.class_to_optimise].item()
        pred_class = torch.argmax(out).item()

        reward = pred_score * (1 - self.weight_sim) + sim_score * self.weight_sim

        return {
            'pyg': molecule,
            'reward': reward * self.discount_factor,
            'reward_pred': pred_score,
            'reward_sim': sim_score,
            'encoding': encoding.numpy(),
            'prediction': {
                'type': 'bin_classification',
                'output': out.numpy().tolist(),
                'for_explanation': pred_class,
                'class': pred_class
            }
        }


class NCF_Tox21(CF_Tox21):

    def __init__(
            self,
            **kwargs
    ):
        super(NCF_Tox21, self).__init__(**kwargs)
        self.class_to_optimise = kwargs['original_molecule'].y.item()
