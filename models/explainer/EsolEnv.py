import torch
import torch.nn.functional as F
import utils
import numpy as np

from config.explainer import Args
from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule
from utils import tanimoto_similarity, rescaled_cosine_similarity, cosine_similarity

class CF_Esol(Molecule):

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
        super(CF_Esol, self).__init__(**kwargs)

        Hyperparams = Args()

        self.fp_length = Hyperparams.fingerprint_length
        self.fp_radius = Hyperparams.fingerprint_radius
        self.discount_factor = discount_factor
        self.model_to_explain = model_to_explain
        self.weight_sim = weight_sim
        self.target = original_molecule.y
        self.orig_pred, _ = model_to_explain(original_molecule.x, original_molecule.edge_index)
        self.distance  = lambda x,y: F.l1_loss(x,y).detach()
        self.base_loss = self.distance(self.orig_pred, self.target).item()
        self.gain = lambda p: torch.sign(
            self.distance(p, self.orig_pred)
        ).item()

        if similarity_measure == "tanimoto":
            self.similarity = lambda x, y: tanimoto_similarity(x, y)

            self.make_encoding = lambda x: utils.morgan_fingerprint(utils.esol_pyg_to_smiles(x),
                                                                    self.fp_length,
                                                                    self.fp_radius)
            self.original_encoding = self.make_encoding(original_molecule)

        elif similarity_measure == "rescaled_neural_encoding":
            self.similarity = lambda x, y: rescaled_cosine_similarity(x,
                                                                      y,
                                                                      similarity_set)

            self.make_encoding = lambda x: model_to_explain(x.x, x.edge_index)[1]
            self.original_encoding = self.make_encoding(original_molecule)

        elif similarity_measure == "neural_encoding":
            self.similarity = lambda x, y: cosine_similarity(x, y)

            self.make_encoding = lambda x: model_to_explain(x.x, x.edge_index)[1]
            self.original_encoding = self.make_encoding(original_molecule)

        elif similarity_measure == "combined":
            self.similarity = lambda x, y: 0.5 * cosine_similarity(x[0], y[0]) + 0.5 * tanimoto_similarity(x[1], y[1])

            self.make_encoding = lambda x: (model_to_explain(x.x, x.edge_index)[1], utils.morgan_fingerprint(utils.esol_pyg_to_smiles(x), self.fp_length, self.fp_radius))
            self.original_encoding = self.make_encoding(original_molecule)

    def _reward(self):

        molecule = Chem.MolFromSmiles(self._state)

        if molecule is None or len(molecule.GetBonds()) == 0:
            return 0.0, 0.0, 0.0

        molecule = utils.mol_to_esol_pyg(molecule)

        pred, _ = self.model_to_explain(molecule.x,
                                        molecule.edge_index)

        sim = self.similarity(self.make_encoding(molecule), self.original_encoding)


        loss = self.distance(pred, self.orig_pred).item()

        gain = self.gain(pred)

        reward = gain * loss * (1 - self.weight_sim) + sim * self.weight_sim

        return {
            'reward': reward * self.discount_factor ** (self.max_steps - self.num_steps_taken),
            'prediction': loss,
            'gain': gain,
            'similarity': sim,
            'pred_class': pred.squeeze().item()
        }


class NCF_Esol(CF_Esol):

    def __init__(
            self,
            **kwargs
    ):
        super(NCF_Esol, self).__init__(**kwargs)
        self.distance  = lambda x,y: -F.l1_loss(x,y).detach()
        self.gain = lambda p: 1
