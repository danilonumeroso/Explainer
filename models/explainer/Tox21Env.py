import utils
import torch

from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule
from config.explainer import Args
from torch.nn import functional as F
from utils import tanimoto_similarity, rescaled_cosine_similarity, cosine_similarity


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

        if similarity_measure == "tanimoto":
            self.similarity = lambda x, y: tanimoto_similarity(x, y)

            self.make_encoding = lambda x: utils.morgan_fingerprint(utils.pyg_to_smiles(x),
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

            self.make_encoding = lambda x: (model_to_explain(x.x, x.edge_index)[1], utils.morgan_fingerprint(utils.pyg_to_smiles(x), self.fp_length, self.fp_radius))
            self.original_encoding = self.make_encoding(original_molecule)


    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)

        if molecule is None or len(molecule.GetBonds()) == 0:
            return 0.0, 0.0, 0.0

        molecule = utils.mol_to_pyg(molecule)

        out, _ = self.model_to_explain(molecule.x, molecule.edge_index)
        out = F.softmax(out, dim=-1).squeeze().detach()

        sim_score = self.similarity(self.make_encoding(molecule), self.original_encoding)
        pred_score = out[self.class_to_optimise].item()
        pred_class = torch.argmax(out).item()

        reward = pred_score * (1 - self.weight_sim) + sim_score * self.weight_sim

        return {
            'reward': reward * self.discount_factor, # ** (self.max_steps - self.num_steps_taken)
            'prediction': pred_score,
            'similarity': sim_score,
            'pred_class': pred_class
        }


class NCF_Tox21(CF_Tox21):

    def __init__(
            self,
            **kwargs
    ):
        super(NCF_Tox21, self).__init__(**kwargs)
        self.class_to_optimise = kwargs['original_molecule'].y.item()
