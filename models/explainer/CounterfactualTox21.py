import utils

from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule
from config.explainer import Args
from torch.nn import functional as F

class CounterfactualTox21(Molecule):
    def __init__(
        self,
        encoder,
        discount_factor,
        mol_fp,
        base_molecule,
        counterfactual_class,
        weight_sim=0.5,
        **kwargs
    ):
        super(CounterfactualTox21, self).__init__(**kwargs)

        Hyperparams = Args()

        self.fp_length = Hyperparams.fingerprint_length
        self.fp_radius = Hyperparams.fingerprint_radius

        self.counterfactual_class = counterfactual_class

        self.mol_fp = mol_fp
        self.encoder = encoder
        self.base_molecule = base_molecule
        pred, self.base_encoding = self._encode(base_molecule)

        assert pred.max(dim=1)[1] != counterfactual_class

        self.discount_factor = discount_factor

        self._similarity = lambda mol1, fp2: \
            DataStructs.TanimotoSimilarity(
                utils.morgan_fingerprint(mol1,
                                         self.fp_length,
                                         self.fp_radius),
                fp2
            )

        self.weight_sim = weight_sim

        self.i = 0

    def _encode(self, molecule):
        output, encoding = self.encoder(molecule.x,
                                        molecule.edge_index,
                                        molecule.batch)

        return F.log_softmax(output, dim=-1), encoding.squeeze()

    def _reward(self):
        """
        Reward of a state.

        Returns:
        Float. QED of the current state.
        """

        molecule = Chem.MolFromSmiles(self._state)

        if molecule is None or len(molecule.GetBonds()) == 0:
            return 0.0, 0.0, 0.0

        molecule = utils.mol_to_pyg(molecule)

        out, encoding = self._encode(molecule)

        sim = self._similarity(self._state, self.mol_fp)

        pred = 1 + max(out[0][self.counterfactual_class].item(), -1)

        reward = pred * (1 - self.weight_sim) + sim * self.weight_sim

        return reward * self.discount_factor \
            ** (self.max_steps - self.num_steps_taken), pred, sim
