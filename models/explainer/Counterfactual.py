# import torch
# import torch.nn.functional as F
import utils

from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule


class Counterfactual(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(
        self,
        discount_factor,
        base_molecule,
        counterfactual_class,
        weight_sim=0.5,
        **kwargs
    ):
        """
        Initializes the class.

        Args:

        * discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
        **kwargs: The keyword arguments passed to the base class.

        """
        super(Counterfactual, self).__init__(**kwargs)

        self.counterfactual_class = counterfactual_class

        self.mol_fp = utils.morgan_fingerprint(
            utils.pyg_to_smiles(base_molecule)
        )

        self.encoder = utils.get_encoder("Tox21", "Encoder")
        self.base_molecule = base_molecule
        pred, self.base_encoding = self._encode(base_molecule)

        assert pred.max(dim=1)[1] != counterfactual_class

        self.discount_factor = discount_factor

        self._similarity = lambda mol1, fp2: \
            DataStructs.TanimotoSimilarity(
                utils.morgan_fingerprint(mol1),
                fp2
            )

        self.weight_sim = weight_sim

        self.i = 0

    def _encode(self, molecule):
        output, encoding = self.encoder(molecule.x,
                                        molecule.edge_index,
                                        molecule.batch)

        return output, encoding.squeeze()

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
