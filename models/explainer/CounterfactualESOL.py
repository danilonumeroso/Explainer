import torch
import torch.nn.functional as F
import utils
import numpy as np
from config.explainer import Args

from rdkit import Chem, DataStructs
from models.explainer.Environment import Molecule


class CounterfactualESOL(Molecule):

    def __init__(
            self,
            discount_factor,
            base_molecule,
            target,
            weights=[1, 1, 1],
            **kwargs
    ):
        super(CounterfactualESOL, self).__init__(**kwargs)

        Hyperparams = Args()
        self.fp_length = Hyperparams.fingerprint_length
        self.fp_radius = Hyperparams.fingerprint_radius

        self.mol_fp = utils.morgan_fingerprint(
            base_molecule.smiles,
            self.fp_length,
            self.fp_radius
        )

        self.encoder = utils.get_encoder("ESOL", Hyperparams.experiment)
        self.base_molecule = base_molecule
        self.target = target

        self.orig_pred, self.base_encoding = self._encode(base_molecule)

        self.distance  = lambda x,y: F.l1_loss(x,y).detach()
        self.base_loss = self.distance(self.orig_pred, self.target).item()

        self.discount_factor = discount_factor

        self._similarity = lambda mol1, fp2: \
            DataStructs.TanimotoSimilarity(
                utils.morgan_fingerprint(mol1,
                                         self.fp_length,
                                         self.fp_radius),
                fp2
            )

        self.w = np.array(weights)

        self.i = 0

    def _encode(self, molecule):
        output, encoding = self.encoder(molecule.x.float(),
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

        molecule = utils.mol_to_esol_pyg(molecule)
        pred, encoding = self._encode(molecule)

        sim = self._similarity(self._state, self.mol_fp)


        loss = self.distance(pred, self.orig_pred).item()

        gain = torch.sign(
            self.distance(pred, self.target) - self.base_loss
        ).item()

        reward = gain * loss * self.w[0] + sim * self.w[2]

        return reward * self.discount_factor \
            ** (self.max_steps - self.num_steps_taken), loss, gain, sim
