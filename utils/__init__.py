from .data import *
from .molecules import *
from .similarity import *
from .queue import *
from .fingerprints import *
from .train import *
from torch_geometric.datasets.molecule_net import x_map as x_map_esol, e_map as e_map_esol

def create_path(output_path):
    if not output_path.exists():
        output_path.mkdir(parents=True)
