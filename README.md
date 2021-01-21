# MEG: Molecular Explanation Generator
This repository contains the official (PyTorch) implementation of MEG, from the paper [Explaining Deep Graph Networks with Molecular Counterfactuals](https://arxiv.org/abs/2011.05134).

# Usage
We assume miniconda (or anaconda) to be installed.

First, install the dependencies:
```
source setup/install.sh [cpu | cu92 | cu101 | cu102]
```
By default, the script will install the 1.6.0 cpu version of PyTorch and PyTorch Geometric. If you want to install the cuda version, just pass the argument. Instead, if you wish to install a different version (e.g, 1.7+) you need to modify the first line of the script:

```
#!/bin/sh
CUDA_VERSION=${1:-cpu}
TORCH_VERSION=1.6.0 # modify this
TORCH_GEOMETRIC_VERSION=1.6.0 # and this
```

The setup script will create a conda environment named "meg".

Now you can train the DGN to be explained by running:
```
python train_dgn.py [tox21 | esol] <experiment_name>
```

Finally, to start training MEG:
```
python train_meg.py [tox21 | esol] <experiment_name>
```
MEG will automatically retrieve the checkpoint of the model and save its
results to runs/<dataset_name>/<experiment_name>/meg_output.

# Cite
Please, cite our paper if you happen to use our code in your own work.

```
@misc{numeroso2020explaining,
      title={Explaining Deep Graph Networks with Molecular Counterfactuals},
      author={Danilo Numeroso and Davide Bacciu},
      year={2020},
      eprint={2011.05134},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
