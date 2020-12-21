#!/bin/sh
CUDA_VERSION=${1:-cpu} # cpu | cu92 | cu101 | cu102
TORCH_VERSION=1.6.0
TORCH_GEOMETRIC_VERSION=1.6.0

conda remove --name meg --all
conda create --name meg python=3.7 -y
conda activate meg

conda install pip -y
conda install rdkit -c rdkit -y
conda install tensorboard -y
conda install -c conda-forge typer -y

python -m pip install geomloss

python -m pip install torch==${TORCH_VERSION}+${CUDA_VERSION} torchvision==0.7.0+${CUDA_VERSION} -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch Geometric dependencies
python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}
