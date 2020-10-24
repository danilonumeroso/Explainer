#!/bin/sh
CUDA_VERSION=cpu # cpu | cu92 | cu101 | cu102
TORCH_VERSION=1.6.0
TORCH_GEOMETRIC_VERSION=1.6.0

conda remove --name meg --all
conda create --name meg python=3.7 -y
conda activate meg

conda install pip -y
conda install rdkit -c rdkit -y
conda install tensorboard -y

python -m pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# PyTorch Geometric dependencies
python -m pip install torch-scatter==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
python -m pip install torch-sparse==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
python -m pip install torch-cluster==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
python -m pip install torch-spline-conv==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html
python -m pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}
