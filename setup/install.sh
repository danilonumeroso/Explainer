#!/bin/sh
CUDA_VERSION=${1:-cpu} # cpu | cu92 | cu101 | cu102
TORCH_VERSION=1.6.0
TORCH_GEOMETRIC_VERSION=1.6.0

conda remove --name meg --all -y
conda create --name meg python=3.7 -y
conda activate meg

conda install pip -y
conda install rdkit -c rdkit -y
conda install tensorboard -y
conda install -c conda-forge typer -y

if [[ ${CUDA_VERSION} == 'cpu' ]]; then
  conda install pytorch==${TORCH_VERSION} torchvision torchaudio cpuonly -c pytorch -y
elif [[ ${CUDA_VERSION} == 'cu92' ]]; then
  conda install pytorch==${TORCH_VERSION} torchvision torchaudio cudatoolkit=9.2 -c pytorch -y
elif [[ ${CUDA_VERSION} == 'cu101' ]]; then
  conda install pytorch==${TORCH_VERSION} torchvision torchaudio cudatoolkit=10.1 -c pytorch -y
elif [[ ${CUDA_VERSION} == 'cu102' ]]; then
  conda install pytorch==${TORCH_VERSION} torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
elif [[ ${CUDA_VERSION} == 'cu110' ]]; then
  conda install pytorch==${TORCH_VERSION} torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
fi

python -m pip install geomloss

# PyTorch Geometric dependencies
python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
python -m pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}
