#!/bin/bash
module load maxwell gcc/9.3 cuda/11.3

export TORCH=1.12.1
export CUDA=cu113
pip install torch==${TORCH}+${CUDA} --extra-index-url https://download.pytorch.org/whl/cu113

pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install gdown sklearn matplotlib torchinfo

pip install tensorflow
pip install spektral