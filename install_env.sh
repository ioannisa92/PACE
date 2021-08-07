
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
conda install -c rdkit rdkit

apt-get update && apt-get install -yq --no-install-recommends libxrender1
apt-get update && apt-get install -yq --no-install-recommends libxext6
apt-get update && apt-get install -yq --no-install-recommends rsync


## installation for pytorch_geometric
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:$CPATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CUDA=cu101

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

pip install torch-geometric