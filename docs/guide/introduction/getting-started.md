# Getting Started
PolyGraphs requires PyTorch and DGL libraries to be configured on your computer before it can be installed. 

## Libraries
Currently, the latest version of PyTorch supported by DGL is `2.3.0`, the following instructions install a CPU version of PyTorch and DGL. See the installation instruction on the [PyTorch](https://pytorch.org/get-started/previous-versions/) and [DGL](https://www.dgl.ai/pages/start.html) getting started guides if you require support for CUDA or platform specific instructions. DGL on Linux has a slightly different installation method to the MacOS instructions shown below.

### pip
```bash
pip install torch==2.3.0 torchdata
pip install pydantic
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### conda
```bash
conda create --name polygraphs
conda activate polygraphs
conda install pytorch==2.3.0 torchdata -c pytorch
conda install pydantic -c conda-forge
conda install -c dglteam dgl
```

After the installation has completed, DGL should be configured to use PyTorch as its [backend](https://docs.dgl.ai/en/latest/install/#working-with-different-backends):

```bash
python -m dgl.backend.set_default_backend pytorch
```

## Install PolyGraphs
You can install and use PolyGraphs library via PyPi:

```bash
pip install polygraphs
```

## Next Steps
- Run a [test simulation](/guide/simulations/running-simulations) to check that PolyGraphs is working
