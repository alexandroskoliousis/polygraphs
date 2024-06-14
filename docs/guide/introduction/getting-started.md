# Getting Started
PolyGraphs requires PyTorch and DGL libraries to be configured on your computer before it can be installed. 

## Libraries
Currently, the latest version of PyTorch supported by DGL is `2.3.0`, the following instructions install a CPU version of PyTorch and DGL. See the installation instruction on the [PyTorch](https://pytorch.org/get-started/previous-versions/) and [DGL](https://www.dgl.ai/pages/start.html) getting started guides if you require support for CUDA or platform specific instructions.

```bash
pip install torch==2.3.0 torchdata
pip install pydantic
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

<details>
  <summary>Conda</summary>
  
```bash
conda install pytorch==2.3.0 torchdata -c pytorch
conda install pydantic -c conda-forge
conda install -c dglteam dgl
```

</details>


DGL should be configured to use PyTorch as its [backend](https://docs.dgl.ai/en/latest/install/#working-with-different-backends) after the installation has completed:

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
