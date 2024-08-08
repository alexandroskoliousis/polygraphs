# Platform Guide

## Quick Start: Desktop
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ python setup.py install
$ python -m dgl.backend.set_default_backend . pytorch
```

For a conda environment:
```bash
$ conda env create -n polygraphs --file environment.yml
$ conda activate polygraphs
$ python setup.py install
$ python -m dgl.backend.set_default_backend . pytorch
```

## Quick Start: Google Colaboratory
```bash
!git clone https://github.com/alexandroskoliousis/polygraphs.git
%cd polygraphs
!pip install -e .
!nvidia-smi
!pip install dgl-cu110
```

## Quick Start: Discovery Cluster
```bash
$ ssh username@login.discovery.neu.edu
[username@login-00 ~] $
```

For succinctness, I omit the prefix `[username@login-00 ~]` in the following commands:
```bash
$ module load python
$ python -V
Python 3.8.1
$ git clone git@github.com:<account name>/polygraphs.git
$ cd polygraphs
$ echo "export PYTHONPATH=$PWD:$PYTHONPATH" >> ~/.bashrc
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install --upgrade pip
(.venv) $ pip install -r requirements-discovery.txt
(.venv) $ python -m dgl.backend.set_default_backend pytorch
(.venv) $ python run.py --help
(.venv) $ srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --export=ALL --pty /bin/bash
```

Once on the allocated machine (say `vm`), run:
```bash
(.venv) [vm] $ python run.py -f configs/test.yaml
```

### Generating and running job array configurations
```bash
(.venv) $ python scripts/job-array-generator.py -f configs/zollman-effect/zollman-effect.yaml -e configs/explorables.json -a test
(.venv) $ sbatch run-array.script
```

### Conda Environment
A conda environment is required to run the Processor module from PolyGraphs inside JupyterLab from OOD. The Discovery Cluster requires a Python 3.8 environment, create a new environment with this version and install the dependencies using `requirements-discovery.txt`:

```bash
$ module load anaconda3
$ conda create -n polygraphs python=3.8
$ conda activate polygraphs
$ pip install -r requirements-discovery.txt
```
