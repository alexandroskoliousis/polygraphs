# PolyGraphs

## Summary

## Quick Start: Desktop

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ python setup.py install
$ python -m dgl.backend.set_default_backend pytorch
```

## Quick Start: Google Colaboratory

```bash
!git clone https://[token]@github.com/alexandroskoliousis/polygraphs.git
%cd polygraphs
!python setup.py install
!nvidia-smi
!pip install dgl-cu110
```

## Quick Start: Discovery Cluster

```bash
$ ssh username@login.discovery.neu.edu
[username@login-00 ~] $
```
For succinctness, I ommit the prefix `[username@login-00 ~]` in the following commands:
```
$ module load python
$ python -V
Python 3.8.1
$ git clone git@github.com:<account name>/polygraphs.git
$ cd polygraphs
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install --upgrade pip
(.venv) $ pip install -r requirements.txt
(.venv) $ python -m dgl.backend.set_default_backend pytorch
(.venv) $ python run.sh --help
(.venv) $ srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --export=ALL --pty /bin/bash
```
Once on the allocated machine (say `vm`), run:
```
(.venv) [vm] $ python run.py -f configs/test.yaml
```

## Generating and running job array configurations
```bash
(.venv) $ python scripts/job-array-generator.py -f configs/zollman-effect/zollman-effect.yaml -e configs/explorables.json
(.venv) $ sbatch run-array.script
```
