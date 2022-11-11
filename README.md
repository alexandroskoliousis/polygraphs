# PolyGraphs

## Summary

## Quick Start: Desktop

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
$ python -m dgl.backend.set_default_backend . pytorch
```
On macos, you may need to install some packages via homebrew, such as `hd5`, `cython` and `yaml`.

For a conda environment:
```bash
$ conda env create -n polygraphs --file environment.yml
$ conda activate polygraphs
$ python setup.py install
$ python -m dgl.backend.set_default_backend . pytorch
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
(.venv) $ pip install -r requirements.txt
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

## Gathering Simulation Results

To gather results after simulations have run into a CSV file for analysis:
```bash
(.venv) $ python scripts/gather.py
```
This will load simulation results from the default location of `~/polygraphs-cache/results` and exports the CSV file in the same folder.

### Optional Arguments

- `-f`: Specify location of results folder
- `-n`: Networks to filter (seperated by spaces)
- `--add-polarisation`: Extract polarisation hyper-parameters
- `--add-reliability`: Extract reliability hyper-parameters
- `--add-statistics`: Extract network statistics (clustering, density)

Examples:
```bash
# Add polarisation hyper-parameter column
(.venv) $ python scripts/gather.py --add-polarisation

# Extract only complete networks and add reliability parameter
(.venv) $ python scripts/gather.py -n complete --add-reliability

# Extract results from a specific folder
(.venv) $ python scripts/gather.py -f ~/polygraphs-cache/results/2022-10-01
```
