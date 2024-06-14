# Install From Source
PolyGraphs requires a Python environment with a number of other packages it depends on to run and analyse simulations. The steps on this page only need to be followed once on your computer, where we will use the terminal to prepare and install PolyGraphs as a package.

## Prerequisites
- [Git](https://git-scm.com/downloads) for cloning the repository
- [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for managing packages and the enviroment for Python
- Terminal for running commands
- Text Editor for editing configuration files

## Clone the Repository
Open the terminal and make a copy of the source code on your local computer by cloning our repository using git. Once the repository has been cloned, change into the newly created `polygraphs` directory to continue the rest of the installation:

```bash
git clone https://github.com/alexandroskoliousis/polygraphs.git
cd polygraphs
```

:::tip
Running the above command will clone the PolyGraphs repository in your home directory, this is the file location that most terminals open when they are started. PolyGraphs will also create a `polygraphs-cache` folder in your home directory to store simulations.
:::

## Create a Conda Environment
PolyGraphs uses conda to manage packages and the environment for Python. A conda environment is a self-contained collection of packages and their dependencies used on a project.

:::tip
There are two distributions of conda available for download from the website, Anaconda and Miniconda. Anaconda contains a large collection of data science packages and a GUI for managing environments. Miniconda has a smaller installation footprint which only contains Python and a few other packages. As this instructions in this guide manages conda from the command line, they are compatible with both distributions.
:::

We can use the `environment.yml` file in the repository to tell conda to create a new environment called `polygraphs` containing packages necessary for PolyGraphs to run:

```bash
conda env create -n polygraphs --file environment.yml
```

## Finish Setup
Once the dependencies have been downloaded, we can finish the installation using packages in the `polygraphs` environment by activating it:

```bash
conda activate polygraphs
```

Next, install PolyGraphs itself as a package in the environment so that we can import it from other Python scripts and notebooks:

```bash
pip install -e .
```

Finally, let's tell [DGL](https://www.dgl.ai/), the graph library used by PolyGraphs to use [PyTorch](https://pytorch.org/) as its backend:

```bash
python -m dgl.backend.set_default_backend . pytorch
```

## Running Simulations
Open a terminal in the `polygraphs` directory you cloned and ensure that the `polygraphs` environment is active by running:

```bash
conda activate polygraphs
```

:::tip
You will need to activate the `polygraphs` conda environment every time you start a new terminal session to use PolyGraphs. 
:::

Once the environment has been activated we can run a simulation using a test configuration available in the repository by running:

```bash
python run.py -f configs/test.yaml
```
