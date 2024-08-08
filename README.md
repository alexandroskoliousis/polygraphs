# PolyGraphs
<p align="center">
  <img src="https://akoliousis.com/polygraphs/eu_email_core.webp" alt="Visualisation of EU Email Core network" height="250">
</p>

<p align="center"><a href="https://polygraphs.sites.northeastern.edu/">Website</a> | <a href="https://akoliousis.com/polygraphs/">Documentation</a> | <a href="https://pypi.org/project/polygraphs/">PyPi</a></p>

PolyGraphs is a scaleable framework for performing simulations on networks built using PyTorch and DGL that can run on CPUs and GPUs.

## Getting Started
PolyGraphs requires and appropriately configured version of PyTorch and DGL before installation, see the [getting started guide](https://akoliousis.com/polygraphs/guide/introduction/getting-started) for more details. You can install the PolyGraphs library via PyPi:

```bash
pip install polygraphs
```

You can run simulations using a configuration file with the `polygraphs` command:

``` bash
polygraphs -f test.yaml
```

### Installing from Source
Advanced users can [install from source](https://akoliousis.com/polygraphs/guide/introduction/install-from-source), see the documentation for more details on running PolyGraphs on the [platform guide](https://akoliousis.com/polygraphs/guide/introduction/platform-guide).

## Analysing Simulation Results
Results from simulations can be processed using the [analysis module](https://akoliousis.com/polygraphs/guide/simulations/processing-results). 

## Papers About PolyGraphs
Ball, B., Koliousis, A., Mohanan, A. et al. [Computational philosophy: reflections on the PolyGraphs project](https://doi.org/10.1057/s41599-024-02619-z). Humanit Soc Sci Commun 11, 186 (2024).

## Contributing
Please file an [issue](https://github.com/alexandroskoliousis/polygraphs/issues) if you encounter a bug or have any suggestions. Bug-fixes, contributions, new features and extensions are welcomed through discussion in issues.
