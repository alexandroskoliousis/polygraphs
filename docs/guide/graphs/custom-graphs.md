# Custom Graphs
Graphs defined using [Graph Modelling Language](https://networkx.org/documentation/stable/reference/readwrite/gml.html) can be imported for use with Polygraphs. Graphs from GML files can be loaded by setting `network.kind` parameter as `gml` and specifying a name in `network.gml.name` and path to GML file in `network.gml.path`. The name parameter is used to identify the network using the configuration file stored with the results during the analysis stage.

PolyGraphs assumes the file contains an undirected graph by default, you can set the parameter `network.gml.directed` to `True` to specify that you have a directed graph.

:::tip
Polygraphs loads the graph from the GML file using the edge list
:::

Remember that the graph used to run the simulation is stored inside the `.bin` file inside the results directory. Ensure that any analysis you do of the results are made using the graph from this file (which can be accessed using the PolyGraphs analyser) to avoid any inconsistencies with the graph.

## Example Configuration
The following configuration file loads a GML file called `custom_graph.gml` which contains a directed graph from the user's home directory:

```yaml
# Kind and size of network (a complete network with 10 agents)
network.kind: "gml"
network.gml.name: "CustomGMLNetwork"
network.gml.path: "~/custom_graph.gml"
network.gml.directed: True

# Initial beliefs are random uniform between 0 and 1
init.kind: "uniform"
# Chance that action B is better than action A
epsilon: 0.01
# Enable logging; print progress every 100 steps
logging.enabled: True
logging.interval: 100
# Enable snapshots; take one every 100 steps
snapshots.enabled: True
snapshots.interval: 100
# Run for 1,000 steps
simulation.steps: 1000
# Set model
op: "BalaGoyalOp"
```

## Creating GML Files
The Polygraphs repository contains an example notebook which demonstrates the [process for generating a GML file](https://github.com/alexandroskoliousis/polygraphs/blob/main/scripts/sixdegreesoffrancisbacon.ipynb) from the [Six Degrees of Francis Bacon](http://sixdegreesoffrancisbacon.com) network.

## Node ID Normalisation
Node ids in GML files are normalised before they are loaded in Polygraphs. Normalised ids for all nodes can be accessed from the `gml_id` of Polygraphs processor graphs object:
```python
processor.graphs[0].pg['ndata']['gml_id']
```
You can also simulate the normalisation process using the `Processor.normalise_gml()` method on any GML file to get a dictionary of resulting node ids:
```python
from polygraphs.analysis import Processor
Processor.normalise_gml("~/path_to/graph.gml.gz")
```