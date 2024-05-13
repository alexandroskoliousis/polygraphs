# Custom Graphs

Custom graphs from GML files can be loaded by setting `network.kind` parameter as `gml` and specifying a name in `network.gml.name`, path to GML file in `network.gml.path`. The name parameter can be used to identify the network in the results during the analysis stage.

:::tip
Polygraphs loads the graph from the GML file using the edge list
:::

Polygraphs assumes the graph is undirected by default. You can set the parameter `network.gml.directed` to `True` to specify that you have a directed graph.

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

## Custom GML Files
The Polygraphs repository contains an example notebook which demonstrates the [process for generating a GML file](https://github.com/alexandroskoliousis/polygraphs/blob/main/scripts/sixdegreesoffrancisbacon.ipynb) from the [Six Degrees of Francis Bacon](http://sixdegreesoffrancisbacon.com) network.
