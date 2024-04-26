# Processing Results
Results can be read using the analysis module from PolyGraphs. The analysis module contains a `Processor` class that is initialised with the location of the results. Simulations are stored by default in the `~/polygraphs-cache` directory.

```python
from polygraphs import analysis

processor = analysis.Processor("~/polygraphs-cache/results")
```

The `get()` method can be used to return a data frame containing the list of results.

```python
processor.get()
```

## Getting Graphs and Beliefs
```python
processor.get_graphs()
```

```python
processor.get_beliefs(0)
```

## Add Parameters from Configuration
```python
processor.add_from_config("snapshots.interval")
```

## Creating Custom Columns
```python
class MyPolygraphAnalysis(PolygraphAnalysis):
    def __init__(self, path):
        super().__init__(path)

    def edges(self):
        edges_list = [nx.number_of_nodes(graph) for graph in self.get_graphs()]
        self.dataframe["edges"] = edges_list
        
    def majority(self):
        def get_majority(iterations, threshold=0.5):
            average_by_iteration = iterations.groupby(level='iteration').mean()
            iterations_above_threshold = average_by_iteration[average_by_iteration['beliefs'] > threshold]
            return iterations_above_threshold.index.tolist()[0]
    
        majority_list_05 = []
        majority_list_075 = []
        for iteration in self.get_beliefs():
            majority_list_05.append(get_majority(iteration))
            majority_list_075.append(get_majority(iteration, 0.75))
            
        self.dataframe['majority_05'] = majority_list_05
        self.dataframe['majority_075'] = majority_list_075
```

```python
x = MyPolygraphAnalysis("~/polygraphs-cache/results")
x.add(x.nodes(), x.majority())
x.add_from_config("snapshots.interval", "reliability")
x.get()
```
