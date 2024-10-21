import dgl  # Importing Deep Graph Library (DGL) for graph manipulation
import networkx as nx  # Importing networkx library for working with graphs


class GraphConverter:
    def get_graph_object(self, filepath):
        # Load graph object from the specified filepath using dgl
        graph, _ = dgl.load_graphs(filepath)
        return graph

    def convert_graph_networkx(self, graph):
        # Remove self-loops from the graph and convert it to a networkx Graph object
        graph = dgl.remove_self_loop(graph[0])
        G = nx.Graph(dgl.to_networkx(graph))
        return G

    def get_networkx_object(self, filepath):
        # Get a networkx Graph object from the specified filepath
        return self.convert_graph_networkx(self.get_graph_object(filepath))


class Graphs:
    """
    The Graphs class stores the graphs of simulations that have been
    explicitly loaded for analysis using the GraphConverter

    This class provides an iterator and get item to access graphs
    """

    def __init__(self, dataframe, graph_converter):
        self.bin_file_path = dataframe["bin_file_path"]
        self.graph_converter = graph_converter
        self.graphs = [None] * len(dataframe)
        self.data = GraphData([None] * len(dataframe), self)
        self.index = 0

    def __getitem__(self, index):
        if index > len(self.graphs):
            raise IndexError("Simulation index out of range")
        return self.get(index)

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.graphs):
            self.index = 0
            raise StopIteration
        value = self.get(self.index)
        self.index += 1
        return value

    def get(self, index):
        # Return a saved graph using its index or load from file
        if self.graphs[index] is not None:
            return self.graphs[index]
        elif index < len(self.graphs):
            dgl_graph = self.graph_converter.get_graph_object(self.bin_file_path[index])
            self.data._set(index, dgl_graph[0])
            nx_graph = self.graph_converter.convert_graph_networkx(dgl_graph)
            self.graphs[index] = nx_graph
            return self.graphs[index]
        else:
            raise IndexError("Simulation index out of range")

class GraphData:
    def __init__(self, data, graphs):
        self.data = data
        self.graphs = graphs
        self.index = 0

    def __getitem__(self, index):
        if index > len(self.data):
            raise IndexError("Simulation index out of range")
        return self.get(index)

    def __setitem__(self, key, value):
        raise AttributeError("Cannot set attribute for graph data")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            self.index = 0
            raise StopIteration
        value = self.get(self.index)
        self.index += 1
        return value

    def _set(self, key, value):
        self.data[key] = value

    def get(self, index):
        # Use the Graphs class to load data, which will place it inside this object
        if self.data[index] is not None:
            return self.data[index]
        elif index < len(self.data):
            self.graphs.get(index)
            return self.data[index]
        else:
            raise IndexError("Simulation index out of range")