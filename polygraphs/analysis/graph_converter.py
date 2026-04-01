import networkx as nx
import ptgraph

# DGL is only needed for legacy .bin files
try:
    import dgl

    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


class GraphConverter:
    """
    Loads graph files and converts them to networkx Graph objects.

    Supports two formats:
        - .pt  (ptgraph) — loaded with ptgraph, no DGL dependency
        - .bin (legacy)  — loaded with DGL (requires DGL to be installed)
    """

    # ------------------------------------------------------------------ #
    #  .pt (ptgraph) loading
    # ------------------------------------------------------------------ #

    def _load_ptgraph_as_networkx(self, filepath):
        """Load a .pt file via ptgraph and convert to an undirected networkx Graph."""
        graphs, _ = ptgraph.load_graphs(filepath)
        graph = ptgraph.remove_self_loop(graphs[0])
        G = nx.Graph(ptgraph.to_networkx(graph))
        G.pg = {"ndata": graph.ndata, "edata": graph.edata}
        return G

    # ------------------------------------------------------------------ #
    #  .bin (DGL legacy) loading
    # ------------------------------------------------------------------ #

    def _load_bin_as_networkx(self, filepath):
        """Load a .bin file via DGL and convert to an undirected networkx Graph."""
        if not DGL_AVAILABLE:
            raise ImportError(
                "DGL is required to load legacy .bin graph files but is not installed."
            )
        graphs, _ = dgl.load_graphs(filepath)
        graph = dgl.remove_self_loop(graphs[0])
        G = nx.Graph(dgl.to_networkx(graph))
        G.pg = {"ndata": graph.ndata, "edata": graph.edata}
        return G

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_networkx_object(self, filepath):
        """
        Load a graph file and return a networkx Graph.

        Parameters:
            filepath: Path to a .pt or .bin file. Format is inferred from extension.
        """
        if str(filepath).endswith(".pt"):
            return self._load_ptgraph_as_networkx(filepath)
        else:
            return self._load_bin_as_networkx(filepath)


class Graphs:
    """
    Lazily loads and caches networkx graphs for a set of simulations.

    Supports both .pt and legacy .bin files via GraphConverter.
    """

    def __init__(self, dataframe, graph_converter=None):
        self.bin_file_path = dataframe["bin_file_path"]
        self.graph_converter = graph_converter or GraphConverter()
        self.graphs = [None] * len(dataframe)
        self.index = 0

    def __getitem__(self, index):
        if index >= len(self.graphs):
            raise IndexError("Simulation index out of range")
        return self.get(index)

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.graphs):
            raise StopIteration
        value = self.get(self.index)
        self.index += 1
        return value

    def load(self, index):
        """Load a single graph by index."""
        filepath = self.bin_file_path.iloc[index]
        graph = self.graph_converter.get_networkx_object(filepath)
        self.graphs[index] = graph

    def get(self, index):
        """Return the networkx graph at *index*, loading it lazily if needed."""
        if self.graphs[index] is not None:
            return self.graphs[index]
        if index < len(self.graphs):
            self.load(index)
            return self.graphs[index]
        raise IndexError("Simulation index out of range")
