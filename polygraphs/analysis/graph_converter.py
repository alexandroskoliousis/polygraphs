import dgl  # Importing Deep Graph Library (DGL) for graph manipulation
import networkx as nx  # Importing networkx library for working with graphs


class Graphs:
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
