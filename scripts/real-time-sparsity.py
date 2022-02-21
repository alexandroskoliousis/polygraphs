import h5py


def sparsity(graph):
    """
    Returns sparsity level of given DGL graph.
    """
    # Remove self-loops
    g = dgl.remove_self_loop(graph)
    # Assumes an adjacency matrix of size N x N with M non-zero values
    return g.num_edges() / (g.num_nodes() ** 2)


def acc(graph):
    """
    Returns average clustering coefficient.
    """
    graphx = nx.DiGraph(dgl.to_networkx(graph))
    return nx.algorithms.cluster.average_clustering(graphx)


def apl(graph):
    """
    Returns average shortest path length.
    """
    graphx = nx.DiGraph(dgl.to_networkx(graph))
    return nx.average_shortest_path_length(graphx)


def filterfn(edges):
    return torch.le(edges.src["beliefs"], 0.5)
   

def postprocess(directory, id):
    """
    Post-process graph snapshots
    """
    # Resulting hashtable
    ht = {}
    graphs, _ = dgl.load_graphs(os.path.join(directory, f"{id}.bin"))
    graph = graphs[0]
    fp = h5py.File(os.path.join(directory, f"{id}.hd5"), "r")
    _keys = [int(key) for key in fp.keys()]
    _keys = sorted(_keys)
    for key in _keys:
        graph.ndata["beliefs"] = torch.tensor(fp[str(key)][:])
        # Filter any edge whose source has belief less than 0.5
        inactive = graph.filter_edges(filterfn)
        # Create subgraph
        subgraph = dgl.remove_edges(graph, inactive)
        # Debugging
        s = 'DBG> '
        s += f'From G({graph.num_nodes():3d}, {graph.num_edges():3d})'
        s += f'to G\'({subgraph.num_nodes():3d}, {subgraph.num_edges():3d})'
        print(s)
        # Compute network statistics
        ht[key] = sparsity(subgraph)
    return ht


ht = postprocess("data/test3", 1)
print(ht)