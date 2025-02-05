import os
from collections import defaultdict
import networkx as nx

def normalise_gml(file_path):
    """
    Simulate the normalisation of a GML file in Polygraphs
    """
    # Resolve GML file
    gml_file = os.path.abspath(os.path.expanduser(file_path))
    assert os.path.isfile(gml_file), "GML file not found"
    # Load graph from GML file
    G = nx.read_gml(gml_file, destringizer=int)
    # Load graph using edge list so that we preserve node ids
    edges = []

    for edge in list(nx.to_edgelist(G)):
        try:
            edges.append((int(edge[0]), int(edge[1])))
        except:
            raise ValueError("GML File: Node IDs should be specified as integers")

    # Create normalised table
    tbl = defaultdict(lambda: len(tbl))

    # Normalise node identifiers (from 0 to N) using default dict
    _ = [(tbl[edge[0]], tbl[edge[1]]) for edge in edges]

    # Return the dictionary
    return dict(tbl)