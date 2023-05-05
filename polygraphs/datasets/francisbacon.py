"""
PolyGraph Six Degrees of Francis Bacon Dataset
"""
import os

import pandas as pd
import numpy as np
import torch
import dgl
import networkx as nx

from .dataset import PolyGraphDataset

class FrancisBacon(PolyGraphDataset):
    """
    The Six Degrees of Francis Bacon Dataset from http://sixdegreesoffrancisbacon.com

    Basic dataset statistics:

        Nodes:  13,032
        Edges: 171,540

    Other information:

        - The gml file for this network must be manually placed in the
          data cache:
          ~/polygraphs-cache/data/francisbacon/francisbacon.gml.gz

          This file can be generated using the sixdegreesoffrancisbacon.ipynb
          notebook inside the scripts folder.
    """

    def __init__(self):
        super().__init__(directed=False)

    @property
    def collection(self):
        return "francisbacon"

    def read(self):
        assert os.path.isdir(self.folder)
        gml_file = os.path.join(self.folder, "francisbacon.gml.gz")
        assert os.path.isfile(gml_file), "File not found: francisbacon.gml.gz"

        # Load graph from GML file
        G = nx.read_gml(gml_file, destringizer=int)

        # Load graph using edge list so that we preserve node ids
        edges = [torch.tensor((edge[0], edge[1])) for edge in list(nx.to_edgelist(G))]
        graph = dgl.graph(edges)
        # Convert to a bi-directed DGL graph because this is an undirected graph
        graph = dgl.to_bidirected(graph)

        return graph
