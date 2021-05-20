"""
PolyGraph OGB datasets
"""
import os

import pandas as pd
import numpy as np
import torch
import dgl

from .dataset import PolyGraphDataset
from .utils import unzip


class Collab(PolyGraphDataset):
    """
    The ogbl-collab dataset from https://ogb.stanford.edu/docs/linkprop/#ogbl-collab.

    Basic dataset statistics:

        Nodes:   235,868
        Edges: 1,179,052

    Other information:

        - The network is undirected.
    """

    def __init__(self):
        origin = "http://snap.stanford.edu/ogb/data/linkproppred/collab.zip"
        super().__init__(directed=False, data=origin)

    @property
    def collection(self):
        return "ogbl"

    @staticmethod
    def numpy(filename):
        """
        Loads a .csv.gz file into a numpy array.
        """
        return pd.read_csv(filename, compression="gzip", header=None).values

    def read(self):
        # pylint: disable=no-member
        # self.data

        # Fetch all dataset files (equivalent to `self.data.fetch(self.folder)`)
        self.fetchall()

        # Try extract .zip file
        unzip(self.data.origin)

        # Upon extraction, the contents of `collab.zip` are in `collab`
        collab = os.path.join(self.folder, "collab")
        assert os.path.isdir(collab)

        # Read number of nodes
        num_nodes = Collab.numpy(
            os.path.join(collab, "raw/num-node-list.csv.gz")
        ).item()

        # Read edges. The shape of the array is (m, 2),
        # so we transpose it to get a (2, m) array
        edges = Collab.numpy(os.path.join(collab, "raw/edge.csv.gz")).T.astype(np.int64)

        if not self.directed:
            # Create birectional edges. First, copy each edge:
            #
            # [[a, a, ...]  becomes [[a, a, a, a, ...]
            #  [b, c, ...]]          [b, b, c, c, ...]]
            #
            edges = np.repeat(edges, 2, axis=1)
            #
            # Then, Swap source-destination pairs (diagonally)
            #
            # [[a, a, a, a, ...]  becomes [[a, b, a, c, ...]
            #  [b, b, c, c, ...]]          [b, a, c, a, ...]]
            #
            edges[0, 1::2] = edges[1, 0::2]
            edges[1, 1::2] = edges[0, 0::2]

        # Consider two edges features: a year and a weight representing
        # the number of co-authored papers in that year.
        #
        # Both features have shape (m, 1) of type np.int64.
        weight = Collab.numpy(os.path.join(collab, "raw/edge_weight.csv.gz"))
        tstamp = Collab.numpy(os.path.join(collab, "raw/edge_year.csv.gz"))

        if not self.directed:
            # Create bidirectional edge features
            weight = np.repeat(weight, 2, axis=0)
            tstamp = np.repeat(tstamp, 2, axis=0)

        # Create a DGL graph
        graph = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
        # Set edge features
        graph.edata["w"] = torch.from_numpy(weight)
        graph.edata["t"] = torch.from_numpy(tstamp)
        return graph
