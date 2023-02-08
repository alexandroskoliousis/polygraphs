import dgl
import torch
import networkx as nx

from .. import init
from . import common


class BalaGoyalWeightedOp(common.BalaGoyalOp):
    """
    Initial beliefs weighted by centrality.
    """

    def __init__(self, graph, params):

        super().__init__(graph, params)

        # Modify weights
        size = (graph.num_nodes(),)

        G = dgl.to_networkx(dgl.remove_self_loop(graph))
        centrality = nx.degree_centrality(G)
        weights = torch.Tensor(list(centrality.values()))

        graph.ndata["beliefs"] = init.ones(size) * weights


class BalaGoyalWeighted2Op(common.BalaGoyalOp):
    """
    Initial beliefs weighted by centrality.
    """

    def __init__(self, graph, params):

        super().__init__(graph, params)

        # Modify weights
        size = (graph.num_nodes(),)

        G = dgl.to_networkx(dgl.remove_self_loop(graph))
        centrality = nx.degree_centrality(G)
        weights = torch.Tensor(list(centrality.values()))

        graph.ndata["beliefs"] = init.halfs(size) * weights
