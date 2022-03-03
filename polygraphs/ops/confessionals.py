"""
Confessional models
"""
import torch

from . import core


class BeliefConfessionalOp(core.PolyGraphOp):
    """
    Simple confessional model: Beliefs
    """

    def messagefn(self):
        """
        Message function
        """

        def function(edges):
            return {"beliefs": edges.src["beliefs"]}

        return function

    def reducefn(self):
        """
        Reduce function
        """

        def function(nodes):
            return {"payoffs": torch.mean(nodes.mailbox["beliefs"])}

        return function
