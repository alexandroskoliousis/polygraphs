"""
PolyGraph common operator
"""

from .core import PolyGraphOp


class NoOp(PolyGraphOp):
    """
    No operator
    """

    def messagefn(self):
        """
        Message function
        """

        def function(edges):  # pylint: disable=unused-argument
            return {}

        return function

    def reducefn(self):
        """
        Reduce function
        """

        def function(nodes):  # pylint: disable=unused-argument
            return {}

        return function
