"""
PolyGraph models
"""

import sys

from .core import PolyGraphOp

from .common import (
    NoOp,
    BalaGoyalOp,
    OConnorWeatherallOp,
    OConnorWeatherallSquareRootDistanceOp,
    OConnorWeatherallSquareDistanceOp,
)

from .complex import UnreliableNetworkBasicGullibleOp, UnreliableNetworkBasicAlignedOp


__all__ = [
    "PolyGraphOp",
    "BalaGoyalOp",
    "NoOp",
    "OConnorWeatherallOp",
    "OConnorWeatherallSquareRootDistanceOp",
    "OConnorWeatherallSquareDistanceOp",
    "UnreliableNetworkBasicGullibleOp",
    "UnreliableNetworkBasicAlignedOp"
]


def getbyname(name):
    """
    Returns PolyGraph operator by name.
    """
    assert name and isinstance(name, str)

    def _find():
        for operator in __all__:
            if name.lower() == operator.lower():
                return getattr(sys.modules[__name__], operator)
        return None

    operator = _find()
    if operator is None:
        raise Exception(f"Invalid operator name: {name}")
    return operator
