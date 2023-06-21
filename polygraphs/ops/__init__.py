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
    OConnorWeatherallSquareDistanceOp
)

from .complex import (
    UnreliableNetworkBasicGullibleUniformOp,
    UnreliableNetworkBasicGullibleBinomialOp,
    UnreliableNetworkBasicGullibleNegativeEpsOp,
    UnreliableNetworkBasicAlignedUniformOp,
    UnreliableNetworkBasicAlignedBinomialOp,
    UnreliableNetworkBasicAlignedNegativeEpsOp,
    UnreliableNetworkBasicUnalignedUniformOp
)


from .weightedops import BalaGoyalWeighted2Op, BalaGoyalWeightedOp

__all__ = [
    "PolyGraphOp",
    "BalaGoyalOp",
    "NoOp",
    "OConnorWeatherallOp",
    "OConnorWeatherallSquareRootDistanceOp",
    "OConnorWeatherallSquareDistanceOp",
    "UnreliableNetworkBasicGullibleUniformOp",
    "UnreliableNetworkBasicGullibleBinomialOp",
    "UnreliableNetworkBasicGullibleNegativeEpsOp",
    "UnreliableNetworkBasicAlignedUniformOp",
    "UnreliableNetworkBasicAlignedBinomialOp",
    "UnreliableNetworkBasicAlignedNegativeEpsOp",
    "UnreliableNetworkBasicUnalignedUniformOp",
    "BalaGoyalWeightedOp",
    "BalaGoyalWeighted2Op"
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
