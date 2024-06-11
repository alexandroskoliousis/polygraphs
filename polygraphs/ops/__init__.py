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
    UnreliableNetworkIdealOp,
    UnreliableNetworkBasicGullibleUniformOp,
    UnreliableNetworkBasicGullibleBinomialOp,
    UnreliableNetworkBasicGullibleNegativeEpsOp,
    UnreliableNetworkBasicAlignedUniformOp,
    UnreliableNetworkBasicAlignedBinomialOp,
    UnreliableNetworkBasicAlignedNegativeEpsOp,
    UnreliableNetworkBasicUnalignedUniformOp,
    UnreliableNetworkModifiedAlignedUniformOp,
    UnreliableNetworkModifiedAlignedBinomialOp,
    UnreliableNetworkModifiedAlignedNegativeEpsOp
)


from .weightedops import BalaGoyalWeighted2Op, BalaGoyalWeightedOp

__all__ = [
    "PolyGraphOp",
    "BalaGoyalOp",
    "NoOp",
    "OConnorWeatherallOp",
    "OConnorWeatherallSquareRootDistanceOp",
    "OConnorWeatherallSquareDistanceOp",
    "UnreliableNetworkIdealOp",
    "UnreliableNetworkBasicGullibleUniformOp",
    "UnreliableNetworkBasicGullibleBinomialOp",
    "UnreliableNetworkBasicGullibleNegativeEpsOp",
    "UnreliableNetworkBasicAlignedUniformOp",
    "UnreliableNetworkBasicAlignedBinomialOp",
    "UnreliableNetworkBasicAlignedNegativeEpsOp",
    "UnreliableNetworkBasicUnalignedUniformOp",
    "UnreliableNetworkModifiedAlignedUniformOp",
    "UnreliableNetworkModifiedAlignedBinomialOp",
    "UnreliableNetworkModifiedAlignedNegativeEpsOp",
    "BalaGoyalWeightedOp",
    "BalaGoyalWeighted2Op",
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
