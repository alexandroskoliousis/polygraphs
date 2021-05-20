"""
PolyGraph models
"""

from .core import PolyGraphOp

from .common import NoOp, BalaGoyalOp, OConnorWeatherallOp


__all__ = ["PolyGraphOp", "BalaGoyalOp", "NoOp", "OConnorWeatherallOp"]


def getbyname(name):
    """
    Returns PolyGraph operator by name.
    """
    assert name and isinstance(name, str)

    def _find():
        for operator in __all__:
            if name.lower() == operator.lower():
                return getattr(__file__, operator)
        return None

    operator = _find()
    if operator is None:
        raise Exception(f"Invalid operator name: {name}")
    return operator
