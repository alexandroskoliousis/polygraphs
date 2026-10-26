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

from .unreliable import UnreliableOp, UNRELIABLE_ALIASES

from .weightedops import BalaGoyalWeighted2Op, BalaGoyalWeightedOp

from ..logger import getlogger

log = getlogger()

__all__ = [
    "PolyGraphOp",
    "BalaGoyalOp",
    "NoOp",
    "OConnorWeatherallOp",
    "OConnorWeatherallSquareRootDistanceOp",
    "OConnorWeatherallSquareDistanceOp",
    "UnreliableOp",
    "BalaGoyalWeightedOp",
    "BalaGoyalWeighted2Op",
]


# Operators registered from outside the package (e.g. from a notebook)
_REGISTERED = {}


def _builtin(name):
    """
    Returns the built-in operator called `name`, or None.
    """
    for operator in __all__:
        if name.lower() == operator.lower() and operator != "register_op":
            return getattr(sys.modules[__name__], operator)
    if name.lower() in (alias.lower() for alias in UNRELIABLE_ALIASES):
        return UnreliableOp
    return None


def register_op(name):
    """
    Class decorator that makes an operator selectable as `params.op = name`.
    Built-in names cannot be replaced.
    """

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, PolyGraphOp)):
            raise TypeError(f"{cls!r} must subclass PolyGraphOp")
        if _builtin(name) is not None:
            raise ValueError(f"{name!r} is a built-in operator name")
        if name.lower() in _REGISTERED:
            log.info(f"Replacing registered operator {name!r}")
        _REGISTERED[name.lower()] = cls
        return cls

    return decorator


def getbyname(name):
    """
    Returns PolyGraph operator by name. The original unreliable-network class
    names resolve to UnreliableOp, which reads its configuration from params.
    """
    assert name and isinstance(name, str)

    operator = _builtin(name) or _REGISTERED.get(name.lower())
    if operator is None:
        valid = [n for n in __all__ if n != "register_op"] + sorted(UNRELIABLE_ALIASES)
        valid += sorted(_REGISTERED)
        raise ValueError(
            f"Invalid operator name: {name}. Valid names: {', '.join(valid)}"
        )
    return operator
