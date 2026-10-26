from abc import ABC, abstractmethod

import torch

from ... import init
from ...logger import getlogger

log = getlogger()

_REGISTRY = {}


def register(name):
    """
    Class decorator that makes a sampler selectable by `name`.
    """

    def decorator(cls):
        key = name.lower()
        if not (isinstance(cls, type) and issubclass(cls, Sampler)):
            raise TypeError(f"{cls!r} must subclass Sampler")
        _REGISTRY[key] = cls
        return cls

    return decorator


def create(name, size, params):
    """
    Creates the sampler registered under `name`.
    """
    key = str(name).lower() if name else None
    if key not in _REGISTRY:
        raise ValueError(f"""Unknown sampler {name!r}.
Valid options: {", ".join(tuple(sorted(_REGISTRY)))}""")
    return _REGISTRY[key](size, params)


class Sampler(ABC):
    """
    Base class for samplers.
    """

    def __init__(self, size, params):
        self.size = size
        self.trials = params.trials

    @abstractmethod
    def sample(self): ...


@register("uniform")
class UniformSampler(Sampler):
    def sample(self):
        # Create uniform sampler for unreliable nodes
        return torch.distributions.uniform.Uniform(
            init.zeros(self.size), init.zeros(self.size) + (self.trials + 1)
        ).sample()


@register("binomial")
class BinomialSampler(Sampler):
    def sample(self):
        # Unreliable Binomial Sampler:
        # Payoff: p = 0.5, epsilon = 0
        probs = init.halfs(self.size)

        # Number of Bernoulli trials
        count = init.zeros(self.size) + self.trials

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        return torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        ).sample()


@register("negative_eps")
class NegativeEpsSampler(Sampler):
    def __init__(self, size, params):
        super().__init__(size, params)
        self.epsilon = params.epsilon

    def sample(self):
        # Unreliable Binomial Sampler with a negative epsilon
        probs = init.halfs(self.size) - self.epsilon

        # Number of Bernoulli trials
        count = init.zeros(self.size) + self.trials

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._unreliable_sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        ).sample()
