from abc import ABC, abstractmethod

import torch

from .. import math
from ...logger import getlogger

log = getlogger()

_REGISTRY = {}


def register(name):
    """
    Class decorator that makes a policy selectable by `name`.
    """

    def decorator(cls):
        key = name.lower()
        if not (isinstance(cls, type) and issubclass(cls, UpdatePolicy)):
            raise TypeError(f"{cls!r} must subclass UpdatePolicy")
        if key in _REGISTRY:
            # Re-running a notebook cell redefines the class; replace it
            log.info(f"Replacing registered policy {name!r}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def create(name, op, graph, params):
    """
    Creates the policy registered under `name`.
    """
    key = str(name).lower() if name else None
    if key not in _REGISTRY:
        raise ValueError(f"""Unknown updating policy {name!r}.
Valid options: {", ".join(tuple(sorted(_REGISTRY)))}""")
    return _REGISTRY[key](op, graph, params)


class UpdatePolicy(ABC):
    """
    Base class for updating policies. Subclass PooledUpdate or
    PerNeighbourUpdate rather than this class directly.
    """

    # Sender attributes that messages carry, in addition to payoffs
    message_fields = ()

    def __init__(self, op, graph, params):
        self.op = op
        self.params = params
        self.setup(graph)  # Call setup function to modfify graph

    def setup(self, graph):
        """
        Adds any node attributes the policy needs.
        """
        pass

    def filterfn(self):
        """
        Keeps only edges whose source ran trials (i.e., believes B is better).
        """

        def function(edges):
            return torch.gt(edges.src["payoffs"][:, 1], 0.0)

        return function

    def messagefn(self):
        """
        Sends payoffs, plus any sender attributes in message_fields.
        """
        fields = ("payoffs",) + tuple(self.message_fields)

        def function(edges):
            return {field: edges.src[field] for field in fields}

        return function

    @abstractmethod
    def reducefn(self): ...

    @abstractmethod
    def applyfn(self): ...


def _evidence(nodes, payoffs):
    """
    Helper function to structure Evidence.

    A node observes evidence E denoting the number of successful trials (`values`),
    and the total number of trials (`trials`). The probability of successful trials
    is given by `logits`.
    """

    logits = nodes.data["logits"]
    values = nodes.data["payoffs"][:, 0]
    trials = nodes.data["payoffs"][:, 1]
    return math.Evidence(nodes.data["logits"], payoffs[..., 0], payoffs[..., 1])


class PooledUpdate(UpdatePolicy):
    """
    Sums the evidence from all neighbours, then updates beliefs once.
    """

    def reducefn(self):
        def function(nodes):
            # Reduce the incoming trials from all neighbours
            return {"payoffs": torch.sum(nodes.mailbox["payoffs"], dim=1)}

        return function

    def applyfn(self):
        def function(nodes):
            # Update beliefs in apply stage
            prior = nodes.data["beliefs"]
            evidence = _evidence(nodes, nodes.data["payoffs"])
            return {"beliefs": self.update(prior, evidence)}

        return function

    @abstractmethod
    def update(self, prior, evidence):
        """
        Returns the posterior belief given the pooled evidence.
        """
        ...


class PerNeighbourUpdate(UpdatePolicy):
    """
    Updates beliefs with Jeffrey's rule, once per neighbour; each posterior
    becomes the prior for the next neighbour.
    """

    def reducefn(self):
        def function(nodes):
            prior = nodes.data["beliefs"]
            neighbours = nodes.mailbox["payoffs"].shape[1]
            for i in range(neighbours):
                # Evidence E from the i-th neighbour: successes and trials
                evidence = _evidence(nodes, nodes.mailbox["payoffs"][:, i])
                weight = self.weight(nodes, i, prior, evidence)
                prior = math.jeffrey(prior, evidence, weight)
            return {"beliefs": prior}

        return function

    def applyfn(self):
        # Beliefs were already updated in the reduce phase
        def function(nodes):
            return {"beliefs": nodes.data["beliefs"]}

        return function

    @abstractmethod
    def weight(self, nodes, i, prior, evidence):
        """
        Returns the certainty in the evidence from the i-th neighbour.
        """
        ...


# Policies


@register("gullible")
class GullibleUpdate(PooledUpdate):
    """
    Bayes' rule on pooled evidence: every sender is believed.
    """

    def update(self, prior, evidence):
        return math.bayes(prior, evidence)


@register("ideal")
class IdealUpdate(GullibleUpdate):
    """
    As gullible, but messages from unreliable senders are dropped (including
    an unreliable node's message to itself, so unreliable nodes never update).
    """

    needs_unreliable_sampler = False

    def filterfn(self):
        has_evidence = super().filterfn()

        def function(edges):
            return has_evidence(edges) & edges.src["reliability"].bool()

        return function


@register("modified")
class ModifiedUpdate(PooledUpdate):
    """
    Jeffrey's rule on pooled evidence, weighted by params.reliability.
    """

    def update(self, prior, evidence):
        return math.jeffrey(prior, evidence, self.params.reliability)


@register("aligned")
class AlignedUpdate(PerNeighbourUpdate):
    """
    Jeffrey's rule per neighbour, weighted by the sender's reliability.
    """

    message_fields = ("reliability",)

    def weight(self, nodes, i, prior, evidence):
        return nodes.mailbox["reliability"][:, i]


@register("unaligned")
class UnalignedUpdate(PerNeighbourUpdate):
    """
    Jeffrey's rule per neighbour, weighted by a fixed trust, params.trust.
    """

    # message_fields = ("trust",)

    def setup(self, graph):
        pass
        # Configure network trust on evidence
        # trust = torch.ones(self.op._size) * self.params.trust
        # Store trust in graph
        # graph.ndata["trust"] = trust.to(device=self.op.device)

    def weight(self, nodes, i, prior, evidence):
        # Why do we save the trust in this op, cant we just return trust
        return self.params.trust
        # This would avoid having to do the setup trust in graph
        # return nodes.mailbox["trust"][:, i]
