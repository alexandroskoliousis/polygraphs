"""
Complex ops that contain unreliable nodes
"""

import torch
from . import math
from .. import init
from .common import BalaGoyalOp
from ..logger import getlogger

log = getlogger()


class UnreliableOp(BalaGoyalOp):
    """
    Baseclass for Unreliable Ops that draw from an unreliable sampler

    Unreliable networks, Part 1

    There are two types of nodes, reliable and unreliable ones.

    Upon receipt, all nodes apply Bayes rule.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # The shape of all node attributes
        self._size = (graph.num_nodes(),)

        # Configure network reliability
        # Draw binary numbers from a Bernoulli distribution
        # (1s denote reliable nodes)
        self._reliability = torch.bernoulli(torch.ones(self._size) * params.reliability)

        # Given a list of unreliable nodes, make them unreliable
        for node in params.unreliablenodes:
            self._reliability[node] = 0

        # Store network reliability
        graph.ndata["reliability"] = self._reliability.to(device=self._device)

        # Count number of reliable nodes (for debugging purposes)
        nr = torch.count_nonzero(self._reliability)
        log.info(f"{nr.item()} out of {graph.num_nodes()} nodes are reliable")

    def sample(self):
        """
        Draws a sample from a reliable and unreliable sample
        for reliable and unreliable nodes
        """
        # pylint: disable=invalid-name
        # Sample reliable distribution
        b = self._sampler.sample()
        # Sample unreliable distribution
        u = self._unreliable_sampler.sample()
        # Combine samples
        return b * self._reliability + u * (1 - self._reliability)


class UnreliableNetworkBasicGullibleUniformOp(UnreliableOp):
    """
    Unreliable nodes draw from a uniform distribution
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Create uniform sampler for unreliable nodes
        self._unreliable_sampler = torch.distributions.uniform.Uniform(
            init.zeros(self._size), init.zeros(self._size) + (params.trials + 1)
        )


class UnreliableNetworkBasicGullibleBinomialOp(UnreliableOp):
    """
    Unreliable nodes' evidence follow a binomial distribution
    with epsilon of 0/p = 0.5.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Unreliable Binomial Sampler:
        # Payoff: p = 0.5, epsilon = 0
        probs = init.halfs(self._size)

        # Number of Bernoulli trials
        count = init.zeros(self._size) + params.trials

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._unreliable_sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )


class UnreliableNetworkBasicGullibleNegativeEpsOp(UnreliableOp):
    """
    Unreliable nodes' evidence follow a binomial distribution
    with a negative epsilon.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Unreliable Binomial Sampler with a negative epsilon
        probs = init.halfs(self._size) - params.epsilon

        # Number of Bernoulli trials
        count = init.zeros(self._size) + params.trials

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._unreliable_sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )


# ------------------------------------------------------------------------------
# Aligned Ops


class AlignedOp(UnreliableOp):
    """
    Baseclass for Aligned Ops that uses Jeffreys rule

    Unreliable networks, Part 2

    There are two types of nodes, reliable and unreliable ones.

    Upon receipt, all nodes apply Jeffrey's rule.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Store network reliability in the graph
        graph.ndata["reliability"] = self._reliability.to(device=self._device)

    def messagefn(self):
        """
        Message function
        """

        def function(edges):
            return {
                "payoffs": edges.src["payoffs"],
                "reliability": edges.src["reliability"],
            }

        return function

    def reducefn(self):
        """
        Reduce function
        """

        def function(nodes):
            # Log probability of successful trials
            logits = nodes.data["logits"]
            # Prior, P(H) (aka. belief)
            prior = nodes.data["beliefs"]

            # Number of nodes and number of neighbours per node (incoming messages)
            _, neighbours = nodes.mailbox["reliability"].shape
            for i in range(neighbours):
                # A node receives evidence E from its i-th neighbour, say Jill,
                # denoting the number of successful trials and the total number
                # of trials she observed
                values = nodes.mailbox["payoffs"][:, i, 0]
                trials = nodes.mailbox["payoffs"][:, i, 1]

                # Evidence, E
                evidence = math.Evidence(logits, values, trials)

                # Get i-th neighbour reliability
                reliability = nodes.mailbox["reliability"][:, i]

                # log.info(f"Neighbour {i:2d}: reliability {reliability}")

                # Compute posterior belief, in light of soft uncertainty
                # (i.e., network unreliability)
                posterior = math.jeffrey(prior, evidence, reliability)

                # Consider next neighbour
                prior = posterior

            # Return posterior beliefs for each neighbour
            return {"beliefs": posterior}

        return function


class UnreliableNetworkBasicAlignedUniformOp(AlignedOp):
    """
    Aligned op where unreliable nodes draw from a uniform distibution
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Create uniform sampler
        self._unreliable_sampler = torch.distributions.uniform.Uniform(
            init.zeros(self._size), init.zeros(self._size) + (params.trials + 1)
        )


class UnreliableNetworkBasicAlignedBinomialOp(AlignedOp):
    """
    Unreliable nodes' evidence follow a binomial distribution
    with p=0.5/e=0
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Unreliable Binomial Sampler:
        # Payoff: p = 0.5, epsilon = 0
        probs = init.halfs(self._size)

        # Number of Bernoulli trials
        count = init.zeros(self._size) + params.trials

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._unreliable_sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )


class UnreliableNetworkBasicAlignedNegativeEpsOp(AlignedOp):
    """
    Unreliable nodes' evidence follow a binomial distribution
    with a negative epsilon.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Unreliable Binomial Sampler:
        # Payoff: p = 0.5, epsilon = - params.epsilon
        probs = init.halfs(self._size) - params.epsilon

        # Number of Bernoulli trials
        count = init.zeros(self._size) + params.trials

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._unreliable_sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )


# ------------------------------------------------------------------------------
# UnAligned Ops


class UnalignedOp(AlignedOp):
    """
    Baseclass for Unaligned Ops

    Unreliable networks, Part 3

    There are two types of nodes, reliable and unreliable ones.

    Upon receipt, all nodes apply Jeffrey's rule.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Configure network trust on evidence
        self._trust = torch.ones(self._size) * params.trust
        # Store trust in graph
        graph.ndata["trust"] = self._trust.to(device=self._device)

    def messagefn(self):
        """
        Message function
        """

        def function(edges):
            return {"payoffs": edges.src["payoffs"], "trust": edges.src["trust"]}

        return function

    def reducefn(self):
        """
        Reduce function
        """

        def function(nodes):
            # Log probability of successful trials
            logits = nodes.data["logits"]
            # Prior, P(H) (aka. belief)
            prior = nodes.data["beliefs"]

            # Number of nodes and number of neighbours per node (incoming messages)
            _, neighbours = nodes.mailbox["trust"].shape
            for i in range(neighbours):
                # A node receives evidence E from its i-th neighbour, say Jill,
                # denoting the number of successful trials and the total number
                # of trials she observed
                values = nodes.mailbox["payoffs"][:, i, 0]
                trials = nodes.mailbox["payoffs"][:, i, 1]

                # Evidence, E
                evidence = math.Evidence(logits, values, trials)

                # Get i-th neighbour reliability
                trust = nodes.mailbox["trust"][:, i]

                # log.info(f"Neighbour {i:2d}: reliability {reliability}")

                # Compute posterior belief, in light of soft uncertainty
                # (i.e., network unreliability)
                posterior = math.jeffrey(prior, evidence, trust)

                # Consider next neighbour
                prior = posterior

            # Return posterior beliefs for each neighbour
            return {"beliefs": posterior}

        return function


class UnreliableNetworkBasicUnalignedUniformOp(UnalignedOp):
    """
    Unreliable nodes' evidence follow a uniform distribution.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # Create uniform sampler
        self._unreliable_sampler = torch.distributions.uniform.Uniform(
            init.zeros(self._size), init.zeros(self._size) + (params.trials + 1)
        )
