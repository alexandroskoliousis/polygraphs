"""
Polygraph simulations and modules.
"""
import abc
import torch

from . import init


# Bayes rule operations


def _Pr(q, a, b):  # pylint: disable=invalid-name
    """
    An agent observes evidence that consists of `a` successful signals (with probability `q`)
    and `b` unsuccessful ones (with probability `1 - q`).

    Since signals are generated independently, we can multiple their probabilities.
    """
    return torch.pow(q, a) * torch.pow(1 - q, b)


def _marginal(p, q, a, b):  # pylint: disable=invalid-name
    """
    Marginal likelyhood, P(E) = P(B)P(E|B) + P(A)P(E|A)
    """
    return p * _Pr(q, a, b) + (1 - p) * _Pr(1 - q, a, b)


def _likelyhood(q, a, b):  # pylint: disable=invalid-name
    """
    Likelyhood, P(E|B). See _Pr() above.
    """
    return _Pr(q, a, b)


class PolyGraphOp(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base operator, from which all other operators are derived.
    """

    def __init__(self, graph, params):
        super().__init__()

        # The shape of all node attributes
        size = (graph.num_nodes(),)

        # Node beliefs that action B is better
        graph.ndata["belief"] = init.init(size, params.init).double()
        # Action B yields Bernoulli payoff of 1 (success) with probability p (= 0.5 + e)
        probs = init.halfs(size) + params.epsilon
        # Number of Bernoulli trials
        count = init.zeros(size) + params.trials
        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )

        # Store action B's probability of success as a graph node attribute
        graph.ndata["probability"] = probs

    def _sample(self):
        """
        Draws a sample from the binomial distribution.
        """
        return self._sampler.sample()

    def _trials(self):
        """
        Returns number of Bernoulli trials.
        """
        return self._sampler.total_count

    def experiment(self, graph):
        """
        Simulates an "experiment", where nodes who believe action B is better independently
        observe the payoffs from following action B by sampling a binomial distribution.
        """
        # Consider only nodes who believe action B is better
        mask = graph.ndata["belief"] > 0.5
        # Repeat mask
        mask = mask.tile((2, 1))
        # Sample distribution (a node observes only a few successful trials out of all trials)
        payoffs, trials = self._sample(), self._trials()
        result = torch.stack((payoffs, trials))
        # Apply mask
        result = result * mask
        # Store per-node payoffs as a graph node attribute
        graph.ndata["payoff"] = result.T

    @staticmethod
    def filterfn(edges):
        """
        Filters edges from graph; Returns a 1-D boolean tensor indicating
        whether the corresponding edge should be considered or not.
        """
        return torch.ones((len(edges),)).type(torch.bool)

    @abc.abstractstaticmethod
    def messagefn(edges):  # pylint: disable=unused-argument
        """
        Message function
        """
        raise NotImplementedError

    @abc.abstractstaticmethod
    def reducefn(nodes):  # pylint: disable=unused-argument
        """
        Reduce function
        """
        raise NotImplementedError

    @staticmethod
    def applyfn(nodes):  # pylint: disable=unused-argument
        """
        Update function
        """
        return nodes.data

    def forward(self, graph, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Forward function
        """
        # Generate a local signal (message to be sent)
        self.experiment(graph)
        # For use of static methods
        T = self.__class__
        # Filter valid edges along which messages will be sent
        edges = graph.filter_edges(T.filterfn)
        # Send messages along valid edges; and receive them at
        # edge destination nodes
        graph.send_and_recv(edges, T.messagefn, T.reducefn, T.applyfn)


class NoOp(PolyGraphOp):
    """
    No operator
    """

    @staticmethod
    def messagefn(edges):  # pylint: disable=unused-argument
        """
        Message function
        """
        return {}

    @staticmethod
    def reducefn(nodes):  # pylint: disable=unused-argument
        """
        Reduce function
        """
        return {}


class BalaGoyalOp(PolyGraphOp):
    """
    Learning from neighbours (Bala & Goyal, 1998)
    """

    @staticmethod
    def filterfn(edges):
        """
        Filter function
        """
        # Filter out edges whose source has no evidence to report
        return torch.gt(edges.src["payoff"][:, 1], 0.0)

    @staticmethod
    def messagefn(edges):
        """
        Message function
        """
        return {"payoff": edges.src["payoff"]}

    @staticmethod
    def reducefn(nodes):
        """
        Reduce function
        """
        return {"payoff": torch.sum(nodes.mailbox["payoff"], dim=1)}

    @staticmethod
    def applyfn(nodes):
        """
        Update function
        """
        # Probability that action B is successful
        probability = nodes.data["probability"]
        # A node observes evidence E denoting the number of
        # successful trials and the total number of trials
        success = nodes.data["payoff"][:, 0]
        failure = nodes.data["payoff"][:, 1] - success
        # Prior, P(H) (aka. belief that B is better)
        prior = nodes.data["belief"]
        # Marginal likelyhood, P(E)
        marginal = _marginal(prior, probability, success, failure)
        # Likelyhood, P(E|H)
        likelyhood = _likelyhood(probability, success, failure)
        # Posterior, P(H|E)
        posterior = prior * likelyhood / marginal

        # Update node attribute
        return {"belief": posterior}


class OConnorWeatherallOp(PolyGraphOp):
    """
    Scientific polarisation (O'Connor & Weatherall, 2018)
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # The shape of all node attributes
        size = (graph.num_nodes(),)
        # Multiplier that captures how quickly agents become uncertain about
        # the evidence of their peers as their beliefs diverge.
        graph.ndata["mistrust"] = init.zeros(size) + params.mistrust
        # With anti-updating, agents not only ignore evidence from neighbors
        # they do not trust, they also consider evidence less likely.
        graph.ndata["anti-updating"] = (
            torch.zeros(size).type(torch.bool) | params.antiupdating
        )

    @staticmethod
    def filterfn(edges):
        """
        Filters edges from graph; Returns a 1-D boolean tensor indicating
        whether the corresponding edge should be considered or not.
        """
        # Filter out edges whose source has no evidence to report
        return torch.gt(edges.src["payoff"][:, 1], 0.0)

    @staticmethod
    def messagefn(edges):
        """
        Message function
        """
        return {"payoff": edges.src["payoff"], "belief": edges.src["belief"]}

    @staticmethod
    def _isantiupdating(tensor):
        assert torch.all(tensor) or torch.all(torch.logical_not(tensor))
        return torch.all(tensor)

    @staticmethod
    def reducefn(nodes):  # pylint: disable=too-many-locals
        """
        Reduce function
        """
        #
        # Node attributes of the model
        #
        # Probability that action B is successful
        probability = nodes.data["probability"]
        # Multiplier that captures how quickly an agent becomes uncertain about evidence
        # received from neighbours whose beliefs are different from its own
        mistrust = nodes.data["mistrust"]
        # Prior, P(H) (aka. belief)
        prior = nodes.data["belief"]
        # Whether to use anti-updating rule or not (see below)
        antiupdating = OConnorWeatherallOp._isantiupdating(nodes.data["anti-updating"])

        # Number of nodes and number of neighbours per node (incoming messages)
        _, neighbours = nodes.mailbox["belief"].shape
        for i in range(neighbours):
            # A node receives evidence E from its i-th neighbour, say Jill, denoting the
            # number of successful trials and the total number of trials she observed
            success = nodes.mailbox["payoff"][:, i, 0]
            failure = nodes.mailbox["payoff"][:, i, 1] - success

            # The difference in belief between an agent and its i-th neighbour
            delta = torch.abs(prior - nodes.mailbox["belief"][:, i])

            # Marginal likelyhood, P(E)
            marginal = _marginal(prior, probability, success, failure)
            # Likelyhood, P(E|H)
            likelyhood = _likelyhood(probability, success, failure)
            # Posterior belief, P(H|E), obtained via strict conditionalization on evidence, E
            belief = prior * likelyhood / marginal

            # Posterior belief if evidence did not occur, P(H|~E)
            # P(~E|H) = 1 - P(E|H)
            # P(~E) = 1 - P(E)
            misbelief = prior * (1.0 - likelyhood) / (1.0 - marginal)

            # Compute belief that the evidence E is real, P(E)(d)
            if antiupdating:
                certainty = torch.max(
                    1.0 - delta * mistrust * (1.0 - marginal),
                    torch.zeros((len(nodes),)),
                )
            else:
                # Consider an agent u and one of its neighbours, v. As beliefs between u and v
                # diverge (delta towards 1), agent u simply ignores the evidence of agent v.
                #
                # If delta becomes 1, uncertainty ~ marginal. In other words, agent u's belief
                # remains unchanged in light of agent v's evidence.
                #
                # The multiplier simply determines how far apart beliefs have to become before
                # agent u begins to ignore the evidence of its neighbour, v (since delta never
                # becomes 1)
                certainty = 1.0 - torch.min(
                    torch.ones((len(nodes),)), delta * mistrust
                ) * (1.0 - marginal)
            # Update posterior belief, in light of uncertainty
            posterior = belief * certainty + misbelief * (1.0 - certainty)
            # Consider next neighbour
            prior = posterior

        # Return posterior beliefs for each neighbour
        return {"belief": posterior}
