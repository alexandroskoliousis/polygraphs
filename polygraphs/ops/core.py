"""
PolyGraph basic operator
"""

import abc
from dataclasses import dataclass
import torch

from .. import init


class PolyGraphOp(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base operator from which all other operators are derived.
    """

    @dataclass
    class Record:
        """Defines additional fields for CSV."""
        ...

    def __init__(self, graph, params):
        super().__init__()
        
        # Set device for experimentation
        self._device = params.device

        # The shape of all node attributes
        size = (graph.num_nodes(),)

        # Node beliefs that action B is better
        graph.ndata["beliefs"] = init.init(size, params.init).to(device=self._device)

        # Action B yields Bernoulli payoff of 1 (success) with probability p (= 0.5 + e)
        probs = init.halfs(size) + params.epsilon

        # Number of Bernoulli trials
        count = init.zeros(size) + params.trials

        # Upper and lower bounds of simulation
        self.upperlower = params.upperlower
        self.lowerupper = params.lowerupper

        # Each node gets a private signal that provides information
        # about whether action B is indeed a good action
        self._sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )

        # Store action B's probability of success as a graph node attribute
        graph.ndata["logits"] = self._sampler.logits.to(device=self._device)

    def sample(self):
        """
        Draws a sample from the binomial distribution.
        """
        return self._sampler.sample()

    def trials(self):
        """
        Returns number of Bernoulli trials.
        """
        return self._sampler.total_count

    # Experiment functions

    def experiment(self, graph):
        """
        Simulates an "experiment", where nodes who believe action B is better independently
        observe the payoffs from following action B by sampling a binomial distribution.
        """
        # Consider only nodes who believe action B is better
        mask = graph.ndata["beliefs"] > 0.5
        # Repeat mask
        mask = mask.tile((2, 1))
        # Sample distribution (a node observes only a few successful trials out of all trials)
        result = torch.stack((self.sample(), self.trials())).to(device=self._device)
        # Apply mask
        result = result * mask
        # Store per-node payoffs as a graph node attribute
        graph.ndata["payoffs"] = result.T
        
    def filterfn(self):  # pylint: disable=no-self-use
        """
        Filters edges from graph; Returns a 1-D boolean tensor indicating
        whether the corresponding edge should be considered or not.
        """

        def function(edges):
            return torch.ones((len(edges),), device=self._device).type(torch.bool)

        return function

    @abc.abstractmethod
    def messagefn(self):
        """
        Message function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reducefn(self):  # pylint: disable=unused-argument
        """
        Reduce function.
        """
        raise NotImplementedError

    def applyfn(self):  # pylint: disable=no-self-use
        """
        Update function.
        """
        def function(nodes):
            return nodes.data

        return function

    def forward(self, graph, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Forward function.
        """
        # Generate a local signal (message to be sent)
        self.experiment(graph)
        # Filter valid edges along which messages will be sent
        edges = graph.filter_edges(self.filterfn())
        # Send messages along valid edges; and receive them at
        # edge destination nodes
        graph.send_and_recv(edges, self.messagefn(), self.reducefn(), self.applyfn())
        return graph.ndata["beliefs"]

    # Simulation Functions

    def undefined(self, graph) -> bool:
        """
        Returns `True` if graph beliefs contain undefined values (`nan` or `inf`).
        """
        belief = graph.ndata["beliefs"]
        result = torch.isnan(belief).any() or torch.isinf(belief).any()
        return result.item()

    def converged(self, graph) -> bool:
        """
        Returns `True` if graph has converged.
        """
        tensor = graph.ndata["beliefs"]
        result = torch.all(torch.gt(tensor, self.lowerupper)) or torch.all(
            torch.le(tensor, self.upperlower)
        )
        return result.item()

    def terminated(self, graph) -> bool:
        """
        Terminiation condition to indicate that simulation has finished.
        Checks if we have an invalid simulation or if we met the
        conditions to stop the simulation.
        """
        return any((self.undefined(graph), self.converged(graph)))    

    def consensus(self, graph) -> str:
        """
        Returns action ('A', 'B', or '?') agreed by all agents in the network.
        """
        if self.converged(graph):
            belief = graph.ndata["beliefs"]
            return "B" if torch.all(torch.gt(belief, self.lowerupper)) else "A"
        return "?"    

    def record(self, graph) -> "Record":
        """Returns instance of Record with additional sim results for CSV."""
        return self.Record()    
