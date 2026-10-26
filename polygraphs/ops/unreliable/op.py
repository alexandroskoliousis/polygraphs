import torch

from . import update
from . import samplers
from ..core import PolyGraphOp
from ...logger import getlogger

log = getlogger()

# Original class names registry (updating strategy, sampling strategy)
UNRELIABLE_ALIASES = {
    "UnreliableNetworkIdealOp": ("ideal", None),
    "UnreliableNetworkBasicGullibleUniformOp": ("gullible", "uniform"),
    "UnreliableNetworkBasicGullibleBinomialOp": ("gullible", "binomial"),
    "UnreliableNetworkBasicGullibleNegativeEpsOp": ("gullible", "negative_eps"),
    "UnreliableNetworkBasicAlignedUniformOp": ("aligned", "uniform"),
    "UnreliableNetworkBasicAlignedBinomialOp": ("aligned", "binomial"),
    "UnreliableNetworkBasicAlignedNegativeEpsOp": ("aligned", "negative_eps"),
    "UnreliableNetworkBasicUnalignedUniformOp": ("unaligned", "uniform"),
    "UnreliableNetworkModifiedAlignedUniformOp": ("modified", "uniform"),
    "UnreliableNetworkModifiedAlignedBinomialOp": ("modified", "binomial"),
    "UnreliableNetworkModifiedAlignedNegativeEpsOp": ("modified", "negative_eps"),
}


class UnreliableOp(PolyGraphOp):
    """
    Baseclass for Unreliable Ops that draw from an unreliable sampler

    Unreliable networks, Part 1

    There are two types of nodes, reliable and unreliable ones.
    """

    def __init__(self, graph, params):
        super().__init__(graph, params)
        # The shape of all node attributes
        self._size = (graph.num_nodes(),)

        # Resolve updating and sampling strategies from preconfigured registry
        # using the op name
        updating, sampler = UNRELIABLE_ALIASES.get(params.op, (None, None))

        # Try and get unreliable sampling strategy from params
        self._unreliable_sampler = None
        sampler_strategy = params.sampler or sampler
        log.info(f"Sampler: {sampler_strategy}")

        # Create sampler if we need it
        if sampler_strategy:
            self._unreliable_sampler = samplers.create(
                sampler_strategy, self._size, params
            )

        # Store the reliability parameter
        self._network_reliability = params.reliability

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

        # Update strategies contain the graph functions
        # Try and get the update strategy from params
        self._update = None
        update_strategy = params.updating or updating
        log.info(f"Update Strategy: {update_strategy}")

        # Create update strategy after the graph has been populated with
        # unreliable nodes
        self._update = update.create(update_strategy, self, graph, params)

    @property
    def size(self):
        """
        The shape of all node attributes.
        """
        return self._size

    @property
    def device(self):
        """
        The device that node attributes live on.
        """
        return self._device

    @property
    def reliability(self):
        """
        Per-node reliability mask (1 = reliable), on the CPU.
        """
        return self._reliability

    # Sampling

    def sample(self):
        """
        Reliable nodes sample the reliable distribution; unreliable nodes
        the unreliable one.
        """
        # pylint: disable=invalid-name
        # Sample reliable distribution
        b = self._sampler.sample()

        # Return if we do not have an unreliable sampler
        if self._unreliable_sampler is None:
            return b

        # Sample unreliable distribution
        u = self._unreliable_sampler.sample()
        # Combine samples
        return b * self._reliability + u * (1 - self._reliability)

    # Graph hooks, delegated to the strategy

    def filterfn(self):
        return self._update.filterfn()

    def messagefn(self):
        return self._update.messagefn()

    def reducefn(self):
        return self._update.reducefn()

    def applyfn(self):
        return self._update.applyfn()
