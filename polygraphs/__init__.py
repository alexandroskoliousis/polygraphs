"""
PolyGraph module
"""
import os
import uuid
import datetime
import random as rnd

import dgl
import torch
import numpy as np

from . import hyperparameters as hparams
from . import metadata
from . import monitors
from . import graphs
from . import ops
from . import logger
from . import timer


log = logger.getlogger()

# Cache data directory for all results
_RESULTCACHE = "~/polygraphs-cache/results"


def _mkdir(results="auto", attempts=10):
    """
    Creates unique directory to store simulation results.
    """
    if results == "auto":
        head = _RESULTCACHE
        date = datetime.date.today().strftime("%Y-%m-%d")
        for attempt in range(attempts):
            # Generate unique id
            results = os.path.join(os.path.expanduser(head), date, uuid.uuid4().hex)
            # Likely
            if not os.path.isdir(results):
                break
        # Unlikely error
        assert (
            attempt + 1 < attempts
        ), f"Failed to generate unique id after {attempts} attempts"

    # Create result directory, or raise an exception if it already exists
    os.makedirs(results)

    return results


def random(seed=0):
    """
    Set random number generator for PolyGraph simulations.
    """
    # Set PyTorch random number generator (RNG) for all devices (CPU and CUDA)
    torch.manual_seed(seed)
    # Set NumPy RNG
    np.random.seed(seed)
    # Set Python RNG
    rnd.seed(seed)
    # Set GDL RNG
    dgl.random.seed(seed)


def simulate(params, op=None, **meta):  # pylint: disable=invalid-name
    """
    Runs a PolyGraph simulation multiple times.

    Args:
        params: PolyGraph hyper-parameters
        obj:    PolyGraph op
    """
    assert isinstance(params, hparams.PolyGraphHyperParameters)
    # Check that either params.op is set, or op is set,
    # but never both (unless they are the same)
    if (params.op is None) == (op is None):
        # Are both None?
        if (params.op is None) and (op is None):
            raise ValueError("Operator not set")
        else:
            raise ValueError("Either params.op or op must be set, but not both")
    if op is None:
        # Get operator by name
        op = ops.getbyname(params.op)
    else:
        # Set operator name in hyper-parameters for future reference
        params.op = op.__name__
    # Collection of simulation results
    results = metadata.PolyGraphSimulation(**meta)
    # Run multiple simulations and collect results
    for idx in range(params.simulation.repeats):
        log.debug("Simulation #{:04d} starts".format(idx + 1))
        # Create a DGL graph with given configuration
        graph = graphs.create(params.network)
        # Create a model with given configuration
        model = op(graph, params)
        # Create a logging hook
        if params.logging.enabled:
            hooks = [monitors.MonitorHook(interval=params.logging.interval)]
        else:
            hooks = None
        # Run simulation
        result = simulate_(
            graph,
            model,
            steps=params.simulation.steps,
            mistrust=params.mistrust,
            lowerupper=params.lowerupper,
            upperlower=params.upperlower,
            hooks=hooks,
        )
        results.add(*result)
        log.info(
            "Sim #{:04d}: "
            "{:6d} steps "
            "{:7.2f}s; "
            "action: {:1s} "
            "undefined: {:<1} "
            "converged: {:<1} "
            "polarized: {:<1} ".format(idx + 1, *result)
        )
    # End repeats
    # Store simulation results, configuration and other metadata
    # Create destination directory (if not exists)
    params.simulation.results = _mkdir(params.simulation.results)
    # Export hyper-parameters
    params.toJSON(params.simulation.results)
    # Export results
    results.store(params.simulation.results)
    return results


def simulate_(
    graph, model, steps=1, hooks=None, mistrust=0.0, lowerupper=0.5, upperlower=0.99
):
    """
    Runs a simulation either for a finite number of steps or until convergence.

    Returns:
        A 4-tuple that consists of (in order):
            a) number of simulation steps
            b) wall-clock time
            c) whether the network has converged or not
            d) whether the network is polarised or not
    """

    def cond(step):
        return step < steps if steps else True

    clock = timer.Timer()
    clock.start()
    step = 0
    terminated = None
    while cond(step):
        step += 1
        # Forward operation on the graph
        _ = model(graph)
        # Monitor progress
        if hooks:
            for hook in hooks:
                hook.mayberun(step, graph)
        # Check termination conditions:
        # - Are beliefs undefined (contain nan or inf)?
        # - Has the network converged?
        # - Is it polarised?
        terminated = (
            undefined(graph),
            converged(graph, upperlower=upperlower, lowerupper=lowerupper),
            polarized(
                graph, upperlower=upperlower, lowerupper=lowerupper, mistrust=mistrust
            ),
        )
        if any(terminated):
            break
    duration = clock.dt()
    if not terminated[0]:
        # Proper exit
        if hooks:
            for hook in hooks:
                hook.conclude(step, graph)
        # Which action did the network decide to take?
        act = consensus(graph, lowerupper=lowerupper)
    else:
        # Beliefs are undefined, and so is the action
        act = "?"

    # Are beliefs undefined (contain nan or inf)?
    # Has the network converged?
    # Is it polarised?
    # How many simulation steps were performed?
    # How long did the simulation take?
    return (
        step,
        duration,
        act,
    ) + terminated


def undefined(graph):
    """
    Returns `True` is graph beliefs contain undefined values (`nan` or `inf`).
    """
    belief = graph.ndata["beliefs"]
    return torch.any(torch.isnan(belief)) or torch.any(torch.isinf(belief))


def consensus(graph, lowerupper=0.99):
    """
    Returns action ('A', 'B', or '?') agreed by all agents in the network.
    """
    if converged(graph, lowerupper=lowerupper):
        belief = graph.ndata["beliefs"]
        return "B" if torch.all(torch.gt(belief, lowerupper)) else "A"
    return "?"


def converged(graph, upperlower=0.5, lowerupper=0.99):
    """
    Returns `True` if graph has converged.
    """
    tensor = graph.ndata["beliefs"]
    result = torch.all(torch.gt(tensor, lowerupper)) or torch.all(
        torch.le(tensor, upperlower)
    )
    return result.item()


def polarized(graph, mistrust=0.0, upperlower=0.5, lowerupper=0.99):
    """
    Returns `True` if graph is polarized.
    """
    # pylint: disable=invalid-name
    if not mistrust:
        return False
    tensor = graph.ndata["beliefs"]
    # All nodes have decided which action to take (e.g. A or B)
    c = torch.all(torch.gt(tensor, lowerupper) | torch.le(tensor, upperlower))
    # There is at least one strong believer
    # that action B is better
    b = torch.any(torch.gt(tensor, lowerupper))
    # There is at least one disbeliever
    a = torch.any(torch.le(tensor, upperlower))
    if a and b and c:
        delta = torch.min(tensor[torch.gt(tensor, lowerupper)]) - torch.max(
            tensor[torch.le(tensor, upperlower)]
        )
        return torch.ge(delta * mistrust, 1).item()
    return False
