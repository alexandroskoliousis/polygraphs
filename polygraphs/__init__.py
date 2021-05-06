"""
PolyGraph module
"""
import os
import uuid
import datetime

import torch

from . import hyperparameters as hparams
from . import metadata
from . import monitors
from . import graphs
from . import ops
from . import logger
from . import timer


log = logger.getlogger()


def _mkdir(results='auto', attempts=10):
    """
    Creates unique directory to store simulation results.
    """
    if results == 'auto':
        head = 'results/simulations/'
        date = datetime.date.today().strftime('%Y-%m-%d')
        for attempt in range(attempts):
            # Generate unique id
            results = os.path.join(head, date, uuid.uuid4().hex)
            # Likely
            if not os.path.isdir(results):
                break
        # Unlikely error
        assert attempt + 1 < attempts, f'Failed to generate unique id after {attempts} attempts'

    # Create result directory, or raise an exception if it already exists
    os.makedirs(results)
    return results


def simulate(params, op=ops.NoOp, **meta):  # pylint: disable=invalid-name
    """
    Runs a PolyGraph simulation multiple times.

    Args:
        params: PolyGraph hyper-parameters
        obj:    PolyGraph op
    """
    assert isinstance(params, hparams.PolyGraphHyperParameters)
    # Collection of simulation results
    results = metadata.PolyGraphSimulation(**meta)
    # Run multiple simulations and collect results
    for idx in range(params.simulation.repeats):
        log.debug('Simulation #{:04d} starts'.format(idx + 1))
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
        result = simulate_(graph,
                           model,
                           steps=params.simulation.steps,
                           mistrust=params.mistrust,
                           hooks=hooks)
        results.add(*result)
        log.info('Sim #{:04d}: '
                 '{:6d} steps '
                 '{:7.2f}s; '
                 'action: {:1s} '
                 'converged: {:<1} '
                 'polarized: {:<1} '.format(idx + 1, *result))
    # End repeats
    # Store simulation results, configuration and other metadata
    # Create destination directory (if not exists)
    params.simulation.results = _mkdir(params.simulation.results)
    # Export hyper-parameters
    params.toJSON(params.simulation.results)
    # Export results
    results.store(params.simulation.results)
    return results


def simulate_(graph, model, steps=1, hooks=None, mistrust=0):
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
        terminated = (converged(graph), polarized(graph, mistrust))
        if any(terminated):
            break
    duration = clock.dt()
    if hooks:
        for hook in hooks:
            hook.conclude(step, graph)
    # Which action did the network decide to take?
    act = consensus(graph)
    # Has the network converged?
    # Is it polarised?
    # How many simulation steps were performed?
    # How long did the simulation take?
    return (step, duration, act,) + terminated


def consensus(graph, lowerupper=0.99):
    """
    Returns action ('A', 'B', or '?') agreed by all agents in the network.
    """
    if converged(graph, lowerupper=lowerupper):
        belief = graph.ndata['belief']
        return 'B' if torch.all(torch.gt(belief, lowerupper)) else 'A'
    return '?'


def converged(graph, upperlower=0.5, lowerupper=0.99):
    """
    Returns `True` if graph has converged.
    """
    tensor = graph.ndata['belief']
    result = torch.all(torch.gt(tensor, lowerupper)) or torch.all(torch.le(tensor, upperlower))
    return result.item()


def polarized(graph, mistrust, upperlower=0.5, lowerupper=0.99):
    """
    Returns `True` if graph is polarized.
    """
    if not mistrust:
        return False
    tensor = graph.ndata['belief']
    # All nodes have decided which action to take (e.g. A or B)
    c = torch.all(torch.gt(tensor, lowerupper) | torch.le(tensor, upperlower))  # pylint: disable=invalid-name
    # There is at least one strong believer
    # that action B is better
    b = torch.any(torch.gt(tensor, lowerupper))  # pylint: disable=invalid-name
    # There is at least one disbeliever
    a = torch.any(torch.le(tensor, upperlower))  # pylint: disable=invalid-name
    if a and b and c:
        delta = torch.min(tensor[torch.gt(tensor, lowerupper)]) - \
                torch.max(tensor[torch.le(tensor, upperlower)])
        return torch.ge(delta * mistrust, 1).item()
    return False
