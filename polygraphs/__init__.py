"""
PolyGraph module
"""
import os
import uuid
import datetime
import dataclasses
import random as rnd
import collections
import json

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

# Removed (exporting PolyGraph to JPEG is deprecated for now)
# from . import visualisations as viz


log = logger.getlogger()

# Cache data directory for all results
_RESULTCACHE = os.getenv("POLYGRAPHS_CACHE") or "~/polygraphs-cache/results"


def _mkdir(directory="auto", attempts=10):
    """
    Creates unique directory to store simulation results.
    """
    # Unique exploration or simulation id
    uid = None
    if not directory:
        # Do nothing
        return uid, directory
    head, tail = os.path.split(directory)
    if tail == "auto":
        # If the parent is set, do not create subdirectory for today
        date = datetime.date.today().strftime("%Y-%m-%d") if not head else ""
        head = head or _RESULTCACHE
        for attempt in range(attempts):
            # Generate unique id string
            uid = uuid.uuid4().hex
            # Generate unique directory
            directory = os.path.join(os.path.expanduser(head), date, uid)
            # Likely
            if not os.path.isdir(directory):
                break
        # Unlikely error
        assert (
            attempt + 1 < attempts
        ), f"Failed to generate unique id after {attempts} attempts"
    else:
        # User-defined directory must not exist
        assert not os.path.isdir(directory), "Results directory already exists"
    # Create result directory, or raise an exception if it already exists
    os.makedirs(directory)
    return uid, directory


def _storeresult(params, result):
    """
    Helper function for storing simulation results
    """
    if params.simulation.results is None:
        return
    # Ensure destination directory exists
    assert os.path.isdir(params.simulation.results)
    # Export results
    result.store(params.simulation.results)


def _storeparams(params, explorables=None):
    """
    Helper function for storing configuration parameters
    """
    if params.simulation.results is None:
        return
    # Ensure destination directory exists
    assert os.path.isdir(params.simulation.results)
    # Export hyper-parameters
    params.toJSON(params.simulation.results, filename="configuration.json")
    # Export explorables
    if explorables:
        fname = os.path.join(params.simulation.results, "exploration.json")
        with open(fname, "w") as fstream:
            json.dump(explorables, fstream, default=lambda x: x.__dict__, indent=4)


def _storegraph(params, graph, prefix):
    """
    Helper function for storing simulated graph
    """
    if not params.simulation.results:
        return
    # Ensure destination directory exists
    assert os.path.isdir(params.simulation.results)
    # Export DGL graph in binary format
    fname = os.path.join(params.simulation.results, f"{prefix}.bin")
    dgl.save_graphs(fname, [graph])
    # Export DGL graph as JPEG
    #
    # Important note:
    #
    #    viz.draw is not well suited for drawing a graph.
    #    Remove for now.
    #
    # fname = os.path.join(params.simulation.results, f"{prefix}.jpg")
    # _, _ = viz.draw(graph, layout="circular", fname=fname)


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


def explore(params, explorables):
    """
    Explores multiple PolyGraph configurations.
    """
    # Get exploration options
    options = {var.name: var.values for var in explorables.values()}
    # Get all possible configurations
    configurations = hparams.PolyGraphHyperParameters.expand(params, options)
    # There must be at least two configurations
    assert len(configurations) > 1
    # Exploration results ought to be stored
    assert params.simulation.results
    # Create parent directory to store results
    _, params.simulation.results = _mkdir(params.simulation.results)
    # Store configuration parameters
    _storeparams(params, explorables=explorables)
    # Intermediate result collection
    collection = collections.deque()
    # Run all
    for config in configurations:
        # Store intermediate results?
        config.simulation.results = os.path.join(
            params.simulation.results, "explorations/auto"
        )
        # Set metadata columns
        meta = {key: config.getattr(var.name) for key, var in explorables.items()}
        # Metadata columns to string
        log.info(
            "Explore {} ({} simulations)".format(
                ", ".join([f"{k} = {v}" for k, v in meta.items()]),
                config.simulation.repeats,
            )
        )
        # Run experiment
        result = simulate(config, **meta)
        collection.append(result)

    # Merge simulation results
    results = metadata.merge(*collection)
    # Store simulation results
    _storeresult(params, results)
    return results


def _resolve_op(params, op):
    """
    Returns the PolyGraph operator.
    """
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
    return op


def _record_type(op):
    """
    Returns the dataclass an operator uses to record its results.
    Extendes the Record dataclass from the op with the default
    SimulationRecord dataclass if provided.
    """
    op_record = getattr(op, "Record")
    # Combine datacalass if Record provided by op
    if op_record:
        return dataclasses.make_dataclass(
            "SimulationResult",
            [],
            bases=(op_record, metadata.SimulationRecord),
            frozen=True)
    return metadata.SimulationRecord


@torch.no_grad()
def simulate(params, op=None, **meta):  # pylint: disable=invalid-name
    """
    Runs a PolyGraph simulation multiple times.

    Args:
        params: PolyGraph hyper-parameters
        obj:    PolyGraph op
    """
    if not isinstance(params, hparams.PolyGraphHyperParameters):
        raise TypeError("params must be PolyGraphHyperParameters")
    op = _resolve_op(params, op)
    # Create result directory
    uid, params.simulation.results = _mkdir(params.simulation.results)
    # Store configuration parameters
    _storeparams(params)

    # Resolve record type for results from op
    record_type = _record_type(op)
    results = metadata.PolyGraphSimulation(record_type, uid=uid, **meta)
    repeats = params.simulation.repeats

    # Run multiple simulations and collect results
    for idx in range(params.simulation.repeats):
        log.debug("Simulation #{:04d} starts".format(idx + 1))
        # Create a DGL graph with given configuration
        graph = graphs.create(params.network)
        # Set device for graph
        graph = graph.to(device=params.device)
        # Create a model with given configuration
        model = op(graph, params)
        # Export graph (beliefs are initialised)
        prefix = f"{(idx + 1):0{len(str(params.simulation.repeats))}d}"
        _storegraph(params, graph, prefix)
        # Set model in evaluation mode
        model.eval()
        # Create hooks
        hooks = []
        if params.logging.enabled:
            # Create logging hook
            hooks += [monitors.MonitorHook(interval=params.logging.interval)]
        if params.snapshots.enabled:
            # Create snaphot hook
            hooks += [
                monitors.SnapshotHook(
                    interval=params.snapshots.interval,
                    messages=params.snapshots.messages,
                    location=params.simulation.results,
                    filename=f"{prefix}.hd5",
                )
            ]
        # Run simulation
        result = simulate_(
            graph,
            model,
            record_type,
            steps=params.simulation.steps,
            hooks=hooks,
        )
        results.add(result)
        log.info(
            f"Sim #{idx + 1:04d}: {result.steps:6d} steps "
            f"{result.duration:7.2f}s; action: {result.action:1s} "
            f"undefined: {result.undefined:<1} converged: {result.converged:<1} "
        )
    # End repeats
    # Store simulation results  
    _storeresult(params, results)
    return results


def simulate_(graph, model, record=metadata.SimulationRecord, steps=1, hooks=None):
    """
    Runs a simulation either for a finite number of steps or until convergence.

    Returns:
        A 4-tuple that consists of (in order):
            a) number of simulation steps
            b) wall-clock time
            c) whether the network has converged or not
            d) any other outcome_fields defined by the op
    """

    def cond(step):
        return step < steps if steps else True

    clock = timer.Timer()
    clock.start()
    step = 0
    action = "?"
    
    while cond(step):
        step += 1
        # Forward operation on the graph
        _ = model(graph)
        # Monitor progress
        if hooks:
            for hook in hooks:
                hook.mayberun(step, graph)
        # Check termination conditions
        if model.terminated(graph):
            break

    undefined = model.undefined(graph)
    converged = model.converged(graph)
    
    duration = clock.dt()
    if not undefined:
        # Proper exit
        if hooks:
            for hook in hooks:
                hook.conclude(step, graph)
        # Which action did the network decide to take?
        action = model.consensus(graph)

    # Get custom Record fields from op.record
    additional_fields = dataclasses.asdict(model.record(graph))

    return record(
        steps=step,
        duration=duration,
        action=action,
        undefined=undefined,
        converged=converged,
        **additional_fields,
    )
