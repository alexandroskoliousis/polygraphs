"""
Script for gathering sim data into a csv

Results should be in ~/polygraphs-cache/results, the csv fill will be created
in the same folder

Run `python gather.py -h` for help

"""
import os
import six
from collections import deque
import pandas as pd
import dgl
import networkx as nx
from pathlib import Path
import argparse

# For progress bars
from tqdm.autonotebook import tqdm

# For access to cached results
from fsspec.implementations.local import LocalFileSystem

# Import polygraphs
from polygraphs import hyperparameters as hp
from polygraphs import metadata

# Progress bar(s) #####


# Default `tqdm` progress bar format
_tfmt = None


# Default `tqdm` progress bar arguments
_targs = {
    "bar_format": _tfmt,
    "unit_scale": True,
    "colour": "green",
    "unit": "experiments",
}


def tbar(total, **kwargs):
    """
    Returns a `tqdm` progress bar.
    """
    return tqdm(total=total, **{**_targs, **kwargs})


# File management #####


def ls(filesystem, directory, ext="jpg"):
    """
    Lists all files with extension `ext` in `directory`.
    """
    paths = filesystem.glob(f"{directory}/**/*.{ext}")
    return paths


# File management, specific to PolyGraphs #####


def _isvalid(path):
    # It a .csv file, but it the name data.csv?
    _, tail = os.path.split(path)
    return tail == "data.csv"


def isunique(path):
    """
    Returns `True` if all simulation results, located at `path`, have the same UID.
    """
    assert _isvalid(path), f"Invalid file: {path}"
    data = pd.read_csv(path)
    # Data contains an attribute named "uid"
    assert "uid" in list(data.columns), "Attribute 'uid' not found"
    # Get UID (as a `pandas.Series` object)
    uid = data.uid.unique()
    if len(uid) > 1:
        return False
    # Likely, since there are single-configuration explorations
    head, _ = os.path.split(path)
    _, tail = os.path.split(head)
    return tail == uid[0]


def getparams(path):
    """
    Loads PolyGraph hyper-parameters for experiment.
    """
    head, _ = os.path.split(path)
    config = os.path.join(head, "configuration.json")
    assert os.path.isfile(config), f"File not found: {config}"
    # Load configuration
    params = hp.PolyGraphHyperParameters.load([config])
    return params


# Access to PolyGraph hyper-parameters #####


def getnetworkkind(arg):
    """
    Returns the network kind of the experiment.
    """
    if not isinstance(arg, hp.PolyGraphHyperParameters):
        # Load PolyGraph hyper-parameters
        params = getparams(arg)
    return params.network.kind


def _complete(params):
    """
    Returns hyper-parameters associated with experiments on complete networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
    }


def _wattsstrogatz(params):
    """
    Returns hyper-parameters associated with experiments on Watts-Strogatz networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon", "knn", "prob"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
        "knn": params.network.wattsstrogatz.knn,
        "prob": params.network.wattsstrogatz.probability,
    }


def _random(params):
    """
    Returns hyper-parameters associated with experiments on random networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon", "prob"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
        "prob": params.network.random.probability,
    }


def _barabasialbert(params):
    """
    Returns hyper-parameters associated with experiments on Barabasi-Albert networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon", "attachments"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
        "attachments": params.network.barabasialbert.attachments,
    }


def _cycle(params):
    """
    Returns hyper-parameters associated with experiments on cycle networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
    }


def _star(params):
    """
    Returns hyper-parameters associated with experiments on star networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
    }


def _snap(params):
    """
    Returns hyper-parameters associated with experiments on snap networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon"}

    return {
        "op": params.op,
        "kind": params.network.snap.name,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
    }


def _karate(params):
    """
    Returns hyper-parameters associated with experiments on karate networks.
    """
    if params is None:
        # Return only set of keys
        return {"op", "kind", "size", "trials", "epsilon"}

    return {
        "op": params.op,
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
    }


extractors = {
    "complete": _complete,
    "wattsstrogatz": _wattsstrogatz,
    "barabasialbert": _barabasialbert,
    "random": _random,
    "cycle": _cycle,
    "star": _star,
    "snap": _snap,
    "karate": _karate
}


# Functions to access `pandas.DataFrame` attributes #####


_default_columns = {
    "steps",
    "duration",
    "action",
    "undefined",
    "converged",
    "polarized",
    "uid",
}


def _containsdefaultcolumns(data):
    """
    Returns True is `data` contains (at least) the default columns.
    """
    return set(data.columns).issuperset(_default_columns)


def _getextracolumns(data):
    """
    Returns column names present in `data` other than the default ones.
    """
    return set(data.columns).difference(_default_columns)


# Graph analytics #####


def acc(graph):
    """
    Returns average clustering coefficient.
    """
    graph = nx.DiGraph(graph)
    return nx.algorithms.cluster.average_clustering(graph)


def density(graph):
    """
    Returns density.
    """
    return nx.density(graph)


# ---------------------------------------------------------------------

# Command line arguments
parser = argparse.ArgumentParser(
    description="Process results data from simulations.",
    epilog="Passing no arguments will load results from the"
    " default location of ~/polygraphs-cache/results",
)

parser.add_argument(
    "-f",
    type=str,
    required=False,
    default=[os.getenv("POLYGRAPHS_CACHE") or "~/polygraphs-cache/results"],
    nargs=1,
    metavar="",
    dest="results",
    help="location of results folder",
)

parser.add_argument(
    "-n",
    type=str,
    required=False,
    default=[],
    choices=list(extractors.keys()),
    nargs="*",
    metavar="",
    dest="filter",
    help="networks to filter (seperated by spaces)",
)

parser.add_argument(
    "--add-polarisation",
    default=False,
    action="store_true",
    help="Extract polarisation hyper-parameters",
    dest="polarisation",
)

parser.add_argument(
    "--add-reliability",
    default=False,
    action="store_true",
    help="Extract reliability hyper-parameters",
    dest="reliability",
)

parser.add_argument(
    "--add-statistics",
    default=False,
    action="store_true",
    help="Extract network statistics",
    dest="statistics",
)

args = parser.parse_args()

# ---------------------------------------------------------------------

# Create local filesystem interface
filesystem = LocalFileSystem()


# List all simulation results in result directory
results = ls(filesystem, args.results[0], ext="csv")
results = list(filter(isunique, results))
print(f"{len(results):5d} results in total")

# Per network analysis
networks = {}

for result in results:
    kind = getnetworkkind(result)
    if kind in networks:
        networks[kind].append(result)
    else:
        networks[kind] = [result]

for net, lst in networks.items():

    # Apply network filter, if any
    if len(args.filter) > 0 and net not in args.filter:
        continue

    print(f"{len(lst):5d} results for {net} networks")

    # Create progress bar
    pbar = tbar(len(lst))

    collection = deque()

    # Metadata extractor for current network kind
    extractor = extractors.get(net)
    if extractor is None:
        raise ValueError(f"Unknown network kind: {net}")

    # Default column list
    cols = list(_default_columns.union(extractor(None)))

    if args.polarisation:
        # Append columns related to polarisation experiments
        cols.extend(["mistrust", "antiupdating"])

    if args.reliability:
        # Add column for reliability hyper-parameter
        cols.extend(["reliability"])

    for result in lst:

        # Read results
        data = pd.read_csv(result)

        # Read hyper-parameters
        params = getparams(result)

        # Extract metadata
        meta = extractor(params)

        # Extend metadata with polarisation hyper-parameters
        if args.polarisation:
            meta.update(
                [("mistrust", params.mistrust), ("antiupdating", params.antiupdating)]
            )

        # Extend reliability parameter
        if args.reliability:
            meta.update([("reliability", params.reliability)])

        # Ensure that data contains at least the default columns
        assert _containsdefaultcolumns(data), f"Missing default columns in {result}"

        # Do results contain extra columns?
        extras = _getextracolumns(data)
        if extras:
            # Set is not empty, each key must be in new metadata
            for key in extras:
                assert key in meta, f"Unknown column: {key}"
                # Since column `key` already exists, remove from new metadata
                meta.pop(key)

        # Append metadata as new columns
        for key, value in six.iteritems(meta):
            data[key] = value

        # Reorder columns
        data = data[cols]

        # There should be as many .bin files as rows in data:
        # E.g.:
        #
        #    001.bin
        #    002.bin
        #    ...
        #    100.bin
        #
        # for 100 rows

        # Directory where graphs are stored
        directory, _ = os.path.split(result)

        count = len(data)
        digits = len(str(count))

        # Graph property collections
        # Graph density
        d = []
        # Graph clustering coefficient
        k = []
        # File paths
        f = []

        subbar = tbar(count, position=0, leave=True, colour="red", unit="simulations")

        for idx in range(count):

            filename = f"{{:0{digits}}}.bin".format(idx + 1)
            filepath = os.path.join(directory, filename)
            assert os.path.exists(filepath), f"File not found: {filepath}"
            
            if args.statistics:
                # Load graph from file
                graphs, _ = dgl.load_graphs(filepath)
                graph = graphs[0]

                # Remove self-loops
                graph = dgl.remove_self_loop(graph)

                # convert graph to networkx format
                graphx = dgl.to_networkx(graph)

                # Collect graph statistics
                d.append(density(graphx))
                k.append(acc(graphx))
            else:
                d.append(1.0)
                k.append(1.0)

            # Collect paths to graphs
            f.append(os.path.relpath(filepath, start=args.results[0]))

            subbar.update()

        # Add graph analytics to result
        data["density"] = d
        data["clustering"] = k
        data["filepath"] = f

        # Keep results
        collection.append(metadata.PolyGraphSimulation.fromframe(data))

        pbar.update()

    # Merge all results
    merged = metadata.merge(*collection)
    merged.store(filename=f"{args.results[0]}/{Path(result).parent.parent.name}-{net}.csv")
