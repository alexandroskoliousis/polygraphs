import argparse
import os
import pandas as pd
import six

from polygraphs import hyperparameters as hp
from polygraphs import metadata


def cli(argv=None):
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Infer PolyGraph exploration")

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        default=None,
        metavar="",
        dest="folder",
        help="folder containing simulation results",
    )

    args = parser.parse_args(argv)

    if not os.path.isdir(args.folder):
        raise Exception(f"Invalid folder: {args.folder}")
    return args.folder


def _isfolder(folder, file):
    """
    Returns True is file is a sub-folder.
    """
    return os.path.isdir(os.path.join(folder, file))


def _isresult(folder, file):
    """
    Returns True is sub-folder contains simulation results.
    """
    configuration = os.path.join(folder, file, "configuration.json")
    data = os.path.join(folder, file, "data.csv")
    return os.path.isfile(configuration) and os.path.isfile(data)


def _containsdefaultcolumns(
    folder,
    file,
    columns={
        "steps",
        "duration",
        "action",
        "undefined",
        "converged",
        "polarized",
        "uid",
    },
):
    """
    Returns True is the sub-folder's results contains only default columns.
    """
    df = pd.read_csv(os.path.join(folder, file, "data.csv"))
    return set(df.columns) == columns


def _getnetworkkind(folder, file):
    """
    Returns the network kind of the simulations.
    """
    # Load PolyGraph hyper-parameters from file(s)
    params = hp.PolyGraphHyperParameters.load(
        [os.path.join(folder, file, "configuration.json")]
    )
    return params.network.kind


def _load(folder, subfolder):
    # Load configuration
    params = hp.PolyGraphHyperParameters.load(
        [os.path.join(folder, subfolder, "configuration.json")]
    )
    # Load data
    data = pd.read_csv(os.path.join(folder, subfolder, "data.csv"))
    return params, data


def _complete(params):
    """
    Extracts metadata from configuration for complete networks.
    """
    return {
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
    }


def _wattsstrogatz(params):
    """
    Extracts metadata from configuration for Watts-Strogatz networks.
    """
    return {
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
        "knn": params.network.wattsstrogatz.knn,
        "probability": params.network.wattsstrogatz.probability,
    }


def _random(params):
    """
    Extracts metadata from configuration for random networks.
    """
    return {
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
        "probability": params.network.random.probability,
    }


def _barabasialbert(params):
    """
    Extracts metadata from configuration for Barabasi-Albert networks.
    """
    return {
        "kind": params.network.kind,
        "size": params.network.size,
        "trials": params.trials,
        "epsilon": params.epsilon,
        "attachments": params.network.barabasialbert.attachments,
    }


def _postprocess(folder, subfolders, extractor):
    explorables = set()
    results = []
    for subfolder in subfolders:

        # Load results
        params, data = _load(folder, subfolder)

        # Extract metadata
        meta = extractor(params)

        # Ensure uniqueness of simulation results
        t = tuple(meta.values())
        if t in explorables:
            print(f"warning: duplicate exploration found: {meta} in {subfolder}")
        else:
            print(subfolder, t)
            explorables.add(t)

        # Append metadata as new columns
        for key, value in six.iteritems(meta):
            data[key] = value

        # Keep results
        results.append(metadata.PolyGraphSimulation.fromframe(data))

    # Merge all results
    return metadata.merge(*results)


if __name__ == "__main__":
    # Parse command-line arguments
    folder = cli()
    print(f"Find exploration in {folder}")
    # List all sub-folders and check their correctness
    subfolders = os.listdir(folder)
    print(f"{len(subfolders)} files found")
    networks = []
    for file in subfolders:
        assert _isfolder(folder, file), f"{file} is not a directory"
        assert _isresult(folder, file), f"{file} is not a simulation result"
        assert _containsdefaultcolumns(folder, file), f"{file} contains invalid columns"
        networks.append(_getnetworkkind(folder, file))
    # At this point we assume that all simulation results in the folder
    # are for the same network kind.
    assert len(set(networks)) == 1, f"Multiple networks found: {set(networks)}"
    kind = networks[0]
    print(kind)
    extractorfn = {
        "complete": _complete,
        "wattsstrogatz": _wattsstrogatz,
        "barabasialbert": _barabasialbert,
        "random": _random,
    }[kind]
    result = _postprocess(folder, subfolders, extractorfn)
    result.store(folder)
    print("Bye.")
