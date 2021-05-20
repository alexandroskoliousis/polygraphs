"""
Run PolyGraph simulation(s)
"""

from collections import deque

import polygraphs as pg

from polygraphs import cli
from polygraphs import hyperparameters as hp
from polygraphs import metadata


if __name__ == "__main__":
    # Read command-line arguments
    args = cli.parse()

    if args.configurations:
        # Load PolyGraph hyper-parameters from file(s)
        params = hp.PolyGraphHyperParameters.load(args.configurations)
    else:
        # Use defaults (not recommended)
        params = hp.PolyGraphHyperParameters()

    if args.explorables:
        # Get exploration options
        options = {var.name: var.values for var in args.explorables.values()}
        # Get all possible configurations
        configurations = hp.PolyGraphHyperParameters.expand(params, options)
    else:
        configurations = [params]

    results = deque()
    for config in configurations:
        meta = {key: getattr(config, var.name) for key, var in args.explorables.items()}
        # Run experiment
        result = pg.simulate(config, **meta)
        results.append(result)

    # Merge simulation results in a single data frame for post-processing.
    # If stream is set, write the resulting data frame to a file
    _ = metadata.merge(*results, stream="results.csv")

    print("Bye.")
