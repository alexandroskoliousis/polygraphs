#!/usr/bin/env python3
"""
Run PolyGraph simulation(s)
"""

import polygraphs as pg

from polygraphs import cli
from polygraphs import hyperparameters as hp

def run():
    # Read command-line arguments
    args = cli.parse()

    if args.configurations:
        # Load PolyGraph hyper-parameters from file(s)
        params = hp.PolyGraphHyperParameters.load(args.configurations)
    else:
        print("PolyGraphs requires a configuration file to run")
        return

    # Set random seed
    if params.seed is not None:
        # Ensure it is an integer
        assert isinstance(params.seed, int) and not params.seed < 0, "Invalid seed"
        if params.seed > 0:
            pg.random(params.seed)

    # Both functions return a `PolyGraphSimulation` object
    if args.explorables:
        _ = pg.explore(params, args.explorables)
    else:
        _ = pg.simulate(params)

    print("Bye.")

if __name__ == "__main__":
    run()
