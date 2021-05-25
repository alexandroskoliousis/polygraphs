"""
Run PolyGraph simulation(s)
"""

import polygraphs as pg

from polygraphs import cli
from polygraphs import hyperparameters as hp


if __name__ == "__main__":
    # Read command-line arguments
    args = cli.parse()

    if args.configurations:
        # Load PolyGraph hyper-parameters from file(s)
        params = hp.PolyGraphHyperParameters.load(args.configurations)
    else:
        # Use defaults (not recommended)
        params = hp.PolyGraphHyperParameters()

    # Both functions return a `PolyGraphSimulation` object
    if args.explorables:
        _ = pg.explore(params, args.explorables)
    else:
        _ = pg.simulate(params)

    print("Bye.")
