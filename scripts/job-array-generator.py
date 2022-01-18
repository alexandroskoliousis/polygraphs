"""
Generates job array configurations.
"""
import os

from polygraphs import cli
from polygraphs import hyperparameters as hp


if __name__ == "__main__":

    extras = [
        (
            ["-a", "--array"],
            {
                "type": str,
                "required": False,
                "default": "test",
                "dest": "array",
                "metavar": "",
                "help": "array name",
            },
        ),
        (
            ["-s", "--store"],
            {
                "type": str,
                "required": False,
                "default": "configs/arrays",
                "dest": "store",
                "metavar": "",
                "help": "directory to store configurations",
            },
        ),
    ]

    # Read command-line arguments
    args = cli.parse(required=True, extras=extras)

    # Load PolyGraph hyper-parameters from file(s)
    params = hp.PolyGraphHyperParameters.load(args.configurations)

    # Get exploration options
    options = {var.name: var.values for var in args.explorables.values()}

    # Get all possible configurations
    configurations = hp.PolyGraphHyperParameters.expand(params, options)

    # Set destination directory
    directory = os.path.join(args.store, args.array)

    for idx, config in enumerate(configurations):
        # Job id
        jid = idx + 1
        # Metadata columns to string
        meta = {key: config.getattr(var.name) for key, var in args.explorables.items()}
        print(
            "Job {:03d}: Explore {}".format(
                jid, ", ".join([f"{k} = {v}" for k, v in meta.items()])
            )
        )
        config.toJSON(directory, filename=f"{args.array}-{jid}.json")

    print(f"{jid} configurations generated")
    print("Bye.")
