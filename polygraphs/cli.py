"""
PolyGraphs command-line interface
"""

import argparse
import os
import json


class Explorable:  # pylint: disable=too-few-public-methods
    """
    An explorable hyper-parameter (that is also JSON serializable)
    """

    def __init__(self, name, values):
        self.name = name
        self.values = values


class Explorer(argparse.Action):  # pylint: disable=too-few-public-methods
    """
    Implements the Action API, returning a callable to process
    hyper-parameter exploration options.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        # There is a single string argument
        arg = values
        # Argument should be a JSON file or string
        if os.path.isfile(arg):
            with open(arg, "r") as stream:
                cfg = json.load(stream)
        else:
            cfg = json.loads(arg)
        explorables = {key: Explorable(*item.values()) for key, item in cfg.items()}
        setattr(namespace, self.dest, explorables)


def parse(argv=None, required=False, extras=None):
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run PolyGraph simulation(s)")

    parser.add_argument(
        "-f",
        "--configure",
        type=str,
        required=required,
        default=[],
        nargs="*",
        metavar="",
        dest="configurations",
        help="hyper-parameter configuration file(s)",
    )

    parser.add_argument(
        "-e",
        "--explore",
        type=str,
        required=required,
        default=None,
        metavar="",
        dest="explorables",
        help="hyper-parameter exploration option(s)",
        action=Explorer,
    )

    # Parse user-defined command-line arguments
    if extras:
        for arg in extras:
            parser.add_argument(*arg[0], **arg[1])

    return parser.parse_args(argv)
