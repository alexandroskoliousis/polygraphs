"""
Generates job array configurations.
"""
import os
import datetime

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

    # Create directories for Slurm output
    date = datetime.date.today().strftime("%Y-%m-%d")
    slurm_path = os.path.join(os.path.expanduser( '~/polygraphs-cache/slurm' ), date)

    if not(os.path.exists(slurm_path) and os.path.isdir(slurm_path)):
        os.makedirs(slurm_path)


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
        config.toJSON(directory, filename=f"{args.array}-{jid}.json", exists_ok=True)

    print(f"{jid} configurations generated")

    # Generate run-array.script (experimental feature)
    f = open("run-array.script", "w")

    f.write("#!/bin/bash\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --ntasks=1\n")
    f.write("#SBATCH --cpus-per-task=8\n")
    f.write("#SBATCH --mem=32GB\n")
    f.write("#SBATCH --time=24:00:00\n")
    f.write("#SBATCH --export=ALL\n")
    f.write("#SBATCH --array=1-{}\n".format(jid))
    f.write("#SBATCH --job-name={}\n".format(args.array))
    f.write("#SBATCH --output={}/{}-slurm-%A_%a.out\n".format(slurm_path, args.array))
    f.write("#SBATCH --error={}/{}-slurm-%A_%a.err\n".format(slurm_path, args.array))
    f.write(
        "python run.py -f {}/{}-${{SLURM_ARRAY_TASK_ID}}.json\n".format(
            directory, args.array
        )
    )
    f.close()
    print("Bye.")
