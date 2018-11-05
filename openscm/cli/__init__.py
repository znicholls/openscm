import argparse
import pandas as pd
import yaml

from .. import __version__

parser = argparse.ArgumentParser(description="OpenSCM")

parser.add_argument("model", type=str, help="Set SCM to use")
parser.add_argument("scenario", type=str, help="Scenario file to load")
parser.add_argument(
    "parameters", type=str, help="Parameter file to load", nargs="?", default=None
)
parser.add_argument(
    "--version", action="version", version="%(prog)s {}".format(__version__)
)


def load_scenario(filepath):
    return pd.read_csv(filepath, index_col=0)


def load_parameters(filepath=None):
    if filepath:
        return yaml.load(open(filepath, "r"))
    return {}


def main():
    model = args.model
    scenario = load_scenario(args.scenario)
    params = load_parameters(args.parameters)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
