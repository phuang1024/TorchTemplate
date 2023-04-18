import argparse

from model import *
from utils import *


def load_latest_model(runs_dir):
    # TODO: Class name
    model = MyModel().to(DEVICE)
    path = get_last_model(get_last_run(runs_dir))
    model.load_state_dict(torch.load(path))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Path to data directory")
    parser.add_argument("--runs", default="runs", help="Path to tensorboard runs directory")
    args = parser.parse_args()

    model = load_latest_model(args.runs)


if __name__ == "__main__":
    main()
