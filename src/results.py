from model import *
from utils import *


def load_latest_model(args):
    # TODO: Class name
    model = MyModel().to(DEVICE)
    path = get_last_model(get_last_run(args))
    model.load_state_dict(torch.load(path))
    return model


def main():
    args = create_parser().parse_args()

    model = load_latest_model(args)


if __name__ == "__main__":
    main()
