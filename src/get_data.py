import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="data")
    parser.add_argument("-j", type=int, default=8)
    args = parser.parse_args()

    # TODO: Actual stuff


if __name__ == "__main__":
    main()
