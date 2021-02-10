import argparse
import pickle


def lookup(path):
    print(pickle.load(open(path, "rb")))


if __name__ == '__main__':
    # Parse params
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True,
                        help="Path to the file you want to lookup (absolute or relative to lookup_pickle.py file)")

    args = parser.parse_args()

    lookup(args.path)
