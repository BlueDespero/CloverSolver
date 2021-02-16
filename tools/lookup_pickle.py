import argparse
import pickle


def lookup(path):
    print(pickle.load(open(path, "rb")))


if __name__ == '__main__':
    # Parse params
    parser = argparse.ArgumentParser(
        description="Script to easily lookup .pickle files. Results and inputs are saved in this format.")
    parser.add_argument('-p', '--path', required=True,
                        help="Path to the file you want to lookup.")

    args = parser.parse_args()

    lookup(args.path)
