import argparse
import os
import pickle

from genetic.common import classic_representation_sudoku_into_full_chromosome
from utils.sudoku_transcription import sudoku_generator


def generate_inputs(path, num_cases, size=9, sudoku_generating_method=sudoku_generator):
    # TODO make name not requered, but generic and generated based on the path directory contents

    generated_cases = []
    while len(generated_cases) < num_cases:
        case = sudoku_generating_method(size)
        case = classic_representation_sudoku_into_full_chromosome(case)
        # TODO add unique filtering
        generated_cases.append(case)

    pickle.dump(generated_cases, open(path, "wb"))


if __name__ == '__main__':
    # Parse params
    parser = argparse.ArgumentParser(
        description="Script to mass produce test examples for main.py",
        epilog="Example use: python3.6 generate_test_inputs.py -c 3 -p './test_cases_9_3.pickle")
    parser.add_argument('-c', '--count', required=True, type=int, help="How many cases should be generated")
    parser.add_argument('-p', '--path', required=True, help="Specify where results should be saved.")
    parser.add_argument('-s', '--size', type=int, default=9, help="Generated sudoku side size")

    args = parser.parse_args()

    generate_inputs(args.path, args.count, args.size)
