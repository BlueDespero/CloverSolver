import argparse
import os
import pickle

from genetic.common import classic_representation_sudoku_into_full_chromosome
from utils.sudoku_transcription import sudoku_generator


def generate_inputs(name, path, num_cases, size=9, unique=True):
    # TODO make name not requered, but generic and generated based on the path directory contents

    generated_cases = []
    while len(generated_cases) < num_cases:
        case = sudoku_generator(size)
        case = classic_representation_sudoku_into_full_chromosome(case)
        # TODO add unique filtering
        generated_cases.append(case)

    pickle.dump(generated_cases, open(os.path.join(path, name), "wb"))


if __name__ == '__main__':
    # Parse params
    parser = argparse.ArgumentParser(
        epilog="Example use: python3.6 generate_test_inputs.py -n test_cases_9_3.pickle -c 3 -p 'tests/test_effectiveness/test_inputs/'")
    parser.add_argument('-n', '--name', required=True, help="Name of the file where test cases will be saved")
    parser.add_argument('-c', '--count', required=True, type=int, help="How many cases should be generated")
    parser.add_argument('-p', '--path', required=True,
                        help="Path to the directory where result should be saved(absolute or relative to generate_test_inputs.py file)")
    parser.add_argument('-s', '--size', type=int, default=9, help="Generated sudoku side size")
    parser.add_argument('-u', '--unique', type=bool, default=True, help="Should we only saving unique tests?")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("Directory specified in path variable doesn't exist!")
        parser.print_help()
        exit(1)

    if os.path.exists(os.path.join(args.path, args.name)):
        print("File with this name already exist!")
        exit(1)

    generate_inputs(args.name, args.path, args.count, args.size, args.unique)
