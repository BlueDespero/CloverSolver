# CloverSolver

Code of "Evolution Algorithms" end project of Antoni Dąbrowski and Cezary Troska

## Setup

Run setup.sh file before you use CloverSolver

`
source ./setup.sh
`

First time this command is run it will create a virtual enviroment with all the modules required by CloverSolver. Next
time you use CS this script will activate and configure the venv.

## Purpose

This project's purpose is to check if sudokus can be solved with genetic algorithms, if they can compete with
deterministic algorithms and which parameters and operators are best suited for this problem.

## Testing solutions

Main tool for mass testing approaches is called main.py. It allows user to set all the parameters for the genetic
algorithm. Details - all the available options among other things - are described in the script tutorial (available with
--help command)

You can specify more than one option for every parameter. In that case script will test all the possible combinations of
all the parameters provided.

WARNING: This script is using multiprocessing to speed up calculations. That might slow down other processes, as
calculations take up a lot of CPU.

## Project results visualization

We collected results of our research in jupyter notebook called CloverSolver.ipynb. Currently it is only available in
polish, as that's the language we used for presentation for our classes.

## Other Tools

Clover solver comes with tools for sudoku solving with genetic algorithms.

To generate testing examples which can be used with main.py script you can use generate_test_inputs.py. Example:

`python3 tools/generate_test_inputs.py -c 100 -p ./test_cases_4_100.pickle -s 4
`

Which will generate a pickle file _test_cases_4_100.pickle_ at current location with binary
chromosome representation of unfinished sudoku puzzle. There will be 100 examples in this file, and sudokus have side
length 4. This file can be used as input for main.py script.

Results of main.py or inputs generated with generate_test_inputs.py can be checked with lookup_pickle.py.

Example:

`
python3 tools/lookup_pickle.py -p tests/test_effectiveness/results/test_cases_4_100.pickle
`
