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

Main tool for mass testing approaches is run_tests.py script located in _tests/test_effectiveness_. You use it by
filling fields responsible for different genetic algorithm parameters.

Here is an example of filled values:

    path_to_tests = "test_inputs"
    test_names = ["test_cases_4_100.pickle"]  

    fitness_function = [quadratic_fitness]
    mutation_operator = [shuffle_column_mutation]
    mutation_rate = [0.2, 0.3, 0.4, 0.5, 0.6]
    crossover_operator = [exchange_two_rows_crossover]
    termination_condition = [default_termination_condition]
    population_merge_function = [lambda_plus_mu]
    iterations = [75, 150]
    population_size = [100, 200]
    number_of_children = [25, 50, 75]

Script will then prepare test cases composed of all the combinations of these parameters and run them for all the
sudokus in the file _test_cases_4_100.pickle_

Results of testing will be placed in location described in save_raport_path variable under name raport_name.

## Project results visualization

We collected results of our research in jupyter notebook called CloverSolver.ipynb. Currently it is only available in
polish, as that's the language we used for presentation for our classes.

## Other Tools

Clover solver comes with tools for sudoku solving with genetic algorithms.

To generate testing examples which can be used with run_tests.py script you can use generate_test_inputs.py. Example:

`python3 utils/generate_test_inputs.py -n test_cases_4_100.pickle -c 100 -p "tests/test_effectiveness/test_inputs/" -s 4
`

Which will generate a pickle file _test_cases_4_100.pickle_ at _tests/test_effectiveness/test_inputs/_ with binary
chromosome representation of unfinished sudoku puzzle. There will be 100 examples in this file, and sudokus have side
length 4. This file can be used as input for run_tests.py script.

Results of run_tests.py or inputs generated with generate_test_inputs.py can be checked with lookup_pickle.py.

Example:

`
python3 utils/lookup_pickle.py -p tests/test_effectiveness/results/test_cases_4_100.pickle
`

This will show the contents of the file pointed to by path _tests/test_effectiveness/results/test_cases_4_100.pickle_
