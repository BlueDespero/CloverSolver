import argparse
import itertools
import multiprocessing
import pickle
import random

import numpy as np
from tqdm import tqdm

from genetic.common import lambda_plus_mu, lambda_coma_mu
from genetic.common import save_raport
from genetic.plugin_algorithms.crossover import exchange_two_rows_crossover, exchange_two_boxes_crossover, \
    exchange_two_columns_crossover, double_point_crossover, single_point_crossover
from genetic.plugin_algorithms.fitness import quadratic_fitness, linear_fitness
from genetic.plugin_algorithms.ga_base import SGA
from genetic.plugin_algorithms.initial_pop import uniform_initial_population
from genetic.plugin_algorithms.mutation import shuffle_column_mutation, shuffle_row_mutation, shuffle_box_mutation, \
    reverse_bit_mutation
from tests.test_common import default_termination_condition

save_raport_path = "results"
raport_iter = 0
to_save = []


def get_save_name(path):
    return path + str(".{}".format(raport_iter))


def print_raport(raport_batch):
    for i, raport in enumerate(raport_batch):
        print("Raport ", i)
        for r in raport:
            print(str(r) + ":")
            if type(raport[r]) is np.ndarray:
                print(raport[r])
            else:
                print("    " + str(raport[r]))
        print("###########################")


def chromosome_to_sudoku(chromosome):
    side_length = round(chromosome.size ** (1 / 3))
    sudoku = np.zeros(shape=(side_length, side_length))
    for i in range(side_length):
        for j in range(side_length):
            index = (side_length * side_length) * i + j * side_length
            field = chromosome[index:index + side_length]
            found = np.where(field == 1)
            if found[0].size != 1:
                sudoku[i][j] = 0
            else:
                sudoku[i][j] = found[0][0] + 1
    return sudoku


def runsingle(algorithm,
              initial_population_generation,
              fitness_function,
              mutation_operator,
              mutation_rate,
              crossover_operator,
              initial_state,
              termination_condition,
              population_merge_function,
              iterations,
              population_size,
              number_of_children,
              return_list
              ):
    best_result, best_fitness, fitness_record = algorithm(
        initial_population_generation=initial_population_generation,
        fitness_function=fitness_function,
        mutation_operator=mutation_operator,
        mutation_rate=mutation_rate,
        crossover_operator=crossover_operator,
        initial_state=initial_state,
        termination_condition=termination_condition,
        population_merge_function=population_merge_function,
        iterations=iterations,
        population_size=population_size,
        number_of_children=number_of_children,

    )

    raport = dict(
        fitness_function=fitness_function.__name__,
        mutation_operator=mutation_operator.__name__,
        mutation_rate=mutation_rate,
        crossover_operator=crossover_operator.__name__,
        population_merge_function=population_merge_function.__name__,
        iterations=iterations,
        population_size=population_size,
        number_of_children=number_of_children,
        initial_state=chromosome_to_sudoku(initial_state),
        best_fitness=best_fitness,
        best_result=chromosome_to_sudoku(best_result),
    )
    return_list.append(raport)


def runbatch(
        fitness_function,
        mutation_operator,
        mutation_rate,
        crossover_operator,
        population_merge_function,
        iterations,
        population_size,
        number_of_children,
        input_paths,
        output_path,
        checkpoint,
        if_print
):
    manager = multiprocessing.Manager()
    global save_raport_path
    save_raport_path = output_path

    algorithm = [SGA]
    initial_population_generation = [uniform_initial_population]
    termination_condition = [default_termination_condition]
    raport_batch = [manager.list()]

    initial_state = []
    for path in input_paths:
        initial_state += pickle.load(open(path, "rb"))

    test_list = list(itertools.product(algorithm, initial_population_generation, fitness_function, mutation_operator,
                                       mutation_rate, crossover_operator, initial_state, termination_condition,
                                       population_merge_function, iterations, population_size, number_of_children,
                                       raport_batch))
    random.shuffle(test_list)

    processes = []
    max_processes_number = 8
    global raport_iter

    for test in tqdm(test_list):
        global to_save
        to_save = raport_batch[0]

        p = multiprocessing.Process(target=runsingle, args=test)

        while p not in processes:
            processes = [x for x in processes if x.is_alive()]
            if len(processes) < max_processes_number:
                p.start()
                processes.append(p)

        if checkpoint != 0 and len(raport_batch[0]) >= checkpoint:
            for p in processes:
                p.join()
            save_raport(save_path=get_save_name(save_raport_path), raport=list(to_save))
            if if_print:
                print_raport(list(to_save))
            print("Saving batch ", raport_iter)
            raport_iter += 1
            raport_batch[0] = manager.list()

    for p in processes:
        p.join()

    save_raport(save_path=get_save_name(save_raport_path), raport=list(to_save))
    if if_print:
        print_raport(list(to_save))


translate_fitness = {
    "lin": linear_fitness,
    "qua": quadratic_fitness
}
translate_crossover = {
    "sp": single_point_crossover,
    "dp": double_point_crossover,
    "etb": exchange_two_boxes_crossover,
    "etr": exchange_two_rows_crossover,
    "etc": exchange_two_columns_crossover
}
translate_lambda = {
    "plus": lambda_plus_mu,
    "coma": lambda_coma_mu
}
translate_mutation = {
    "rb": reverse_bit_mutation,
    "srm": shuffle_row_mutation,
    "scm": shuffle_column_mutation,
    "sbm": shuffle_box_mutation
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', '--input-path', required=True, nargs='+',
                        help="Path to sudoku representations, algorithm inputs.")
    parser.add_argument('-r', '--result-path', required=True, help="Result save path")
    parser.add_argument('--iterations', type=int, default=[100], nargs='*',
                        help="Max amount of iterations. Default: 100")
    parser.add_argument('--population-size', type=int, default=[100], nargs='*',
                        help="Population size. Default: 100")
    parser.add_argument('--children-amount', type=int, default=[50], nargs='*',
                        help="Amount of children produced in every iteration. Default: 50")
    parser.add_argument('--mutation-rate', type=float, default=[0.2], nargs='*',
                        help="Mutation rate. Default: 0.2")
    parser.add_argument('--mutation', default=["rb"], nargs='*', choices=["rb", "srm", "scm", "sbm"],
                        help="Mutation operator.\n"
                             "rb - reverse bit (default)\n"
                             "srm - shuffle rows\n"
                             "scm - shuffle columns\n"
                             "sbm - shuffle boxes")
    parser.add_argument('--crossover', default=["sp"], nargs='*', choices=["sp", "dp", "etb", "etr", "etc"],
                        help="Crossover operator.\n"
                             "sp - single point exchange (default)\n"
                             "dp - double point exchange\n"
                             "etb - exchange boxes\n"
                             "etr - exchange rows\n"
                             "etc - exchange columns")
    parser.add_argument('--lambda-function', default=["plus"], nargs='*', choices=["plus", "coma"],
                        help="Crossover operator.\n"
                             "coma - only children pass to the next generation\n"
                             "plus - children and parents create next generation (default)")
    parser.add_argument('--fitness', default=["lin"], nargs='*', choices=["lin", "qua"],
                        help="Fitness function.\n"
                             "lin - linear fitness (default)\n"
                             "qua - quadratic_fitness")
    parser.add_argument('--print-results', action="store_true",
                        help="Set if you want to print results after script is finished.")
    parser.add_argument('--checkpoints', type=int, default=0,
                        help="Set if you want to save your progress. "
                             "Results will be saved every x tests, where x is the valueset here")

    args = parser.parse_args()

    runbatch(
        fitness_function=[translate_fitness[f] for f in args.fitness],
        mutation_operator=[translate_mutation[m] for m in args.mutation],
        mutation_rate=args.mutation_rate,
        crossover_operator=[translate_crossover[c] for c in args.crossover],
        population_merge_function=[translate_lambda[l] for l in args.lambda_function],
        iterations=args.iterations,
        population_size=args.population_size,
        number_of_children=args.children_amount,
        input_paths=args.input_path,
        output_path=args.result_path,
        if_print=args.print_results,
        checkpoint=args.checkpoints
    )
