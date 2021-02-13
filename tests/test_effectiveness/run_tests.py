import itertools
import os
import pickle
import numpy as np

from tqdm import tqdm

from genetic.common import lambda_plus_mu, lambda_coma_mu
from genetic.common import save_raport
from genetic.plugin_algorithms.crossover import exchange_two_rows_crossover, exchange_two_columns_crossover, \
    exchange_two_boxes_crossover, single_point_crossover, double_point_crossover
from genetic.plugin_algorithms.fitness import quadratic_fitness, linear_fitness
from genetic.plugin_algorithms.ga_base import SGA
from genetic.plugin_algorithms.initial_pop import uniform_initial_population
from genetic.plugin_algorithms.mutation import shuffle_column_mutation, shuffle_box_mutation, shuffle_row_mutation, \
    reverse_bit_mutation
from tests.test_common import default_termination_condition

save_raport_path = "results"
raport_name = "raport_"
raport_batch_max_size = 100
raport_iter = 0
raport_batch = []


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
              number_of_children
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
        algorithm=algorithm.__name__,
        initial_population_generation=initial_population_generation.__name__,
        fitness_function=fitness_function.__name__,
        mutation_operator=mutation_operator.__name__,
        mutation_rate=mutation_rate,
        crossover_operator=crossover_operator.__name__,
        initial_state=initial_state,
        termination_condition=termination_condition.__name__,
        population_merge_function=population_merge_function.__name__,
        iterations=iterations,
        population_size=population_size,
        number_of_children=number_of_children,
        best_result=best_result,
        best_fitness=best_fitness,
        fitness_record=fitness_record
    )

    return raport


def runbatch():
    path_to_tests = r"C:\Users\user\PycharmProjects\CloverSolver\tests\test_effectiveness"
    test_names = ["test_inputs30_4x4"]  # You can add multiple file names - all their test cases will be loaded

    algorithm = [SGA]
    initial_population_generation = [uniform_initial_population]
    fitness_function = [quadratic_fitness]
    mutation_operator = [shuffle_column_mutation]
    mutation_rate = [0.2, 0.3, 0.4, 0.5, 0.6]
    crossover_operator = [exchange_two_rows_crossover]
    termination_condition = [default_termination_condition]
    population_merge_function = [lambda_plus_mu]
    iterations = [75, 150]
    population_size = [100, 200]
    number_of_children = [25, 50, 75]

    initial_state = []

    for name in test_names:
        test_path = os.path.join(path_to_tests, name)
        initial_state += pickle.load(open(test_path, "rb"))

    test_list = itertools.product(algorithm, initial_population_generation, fitness_function, mutation_operator,
                                  mutation_rate, crossover_operator, initial_state, termination_condition,
                                  population_merge_function, iterations, population_size, number_of_children)

    save_raport_path = r"C:\Users\user\PycharmProjects\CloverSolver\tests\test_effectiveness"
    raport_name = "raport_"
    raport_batch_max_size = 500
    raport_iter = 0
    raport_batch = []
    for test in tqdm(list(test_list)):
        global raport_iter
        global raport_batch

        raport = runsingle(*test)
        raport_batch.append(raport)
        if len(raport_batch) >= raport_batch_max_size:
            save_raport(save_path=save_raport_path, raport=raport_batch, name=raport_name + str(raport_iter) + ".pickle")
            print("Saving batch ", raport_iter)
            raport_iter += 1
            raport_batch = []

    save_raport(save_path=save_raport_path, raport=raport_batch, name=raport_name + str(raport_iter) + ".pickle")


if __name__ == '__main__':
    runbatch()
