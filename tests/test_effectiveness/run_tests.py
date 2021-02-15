import itertools
import multiprocessing
import os
import pickle
import random
import signal
from sys import exit

from tqdm import tqdm

from genetic.common import lambda_plus_mu
from genetic.common import save_raport
from genetic.plugin_algorithms.crossover import exchange_two_rows_crossover
from genetic.plugin_algorithms.fitness import quadratic_fitness
from genetic.plugin_algorithms.ga_base import SGA
from genetic.plugin_algorithms.initial_pop import uniform_initial_population
from genetic.plugin_algorithms.mutation import shuffle_column_mutation
from tests.test_common import default_termination_condition

save_raport_path = "results"
raport_name = "raport_"
raport_batch_max_size = 100
raport_iter = 0
to_save = []


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
    return_list.append(raport)


def runbatch():
    manager = multiprocessing.Manager()

    path_to_tests = "test_inputs"
    test_names = ["test_cases_4_100.pickle"]  # You can add multiple file names - all their test cases will be loaded

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
    raport_batch = [manager.list()]

    for name in test_names:
        test_path = os.path.join(path_to_tests, name)
        initial_state += pickle.load(open(test_path, "rb"))

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

        if len(raport_batch[0]) >= raport_batch_max_size:
            for p in processes:
                p.join()
            save_raport(save_path=save_raport_path, raport=list(to_save), name=raport_name + str(raport_iter))
            print("Saving batch ", raport_iter)
            raport_iter += 1
            raport_batch[0] = manager.list()

    for p in processes:
        p.join()

    save_raport(save_path=save_raport_path, raport=list(to_save), name=raport_name + str(raport_iter) + ".pickle")


def keyboardInterruptHandler(signal, frame):
    print("Keyboard interrupt save")
    save_raport(save_path=save_raport_path, raport=list(to_save),
                name=raport_name + str(raport_iter) + "_interrupted" + ".pickle")
    exit(-1)


if __name__ == '__main__':
    signal.signal(signal.SIGINT,
                  keyboardInterruptHandler)  # To make it work in PyCharm select 'Emulate terminal in output console'.
    # You will find this option in settings menu in 'Run' section (next to 'Git' section)
    runbatch()
