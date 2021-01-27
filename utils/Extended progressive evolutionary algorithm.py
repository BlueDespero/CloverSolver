import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

from utils.Sudoku_transcription import sudoku_generator, sudoku_matrix_representation, print_sudoku
from utils.algorithm_x import algorithm_x_first_solution, remove_intersections


def observe_number_of_possible_rows(numbers_of_rows, transcription_matrix):
    queue = numbers_of_rows.copy()
    np.random.shuffle(queue)

    t_matrix = transcription_matrix.copy()

    number_of_possibilities = []

    while queue.size > 0:
        size = t_matrix.shape[0]
        number_of_possibilities.append(size)
        chromosome = ids_to_binary_list(queue, size)

        t_matrix, chromosome = remove_intersections(t_matrix, chromosome, t_matrix, queue[0])

        queue = binary_list_to_ids(chromosome)

    return np.array(number_of_possibilities)


def cumulate_data(size_of_sample, size_of_sudoku):
    coordinates, transcription_matrix = sudoku_matrix_representation(np.zeros([size_of_sudoku, size_of_sudoku]))
    solution = np.array(algorithm_x_first_solution(transcription_matrix))
    temp = observe_number_of_possible_rows(solution, transcription_matrix)
    initial = temp[0]
    current_stage = np.empty(temp.shape[0])
    for k, value in enumerate(temp):
        current_stage[k] = value / (initial - k)
    number_of_possibilities = current_stage
    for i in tqdm(range(size_of_sample)):
        solution = np.array(algorithm_x_first_solution(transcription_matrix))
        temp = observe_number_of_possible_rows(solution, transcription_matrix)
        initial = temp[0]
        data = np.empty(temp.shape[0])
        for k, value in enumerate(temp):
            data[k] = value / (initial - k)

        current_stage += data

        number_of_possibilities = np.vstack([number_of_possibilities, current_stage / (i + 2)])

    return number_of_possibilities


def fitness_function(chromosome, evaluation_matrix,
                     model=lambda once, more: more - once):
    # input:
    # chromosome = 1-D array(n)
    # evaluation_matrix = np.array(n,m) of zeros and ones
    # model = lambda int, int: int; also float type numbers will work well
    # model - describes dependency of coverd elements and coverd more than once on fitness function
    # for default model minimum of fitness function is always zero

    filter = chromosome.astype(bool)
    solution = evaluation_matrix[filter]
    flattened = solution.sum(axis=0)
    covered_more_than_once = np.sum(flattened > 1)
    covered_once = np.sum(flattened == 1)

    # output:
    # int/float
    return model(covered_once, covered_more_than_once) + flattened.shape[0]


def mutation_one(chromosome):
    # input:
    # chromosome = 1-D array

    index_of_chosen_one = np.random.choice(np.argwhere(chromosome == 1).flatten())
    index_of_chosen_zero = np.random.choice(np.argwhere(chromosome == 0).flatten())

    # swap
    new = chromosome.copy()
    new[index_of_chosen_one] = 0
    new[index_of_chosen_zero] = 1

    # output:
    # new = 1-D array
    return new


def limit_iteration(n, k, number_of_possibilities,P):
    index = int((k / n) * number_of_possibilities.shape[0])
    if index == 0:
        return 1*P
    return int(1 / np.log10(1 / (1 - number_of_possibilities[index]))+1)*P


def Extended_progressive_evolutionary_algorithm(evaluation_matrix, size_of_population,
                                                size_of_sudoku, max_iter=None,P=4):
    # initialization of first generation
    population = np.zeros([size_of_population,evaluation_matrix.shape[0]])
    for i in range(size_of_population):
        population[i,np.random.randint(0,evaluation_matrix.shape[0],1)]=1

    number_of_possibilities = cumulate_data(10, size_of_sudoku)[-1, :]
    if not max_iter:
        max_iter = int(np.sum(1 / np.log10(1 / (1 - number_of_possibilities[1:])) + 1))*P + 1

    chromosome_fitness_tracking = np.zeros([size_of_population, max_iter])
    number_of_ones_tracking = np.zeros([size_of_population, max_iter])

    last_added_subset = np.ones(size_of_population) * (-1)

    founded = False
    winning_chromosome_val = np.inf
    winning_chromosome = np.empty([])
    backward_time_horizon = np.zeros(size_of_population)
    for i in tqdm(range(max_iter)):
        for j in range(size_of_population):
            current_fitness = fitness_function(population[j], evaluation_matrix)
            chromosome_fitness_tracking[j, i] = current_fitness
            number_of_ones_tracking[j, i] = np.sum(population[j])

            if number_of_ones_tracking[j, i] == number_of_ones_tracking[j, i - 1]:
                backward_time_horizon[j] += 1
            else:
                backward_time_horizon[j] = 0

            if current_fitness == 0:
                founded = True
                winning_chromosome_val = current_fitness
                winning_chromosome = population[j]
            elif current_fitness < winning_chromosome_val:
                winning_chromosome_val = current_fitness
                winning_chromosome = population[j].copy()
                winning_chromosome[int(last_added_subset[j])] = 0
            else:
                if np.sum(evaluation_matrix[population[j].astype(bool)].sum(axis=0) > 1) == 0:
                    last_added_subset[j] = np.random.choice(np.argwhere(population[j] == 0).flatten())
                    population[j, int(last_added_subset[j])] = 1
                else:
                    if last_added_subset[j] != -1:
                        index_of_chosen_zero = np.random.choice(np.argwhere(population[j] == 0).flatten())
                        population[j, index_of_chosen_zero] = 1
                        population[j, int(last_added_subset[j])] = 0
                        last_added_subset[j] = index_of_chosen_zero
                    else:
                        index_of_chosen_zero = np.random.choice(np.argwhere(population[j] == 0).flatten())
                        index_of_chosen_one = np.random.choice(np.argwhere(population[j] == 1).flatten())
                        population[j, index_of_chosen_one] = 0
                        population[j, index_of_chosen_zero] = 1
                        last_added_subset[j] = index_of_chosen_one

            # crossover
            if not founded and backward_time_horizon[j] > limit_iteration(population[j].shape[0], np.sum(population[j]),
                                                                          number_of_possibilities,P=P):
                # roulette selection
                temp = chromosome_fitness_tracking[:, i - 1] + backward_time_horizon
                normalization_term = np.sum(np.max(temp) - temp)
                if normalization_term:
                    probabilities = (np.max(temp) - temp) / np.sum(np.max(temp) - temp)
                else:
                    probabilities = np.ones(size_of_population) / size_of_population
                id_of_chosen = np.random.choice(np.arange(size_of_population), 1, p=probabilities)
                chosen = population[id_of_chosen][0].copy()
                chosen[int(last_added_subset[id_of_chosen])] = 0
                bin = binary_list_to_ids(chosen)
                mask = (np.random.rand(bin.shape[0]) * 4).astype(int).astype(bool)
                population[j] = ids_to_binary_list(bin[mask], evaluation_matrix.shape[0])

        if founded:
            return chromosome_fitness_tracking[:, :i + 1], number_of_ones_tracking[:, :i + 1], winning_chromosome
    return chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome


def plot_EPEA_solution(chromosome_fitness_tracking, number_of_ones_tracking):
    print("Fitness value")
    print("Best:", np.min(chromosome_fitness_tracking))
    print("Mean:", np.mean(chromosome_fitness_tracking[:, -1]))
    print("Worst:", np.max(chromosome_fitness_tracking[:, -1]))

    print("\nNumber of filled squares")
    print("Max:", np.max(number_of_ones_tracking))
    print("Mean:", np.mean(number_of_ones_tracking[:, -1]))
    print("Min:", np.min(number_of_ones_tracking[:, -1]))

    N = np.arange(chromosome_fitness_tracking.shape[1])
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Progressive evolutionary algorithm')
    axs[0].set_title('Fitness value')
    axs[1].set_title('Number of covered squares')
    for line1, line2 in zip(chromosome_fitness_tracking, number_of_ones_tracking):
        axs[0].plot(N, line1)
        axs[1].plot(N, line2)
    plt.show()


def ids_to_binary_list(ids, length):
    output = np.zeros(length)
    for id in ids:
        output[id] = 1
    return output


def binary_list_to_ids(binary_list):
    output = []
    for i, j in enumerate(binary_list):
        if j:
            output.append(i)
    return np.array(output)
