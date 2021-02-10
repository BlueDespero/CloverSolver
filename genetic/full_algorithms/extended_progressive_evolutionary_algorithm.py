import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from genetic.full_algorithms.estimation_of_success_rate_in_EPEA import cumulate_data
from utils.common import ids_to_binary_list, binary_list_to_ids, fitness_function


def limit_iteration(n, k, number_of_possibilities, P):
    index = int((k / n) * number_of_possibilities.shape[0])
    if index == 0:
        return 1 * P
    return int(1 / np.log10(1 / (1 - number_of_possibilities[index])) + 1) * P


def Extended_progressive_evolutionary_algorithm(transcription_matrix, size_of_population,
                                                size_of_sudoku, max_iter=None, P=4, crossover_parameter=0.5,
                                                tqdm_mode=False):
    # initialization of first generation
    population = np.zeros([size_of_population, transcription_matrix.shape[0]])
    for i in range(size_of_population):
        population[i, np.random.randint(0, transcription_matrix.shape[0], 1)] = 1

    number_of_possibilities = cumulate_data(10, size_of_sudoku, tqdm_mode=tqdm_mode)[-1, :]
    if not max_iter:
        max_iter = int(np.sum(1 / np.log10(1 / (1 - number_of_possibilities[1:])) + 1)) * P + 1

    chromosome_fitness_tracking = np.zeros([size_of_population, max_iter])
    number_of_ones_tracking = np.zeros([size_of_population, max_iter])

    last_added_subset = np.ones(size_of_population) * (-1)

    founded = False
    winning_chromosome_val = np.inf
    winning_chromosome = np.empty([])
    backward_time_horizon = np.zeros(size_of_population)

    if tqdm_mode:
        this_range = tqdm(range(max_iter))
    else:
        this_range = range(max_iter)
    for i in this_range:
        for j in range(size_of_population):
            current_fitness = fitness_function(population[j], transcription_matrix)
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
                if np.sum(transcription_matrix[population[j].astype(bool)].sum(axis=0) > 1) == 0:
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
                                                                          number_of_possibilities, P=P):
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
                mask = (np.random.rand(bin.shape[0]) * (1 / (1 - crossover_parameter))).astype(int).astype(bool)
                population[j] = ids_to_binary_list(bin[mask], transcription_matrix.shape[0])

        if founded:
            return chromosome_fitness_tracking[:, :i + 1], number_of_ones_tracking[:, :i + 1], winning_chromosome
    return chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome


def plot_EPEA_solution(transcription_matrix, size_of_population,
                       size_of_sudoku, max_iter=None, P=4, crossover_parameter=0.5, tqdm_mode=False):
    chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome = Extended_progressive_evolutionary_algorithm(
        transcription_matrix, size_of_population,
        size_of_sudoku, max_iter=max_iter, P=P, crossover_parameter=crossover_parameter, tqdm_mode=tqdm_mode)

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
