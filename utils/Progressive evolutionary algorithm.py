import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def fitness_function(chromosome, evaluation_matrix,
                     model=lambda once, more: more - once):
    # input:
    # chromosome = 1-D array(n)
    # evaluation_matrix = np.array(n,m) of zeros and ones
    # model = lambda int, int: int; also float type numbers will work well
    # model - describes dependency of covered elements and covered more than once on fitness function
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


def Progressive_evolutionary_algorithm(evaluation_matrix, size_of_population, max_iter, tqdm_mode=False):
    # initialization of first generation
    sol = np.hstack(
        [np.ones(1), np.zeros(evaluation_matrix.shape[0] - 1)])
    sol = sol.astype(int)
    np.random.shuffle(sol)
    population = sol
    for _ in range(size_of_population - 1):
        np.random.shuffle(sol)
        population = np.vstack([population, sol])

    chromosome_fitness_tracking = np.zeros([size_of_population, max_iter])
    number_of_ones_tracking = np.zeros([size_of_population, max_iter])

    last_added_subset = np.ones(size_of_population) * (-1)

    founded = False
    winning_chromosome = np.empty([])

    if tqdm_mode:
        this_range = tqdm(range(max_iter))
    else:
        this_range = range(max_iter)

    for i in this_range:
        for j in range(size_of_population):
            current_fitness = fitness_function(population[j], evaluation_matrix)
            chromosome_fitness_tracking[j, i] = current_fitness
            number_of_ones_tracking[j, i] = np.sum(population[j])
            if current_fitness == 0:
                print("I found it!")
                founded = True
                winning_chromosome = population[j]
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
        if founded:
            return chromosome_fitness_tracking[:, :i + 1], number_of_ones_tracking[:, :i + 1], winning_chromosome
    return chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome


def plot_PEA_solution(evaluation_matrix, size_of_population, max_iter, tqdm_mode=False):
    chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome = Progressive_evolutionary_algorithm(
        evaluation_matrix,
        size_of_population,
        max_iter,
        tqdm_mode=tqdm_mode)

    print("Fitness value")
    print("Best:", np.min(chromosome_fitness_tracking.min(axis=1)))
    print("Mean:", np.mean(chromosome_fitness_tracking[:, -1]))
    print("Worst:", np.max(chromosome_fitness_tracking[:, -1]))

    print("\nNumber of filled squares")
    print("Max:", np.max(number_of_ones_tracking.max(axis=1)))
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
