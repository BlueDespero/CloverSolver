import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


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


def Simulated_annealing(evaluation_matrix, no_of_empty_squares, no_of_iterations, mutation, alpha=0.1):
    # input:
    # no_of_iterations = int; number of iterations for each of individual
    # evaluation_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill
    # mutation = lambda chromosome: changed_chromosome
    # alpha = float from 0 to 1; probability of changing solution to worse

    sol = np.hstack([np.ones(no_of_empty_squares), np.zeros(evaluation_matrix.shape[0] - no_of_empty_squares)])
    np.random.shuffle(sol)
    x = sol
    x_cost = fitness_function(x, evaluation_matrix)
    best_val = x_cost
    best_individual = x
    for t in range(no_of_iterations):
        y = mutation(x)
        y_cost = fitness_function(y, evaluation_matrix)
        if (y_cost < x_cost):
            x, x_cost = y, y_cost
        elif (np.random.rand() < np.exp(- alpha * (y_cost - x_cost) * t / no_of_iterations)):
            x, x_cost = y, y_cost
        if x_cost < best_val:
            best_val = x_cost
            best_individual = x

    # output:
    # best_val = int; fitness value of best_individual
    # best_individual = 1-D array
    return best_val, best_individual


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


def plot_simulated_annealing_solution(no_of_repetitions, no_of_iterations, evaluation_matrix, no_of_empty_squares,
                                      mutation, alpha=0.1):
    # input:
    # no_of_repetitions = int; number of individuals
    # no_of_iterations = int; number of iterations for each of individual
    # evaluation_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill
    # mutation = lambda chromosome: changed_chromosome
    # alpha = float from 0 to 1; probability of changing solution to worse

    solutions = np.zeros(no_of_repetitions)
    for i in tqdm(range(no_of_repetitions)):
        score, solution = Simulated_annealing(evaluation_matrix, no_of_empty_squares, no_of_iterations, mutation, alpha)
        solutions[i] = score
    plt.hist(solutions, bins=int(np.sqrt(no_of_repetitions)))
    plt.title("Simulated annealing - " + str(no_of_repetitions) + " solutions ditribution")
    plt.show()
    print("\nBest:", np.min(solutions))
    print("Mean:", np.mean(solutions))
    print("Worst:", np.max(solutions))
