import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from utils.common import fitness_function, mutation_one


def Simulated_annealing(transcription_matrix, no_of_empty_squares, no_of_iterations, alpha=0.1):
    # input:
    # no_of_iterations = int; number of iterations for each of individual
    # transcription_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill
    # mutation = lambda chromosome: changed_chromosome
    # alpha = float from 0 to 1; probability of changing solution to worse

    sol = np.hstack([np.ones(no_of_empty_squares), np.zeros(transcription_matrix.shape[0] - no_of_empty_squares)])
    np.random.shuffle(sol)
    x = sol
    x_cost = fitness_function(x, transcription_matrix)
    best_val = x_cost
    best_individual = x
    for t in range(no_of_iterations):
        y = mutation_one(x)
        y_cost = fitness_function(y, transcription_matrix)
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


def plot_simulated_annealing_solution(no_of_repetitions, no_of_iterations, transcription_matrix, no_of_empty_squares,
                                      alpha=0.1, tqdm_mode=False):
    # input:
    # no_of_repetitions = int; number of individuals
    # no_of_iterations = int; number of iterations for each of individual
    # transcription_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill
    # mutation = lambda chromosome: changed_chromosome
    # alpha = float from 0 to 1; probability of changing solution to worse

    solutions = np.zeros(no_of_repetitions)

    if tqdm_mode:
        this_range = tqdm(range(no_of_repetitions))
    else:
        this_range = range(no_of_repetitions)

    for i in this_range:
        score, solution = Simulated_annealing(transcription_matrix, no_of_empty_squares, no_of_iterations, alpha)
        solutions[i] = score

    plt.hist(solutions, bins=int(np.sqrt(no_of_repetitions)))
    plt.title("Simulated annealing - " + str(no_of_repetitions) + " solutions distribution")
    plt.show()
    print("\nBest:", np.min(solutions))
    print("Mean:", np.mean(solutions))
    print("Worst:", np.max(solutions))
