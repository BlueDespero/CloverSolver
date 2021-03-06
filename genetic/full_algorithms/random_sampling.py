import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from utils.common import fitness_function


def Random_search(no_of_iterations, transcription_matrix, no_of_empty_squares,
                  return_best_individual=False, tqdm_mode=False):
    # input:
    # no_of_iterations = int
    # transcription_matrix = np.array(n,m) of zeros and ones
    # no_of_empty_squares = number of squares to fill
    # return_best_individual = bool

    solutions = np.zeros(no_of_iterations)
    sol = np.hstack([np.ones(no_of_empty_squares), np.zeros(transcription_matrix.shape[0] - no_of_empty_squares)])

    if return_best_individual:
        np.random.shuffle(sol)
        best_individual = sol
        best_val = fitness_function(best_individual, transcription_matrix)
        solutions[0] = best_val
        if tqdm_mode:
            this_range = tqdm(range(1, no_of_iterations))
        else:
            this_range = range(1, no_of_iterations)
        for i in this_range:
            np.random.shuffle(sol)
            current_individual = sol
            current_val = fitness_function(current_individual, transcription_matrix)
            if current_val < best_val:
                best_val = current_val
                best_individual = current_individual
            solutions[i] = current_val

        # output:
        # solutions = np.array(no_of_iterations) with score of each individual
        # best_individual = np.array(transcription_matrix.shape[0]); chromosome of best best_individual
        return solutions, best_individual
    else:

        if tqdm_mode:
            this_range = tqdm(range(no_of_iterations))
        else:
            this_range = range(no_of_iterations)

        for i in this_range:
            np.random.shuffle(sol)
            solutions[i] = fitness_function(sol, transcription_matrix)

        # output:
        # solutions = np.array(no_of_iterations) with score of each individual
        return solutions


def plot_random_search_solution(no_of_iterations, transcription_matrix, no_of_empty_squares, tqdm_mode=False):
    # input:
    # no_of_iterations = int
    # transcription_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill

    solutions = Random_search(no_of_iterations, transcription_matrix, no_of_empty_squares, tqdm_mode=tqdm_mode)
    plt.hist(solutions, bins=int(no_of_iterations ** (1 / 3)))
    plt.title("Random search - " + str(no_of_iterations) + " solutions distribution")
    plt.show()
    print("\nBest:", np.min(solutions))
    print("Mean:", np.mean(solutions))
    print("Worst:", np.max(solutions))


def plot_random_search_solution_multi(no_of_reps, no_of_iterations, transcription_matrix, no_of_empty_squares,
                                      tqdm_mode=False):
    # input:
    # no_of_iterations = int
    # transcription_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill
    solutions = []
    for i in range(no_of_reps):
        solutions.append(
            np.min(Random_search(no_of_iterations, transcription_matrix, no_of_empty_squares, tqdm_mode=tqdm_mode)))
    plt.hist(solutions, bins=int(len(solutions) ** (1 / 3)))
    plt.title("Random search - " + str(no_of_iterations) + " repetitions")
    plt.show()
    print("\nBest:", np.min(solutions))
    print("Mean:", np.mean(solutions))
    print("Worst:", np.max(solutions))
