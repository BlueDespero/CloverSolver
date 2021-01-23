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


def Random_search(no_of_iterations, evaluation_matrix, no_of_empty_squares,
                  return_best_individual=False):
    # input:
    # no_of_iterations = int
    # evaluation_matrix = np.array(n,m) of zeros and ones
    # no_of_empty_squares = number of squares to fill
    # return_best_individual = bool


    solutions = np.zeros(no_of_iterations)
    sol = np.hstack([np.ones(no_of_empty_squares), np.zeros(evaluation_matrix.shape[0] - no_of_empty_squares)])

    if return_best_individual:
        np.random.shuffle(sol)
        best_individual = sol
        best_val = fitness_function(best_individual, evaluation_matrix)
        solutions[0] = best_val
        for i in tqdm(range(1, no_of_iterations)):
            np.random.shuffle(sol)
            current_individual = sol
            current_val = fitness_function(current_individual, evaluation_matrix)
            if current_val < best_val:
                best_val = current_val
                best_individual = current_individual
            solutions[i] = current_val

        # output:
        # solutions = np.array(no_of_iterations) with score of each individual
        # best_individual = np.array(evaluation_matrix.shape[0]); chromosome of best best_individual
        return solutions, best_individual
    else:
        for i in tqdm(range(no_of_iterations)):
            np.random.shuffle(sol)
            solutions[i] = fitness_function(sol, evaluation_matrix)

        # output:
        # solutions = np.array(no_of_iterations) with score of each individual
        return solutions

def plot_random_search_solution(no_of_iterations, evaluation_matrix, no_of_empty_squares):
    # input:
    # no_of_iterations = int
    # evaluation_matrix = np.array(n,m) matrix of zeros and ones
    # no_of_empty_squares = int; number of squares to fill

    solutions = Random_search(no_of_iterations, evaluation_matrix, no_of_empty_squares)
    plt.hist(solutions,bins=int(no_of_iterations**(1/3)))
    plt.title("Random search - "+str(no_of_iterations)+" solutions ditribution")
    plt.show()
    print("\nBest:",np.min(solutions))
    print("Mean:",np.mean(solutions))
    print("Worst:",np.max(solutions))