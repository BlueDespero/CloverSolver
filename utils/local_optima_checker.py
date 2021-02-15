from scipy.special import binom
from tqdm.auto import tqdm

from utils.algorithm_x import remove_intersections
from utils.common import *


def increment(chromosome, max_val):
    incremented = False
    max_vals = np.arange(chromosome.shape[0]) + (max_val - chromosome.shape[0])
    for i in range(chromosome.shape[0] - 1, -1, -1):
        if chromosome[i] < max_vals[i]:
            chromosome[i] += 1
            for j in range(i + 1, chromosome.shape[0]):
                chromosome[j] = chromosome[j - 1] + 1
            if chromosome[0] >= max_vals[0]:
                incremented = True
            break
    return incremented


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def Local_optima_estimator(transcription_matrix, size, true_solution):
    acceptable = np.zeros(size ** 2)
    not_acceptable = np.zeros(size ** 2)
    true_acceptable = np.zeros(size ** 2)

    for i in range(size ** 2 - 1, -1, -1):
        print(i, "\n")
        chromosome = np.arange(size ** 2 - i)
        for _ in tqdm(
                range(int(np.sum([binom(transcription_matrix.shape[0], (size ** 2 - j)) for j in range(size ** 2)])))):
            # for _ in tqdm(range(int(binom(transcription_matrix.shape[0],(size**2-i))))):
            chromosome_bin = ids_to_binary_list(chromosome, transcription_matrix.shape[0]).astype(bool)
            number_of_ones = np.sum(chromosome_bin)
            is_acceptable = np.sum(transcription_matrix[chromosome_bin].sum(axis=0) > 1) == 0

            if is_acceptable:
                acceptable[number_of_ones - 1] += 1
                if sublist(chromosome, true_solution):
                    true_acceptable[number_of_ones - 1] += 1
            else:
                not_acceptable[number_of_ones - 1] += 1

            incremented = increment(chromosome, transcription_matrix.shape[0])
            if incremented:
                break

    print(repr(true_acceptable))
    print(repr(acceptable))
    print(repr(not_acceptable))
    return acceptable / (acceptable + not_acceptable)


def Local_optima_estimator2(transcription_matrix):
    Local_optima = 0
    Not_local_optima = 0

    for i in tqdm(range(2 ** transcription_matrix.shape[0])):
        chromosome = fill_chromosome(bin(i)[2:], transcription_matrix.shape[0])
        fitness_value = fitness_function(chromosome, transcription_matrix)
        for j in range(transcription_matrix.shape[0]):
            temp_chromosome = chromosome.copy()
            temp_chromosome[j] = abs(chromosome[j] - 1)
            if fitness_value < fitness_function(temp_chromosome, transcription_matrix):
                Local_optima += 1
            else:
                Not_local_optima += 1
    print("No. of local optima:", Local_optima)
    print("No. of not optima:", Not_local_optima)
    return Local_optima, Not_local_optima


def fill_chromosome(partial_solution, length_of_chromosome):
    return np.array([int(bit) for bit in partial_solution + "0" * (length_of_chromosome - len(partial_solution))])


def local_minima_counter(given_matrix):
    order = np.arange(given_matrix.shape[0])
    np.random.shuffle(order)

    def sub_alg(sets, original_indexes, depth):
        if sets.size == 0:
            return [-1]

        # Sort columns by number of ones
        # sets = (sets.T[np.argsort(sets.sum(axis=0))]).T

        rows = sets[sets[:, 0] == 1]
        objective_row_numbers = original_indexes[sets[:, 0] == 1]

        if rows.size == 0:
            return [-1]

        solution = []
        for i, objective_row_number in zip(range(rows.shape[0]), objective_row_numbers):

            sets_c, new_indexes = remove_intersections(sets, original_indexes, rows, i)

            previous_solution = sub_alg(sets_c, new_indexes, depth + 1)

            if previous_solution == [-1]:
                solution.append([objective_row_number])

            else:
                solution += [element + [objective_row_number] for element in previous_solution]

        return solution

    l = sub_alg(given_matrix[order], np.arange(given_matrix.shape[0])[order], 0)
    return l


def solution_tree(tree_matrix, coordinates):
    # TODO Sudoku solving can be represented as a decision tree, which shows difficulty level of its - this function
    #  will do this.
    return None
