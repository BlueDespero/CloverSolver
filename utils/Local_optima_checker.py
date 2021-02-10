import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.special import binom
from time import time

from utils.Sudoku_transcription import *
from utils.algorithm_x import remove_intersections
from utils.Common_functions import *


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
