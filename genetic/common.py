import numpy as np


def sort_population_by_fitness(population, fitness):
    p = np.array(fitness.argsort())
    return np.array([fitness[i] for i in p]), np.array([population[i] for i in p])


def roulette_wheel_selection(set_o, weights, except_for=[]):
    for e in except_for:
        del_indexes = [i for i, s in enumerate(set_o) if list(s) == list(e)]
        set_o = np.delete(set_o, del_indexes, axis=0)
        weights = np.delete(weights, del_indexes)

    w_sum = np.sum(weights)
    border = np.random.uniform(0, w_sum)
    current = 0
    for i, s in enumerate(set_o):
        current += weights[i]
        if current > border:
            return s
    return except_for[0]


def lambda_coma_mu(parents, children):
    return children


def lambda_plus_mu(parents, children):
    return np.concatenate((parents, children))


def cube(x):
    if 0 <= x:
        return x ** (1. / 3.)
    return -(-x) ** (1. / 3.)


def sudoku_size_from_solution(individual):
    return int(cube(individual.shape[0]))


def sudoku_full_constraints_set(size=9):
    Row_column = np.zeros([size ** 3, size ** 2])
    Row_number = np.zeros([size ** 3, size ** 2])
    Column_number = np.zeros([size ** 3, size ** 2])
    Box_number = np.zeros([size ** 3, size ** 2])

    for x in range(size):
        for y in range(size):
            for z in range(1, size + 1):
                Box_number[x * size * size + y * size + z - 1, int(
                    int(x / np.sqrt(size)) + int(y / np.sqrt(size)) * np.sqrt(size)) * size + z - 1] = 1
                Row_column[x * size * size + y * size + z - 1, x * size + y] = 1
                Row_number[x * size * size + y * size + z - 1, x * size + z - 1] = 1
                Column_number[x * size * size + y * size + z - 1, y * size + z - 1] = 1

    # output:
    # matrix of constraints
    return np.hstack([Row_column, Row_number, Column_number, Box_number])


def classic_representation_sudoku_into_full_chromosome(sudoku):
    chromosome = np.zeros(sudoku.shape[0] ** 3)
    for i, value in enumerate(sudoku.flatten()):
        if value != 0:
            index = int((i * sudoku.shape[0]) + (value - 1))
            chromosome[index] = 1
    return chromosome
