import numpy as np

from genetic.common import sudoku_size_from_solution


def mutate_population(popultaion, mutation_operator, mutation_rate, initial_situation):
    for i, p in enumerate(popultaion):
        if np.random.uniform(0, 1) > mutation_rate:
            popultaion[i] = mutation_operator(p, initial_situation)

    return popultaion


def reverse_bit_mutation(individual, initial_situation):
    index = np.random.choice(range(individual.shape[0]), 1)[0]
    individual[index] ^= 1

    return np.bitwise_or(individual, initial_situation)


def setup_phenotype_mutation(individual, initial_situation):
    individual = np.bitwise_and(individual, np.invert(initial_situation))

    sudoku_size = sudoku_size_from_solution(individual)
    row_lenght = sudoku_size ** 2
    chosen_element = np.random.choice(range(sudoku_size), 1)[0]

    return individual, sudoku_size, row_lenght, chosen_element


def shuffle_row_mutation(individual, initial_situation):
    individual, sudoku_size, row_lenght, chosen_row = setup_phenotype_mutation(individual, initial_situation)

    row_begin = chosen_row * row_lenght
    row_end = row_begin + row_lenght
    row = individual[row_begin:row_end]
    row = np.reshape(row, (-1, sudoku_size))
    individual[row_begin:row_end] = np.random.permutation(row).flatten()

    return np.bitwise_or(individual, initial_situation)


def shuffle_column_mutation(individual, initial_situation):
    individual, sudoku_size, row_lenght, chosen_column = setup_phenotype_mutation(individual, initial_situation)

    column_begin = chosen_column * sudoku_size
    column_end = column_begin + sudoku_size
    column_values = np.array(
        [individual[column_begin + (i * row_lenght):column_end + (i * row_lenght)] for i in range(sudoku_size)])
    column_values = np.random.permutation(column_values)

    for i in range(sudoku_size):
        individual[column_begin + (i * row_lenght):column_end + (i * row_lenght)] = column_values[i]

    return np.bitwise_or(individual, initial_situation)


def shuffle_box_mutation(individual, initial_situation):
    individual, sudoku_size, row_lenght, chosen_box = setup_phenotype_mutation(individual, initial_situation)

    box_size = int(np.sqrt(sudoku_size))

    box_begin = (chosen_box // box_size) * (box_size * sudoku_size)
    box_end = box_begin + (box_size * sudoku_size)
    box_values = np.array(
        [individual[box_begin + (i * row_lenght):box_end + (i * row_lenght)] for i in range(box_size)])
    box_values = np.reshape(box_values.flatten(), (-1, sudoku_size))
    box_values = np.random.permutation(box_values).flatten()
    box_values = np.reshape(box_values, (-1, box_size * sudoku_size))

    for i in range(box_size):
        individual[box_begin + (i * row_lenght):box_end + (i * row_lenght)] = box_values[i]

    return np.bitwise_or(individual, initial_situation)
