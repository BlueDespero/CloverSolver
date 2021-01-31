import numpy as np

from genetic.common import sudoku_size_from_solution


def crossover_population(population, fitness, crossover_operator, selection_method, number_of_children,  initial_situation):
    children_set = []
    for i in range(0, number_of_children, 2):
        parent_1 = selection_method(set_o=np.copy(population), weights=np.copy(fitness))
        parent_2 = selection_method(set_o=np.copy(population), weights=np.copy(fitness), except_for=[parent_1])

        child_1, child_2 = crossover_operator(parent_1, parent_2)
        children_set += [np.bitwise_or(child_1, initial_situation), np.bitwise_or(child_2, initial_situation)]

    return np.array(children_set)


def single_point_crossover(parent_1, parent_2):
    crossover_point = np.random.choice(range(parent_1.shape[0]), 1)[0]
    child_1 = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
    child_2 = np.concatenate((parent_2[:crossover_point], parent_1[crossover_point:]))
    return child_1, child_2


def double_point_crossover(parent_1, parent_2):
    crossover_point = np.sort(np.random.choice(range(parent_1.shape[0]), 2))
    child_1 = np.concatenate(
        (parent_1[:crossover_point[0]], parent_2[crossover_point[0]:crossover_point[1]], parent_1[crossover_point[1]:]))
    child_2 = np.concatenate(
        (parent_2[:crossover_point[0]], parent_1[crossover_point[0]:crossover_point[1]], parent_2[crossover_point[1]:]))
    return child_1, child_2


def phenotype_crossover_setup(parent_1, parent_2):
    sudoku_size = sudoku_size_from_solution(parent_1)
    row_lenght = sudoku_size ** 2
    exchange_number = np.random.choice(range(sudoku_size), 1)[0]
    child_1, child_2 = np.copy(parent_1), np.copy(parent_2)

    return sudoku_size, row_lenght, exchange_number, child_1, child_2


def exchange_two_rows_crossover(parent_1, parent_2):
    sudoku_size, row_lenght, row_to_exchange, child_1, child_2 = phenotype_crossover_setup(parent_1, parent_2)

    crossover_begin = row_to_exchange * row_lenght
    crossover_end = crossover_begin + row_lenght  # crossover point one begin and end point

    child_1[crossover_begin:crossover_end] = parent_2[crossover_begin:crossover_end]
    child_2[crossover_begin:crossover_end] = parent_1[crossover_begin:crossover_end]

    return child_1, child_2


def exchange_two_columns_crossover(parent_1, parent_2):
    sudoku_size, row_lenght, column_to_exchange, child_1, child_2 = phenotype_crossover_setup(parent_1, parent_2)

    crossover_begin = column_to_exchange * sudoku_size
    crossover_end = crossover_begin + sudoku_size  # crossover point one begin and end point

    for i in range(sudoku_size):
        child_1[crossover_begin:crossover_end] = parent_2[crossover_begin:crossover_end]
        child_2[crossover_begin:crossover_end] = parent_1[crossover_begin:crossover_end]
        crossover_begin += row_lenght
        crossover_end += row_lenght

    return child_1, child_2


def exchange_two_boxes_crossover(parent_1, parent_2):
    sudoku_size, row_lenght, box_to_exchange, child_1, child_2 = phenotype_crossover_setup(parent_1, parent_2)

    sudoku_size = sudoku_size_from_solution(parent_1)
    row_lenght = sudoku_size ** 2

    box_to_exchange = np.random.choice(range(sudoku_size), 1)[0]
    box_size = int(np.sqrt(sudoku_size))

    crossover_begin = row_lenght * (box_to_exchange // box_size) + sudoku_size * (box_to_exchange % box_size)
    crossover_end = crossover_begin + box_size * sudoku_size

    for i in range(box_size):
        child_1[crossover_begin:crossover_end] = parent_2[crossover_begin:crossover_end]
        child_2[crossover_begin:crossover_end] = parent_1[crossover_begin:crossover_end]
        crossover_begin += row_lenght
        crossover_end += row_lenght

    return child_1, child_2
