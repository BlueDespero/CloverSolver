import numpy as np

from genetic.common import sudoku_size_from_solution


def mutate_population(popultaion, mutation_operator, mutation_rate, initial_situation):
    for i, p in enumerate(popultaion):
        if np.random.uniform(0, 1) > mutation_rate:
            popultaion[i] = np.bitwise_or(mutation_operator(p), initial_situation)

    return popultaion


def reverse_bit_mutation(individual):
    index = np.random.choice(range(individual.shape[0]), 1)[0]
    individual[index] ^= 1

    return individual


def swap_two_rows_mutation(individual):
    sudoku_size = sudoku_size_from_solution(individual)
    row_lenght = sudoku_size ** 2
    row_one_to_swap, row_two_to_swap = np.random.choice(range(sudoku_size), 2)

    m_point_one_begin = row_one_to_swap * row_lenght
    m_point_one_end = m_point_one_begin + row_lenght  # mutation point one begin and end point
    m_point_two_begin = row_two_to_swap * row_lenght
    m_point_two_end = m_point_two_begin + row_lenght  # mutation point two begin and end point

    individual[m_point_one_begin:m_point_one_end], individual[m_point_two_begin:m_point_two_end] = [
        individual[m_point_two_begin:m_point_two_end], individual[m_point_one_begin:m_point_one_end]]

    return individual


def swap_two_columns_mutation(individual):
    sudoku_size = sudoku_size_from_solution(individual)
    row_lenght = sudoku_size ** 2
    column_one_to_swap, column_two_to_swap = np.random.choice(range(sudoku_size), 2)

    m_point_one_begin = column_one_to_swap * sudoku_size
    m_point_one_end = m_point_one_begin + sudoku_size  # mutation point one begin and end point
    m_point_two_begin = column_two_to_swap * sudoku_size
    m_point_two_end = m_point_two_begin + sudoku_size  # mutation point two begin and end point

    for i in range(sudoku_size):
        individual[m_point_one_begin:m_point_one_end], individual[m_point_two_begin:m_point_two_end] = [
            individual[m_point_two_begin:m_point_two_end], individual[m_point_one_begin:m_point_one_end]]
        m_point_one_begin += row_lenght
        m_point_one_end += row_lenght
        m_point_two_begin += row_lenght
        m_point_two_end += row_lenght

    return individual


def swap_two_boxes_mutation(individual):
    sudoku_size = sudoku_size_from_solution(individual)
    row_lenght = sudoku_size ** 2
    box_size = int(np.sqrt(sudoku_size))
    box_one_to_swap, box_two_to_swap = np.random.choice(range(sudoku_size), 2)

    m_point_one_begin = row_lenght * (box_one_to_swap // box_size) + sudoku_size * (box_one_to_swap % box_size)
    m_point_one_end = m_point_one_begin + box_size * sudoku_size
    m_point_two_begin = row_lenght * (box_two_to_swap // box_size) + sudoku_size * (box_two_to_swap % box_size)
    m_point_two_end = m_point_two_begin + box_size * sudoku_size

    for i in range(box_size):
        individual[m_point_one_begin:m_point_one_end], individual[m_point_two_begin:m_point_two_end] = [
            individual[m_point_two_begin:m_point_two_end], individual[m_point_one_begin:m_point_one_end]]
        m_point_one_begin += row_lenght
        m_point_one_end += row_lenght
        m_point_two_begin += row_lenght
        m_point_two_end += row_lenght

    return individual
