from utils.algorithm_x import remove_intersections
from utils.common import ids_to_binary_list, binary_list_to_ids
from utils.sudoku_transcription import *


def sudoku_matrix_representation2(grid):
    # input:
    # grid = np.array([n,n]) full of integers from range 1 to n
    # or zeros representing empty squares
    # Difference between this and previous method is that, here I remove all rows and columns that intersects with
    # initial values in grid.

    size = grid.shape[0]

    # grid to coordinates and values
    coordinates_and_values = np.array([[x, y, grid[x, y]] for x in range(size) for y in range(size) if grid[x, y] != 0])

    Row_column = np.zeros([size ** 3, size ** 2])
    Row_number = np.zeros([size ** 3, size ** 2])
    Column_number = np.zeros([size ** 3, size ** 2])
    Box_number = np.zeros([size ** 3, size ** 2])

    coordinates = np.array(
        [[x, y, z] for x in range(size) for y in range(size) for z in range(1, size + 1)])

    for x in range(size):
        for y in range(size):
            for z in range(1, size + 1):
                Box_number[x * size * size + y * size + z - 1, int(
                    int(x / np.sqrt(size)) + int(y / np.sqrt(size)) * np.sqrt(size)) * size + z - 1] = 1
                Row_column[x * size * size + y * size + z - 1, x * size + y] = 1
                Row_number[x * size * size + y * size + z - 1, x * size + z - 1] = 1
                Column_number[x * size * size + y * size + z - 1, y * size + z - 1] = 1

    rows_with_set_value = []

    for x, y, z in coordinates_and_values:
        rows_with_set_value.append(x * (size ** 2) + y * size + z - 1)
    rows_with_set_value = np.array(rows_with_set_value)

    transcription_matrix = np.hstack([Row_column, Row_number, Column_number, Box_number])
    ids = ids_to_binary_list(np.arange(transcription_matrix.shape[0]), transcription_matrix.shape[0])
    for row_id in rows_with_set_value:
        _, temp_ids = remove_intersections(transcription_matrix,
                                           np.arange(transcription_matrix.shape[0]),
                                           [transcription_matrix[row_id]], 0)
        ids = np.logical_and(ids, ids_to_binary_list(temp_ids, ids.shape[0])).astype(int)
    ids = binary_list_to_ids(ids)
    coordinates = coordinates[ids]

    transcription_matrix = transcription_matrix[ids]
    transcription_matrix = transcription_matrix[:, transcription_matrix.sum(axis=0) != 0]

    # output:
    # tuple with two elements
    # first: np.array([n,3]), each row [x,y,z] represents coordinates (x,y) of number z
    # second: np.array([n,m]), matrix of constraints
    return coordinates.astype(int), transcription_matrix.astype(int)


def print_sudoku2(initial_grid, chromosome, coordinates, size_of_grid):
    # input optional:
    # initial_grid = np.array([n,n]) full of integers from 1 to n or 0 representing initialize form of sudoku
    # solution = np.array([n**2]) or list - list of numbers of rows of transcription matrix
    # coordinates = np.array([m,3]) - each row [x,y,z] is list of coordinates (x,y) of value z

    solution_grid = grid_from_coordinates(chromosome, coordinates, size_of_grid)
    for y in range(initial_grid.shape[1]):
        s = ""
        for x in range(initial_grid.shape[0]):
            if initial_grid[y, x] != 0:
                s += str(int(initial_grid[y, x]))
            elif solution_grid[y, x] != 0:
                s += str(int(solution_grid[y, x]))
            else:
                s += "_"
            if x % np.sqrt(initial_grid.shape[0]) == np.sqrt(initial_grid.shape[0]) - 1 and x != initial_grid.shape[
                0] - 1:
                s += "|"
        if y % np.sqrt(initial_grid.shape[1]) == np.sqrt(initial_grid.shape[1]) - 1 and y != initial_grid.shape[1] - 1:
            s += "\n" + (int(np.sqrt(initial_grid.shape[0])) * "-" + "+") * int(
                np.sqrt(initial_grid.shape[0]) - 1) + int(
                np.sqrt(initial_grid.shape[0])) * "-"
        print(s)
