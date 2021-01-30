import numpy as np

from utils.algorithm_x import algorithm_x_first_solution, algorithm_x


def sudoku_matrix_representation(grid):
    # input:
    # grid = np.array([n,n]) full of intigers from range 1 to n
    # or zeros representing empty squares

    size = grid.shape[0]

    # grid to coordinates and values
    coordinates_and_values = np.array([[x, y, grid[x, y]] for x in range(size) for y in range(size) if grid[x, y] != 0])

    Row_column = np.zeros([size ** 3, size ** 2])
    Row_number = np.zeros([size ** 3, size ** 2])
    Column_number = np.zeros([size ** 3, size ** 2])
    Box_number = np.zeros([size ** 3, size ** 2])

    indexes_of_rows_to_delete = []

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

    # removing rows to match condition of given grid
    for x, y, z in coordinates_and_values:
        for i in range(size):
            if i + 1 != z:
                indexes_of_rows_to_delete.append([int(x * size * size + y * size + i)])

    coordinates = np.delete(coordinates, indexes_of_rows_to_delete, axis=0)
    Row_column = np.delete(Row_column, indexes_of_rows_to_delete, axis=0)
    Row_number = np.delete(Row_number, indexes_of_rows_to_delete, axis=0)
    Column_number = np.delete(Column_number, indexes_of_rows_to_delete, axis=0)
    Box_number = np.delete(Box_number, indexes_of_rows_to_delete, axis=0)

    # output:
    # tuple with two elements
    # first: np.array([n,3]), each row [x,y,z] represents coordinates (x,y) of number z
    # second: np.array([n,m]), matrix of constraints
    return coordinates, np.hstack([Row_column, Row_number, Column_number, Box_number])


def print_sudoku(grid=np.empty([0, 0]), chromosome=None, coordinates=None,size_of_grid=None):
    # input optional:
    # 1.
    # grid = np.array([n,n]) full of intigers from 1 to n or 0 representing empty square
    # 2.
    # solution = np.array([n**2]) or list - list of numbers of rows of transcription matrix
    # coordinates = np.array([m,3]) - each row [x,y,z] is list of coordinates (x,y) of value z

    if grid.size > 0:
        for y in range(grid.shape[1]):
            s = ""
            for x in range(grid.shape[0]):
                if grid[y, x] != 0:
                    s += str(int(grid[y, x]))
                else:
                    s += "_"
                if x % np.sqrt(grid.shape[0]) == np.sqrt(grid.shape[0]) - 1 and x != grid.shape[0] - 1:
                    s += "|"
            if y % np.sqrt(grid.shape[1]) == np.sqrt(grid.shape[1]) - 1 and y != grid.shape[1] - 1:
                s += "\n" + (int(np.sqrt(grid.shape[0])) * "-" + "+") * int(np.sqrt(grid.shape[0]) - 1) + int(
                    np.sqrt(grid.shape[0])) * "-"
            print(s)
    else:
        print_sudoku(grid=grid_from_coordinates(chromosome, coordinates,size_of_grid))

def grid_from_coordinates(chromosome, coordinates):
    X = []
    Y = []
    Z = []
    for i in chromosome:
        x, y, z = coordinates[i]
        X.append(x)
        Y.append(y)
        Z.append(z)
    if size_of_grid:
        size = size_of_grid
    else:
        size = max(int(np.sqrt(len(X))), max(Z))
    grid = np.zeros([size, size])
    for x, y, z in zip(X, Y, Z):
        grid[x, y] = z
    return grid


def check_if_range(grid, size):
    if [l for l in [sorted(g) for g in grid] if l != range(1, size + 1)] != []:
        return False


def sudoku_solution_checker(coordinates, chromosome):
    # input:
    # coordinates = np.array([m,3]) - each row [x,y,z] is
    # a list of coordinates (x,y) and value z
    # chromosome = np.array([n]) or list - list of indexes of rows
    # from index_of_constraint_matrix

    size = np.sqrt(chromosome.shape[0])

    if size % 1 != 0:
        return False

    size = int(size)

    grid = grid_from_coordinates(chromosome, coordinates)

    # check row-number
    if not check_if_range(grid, size):
        return False

    # check column number
    if not check_if_range(grid.T, size):
        return False

    # check box-number
    Box = np.zeros([size, size])
    numerator = [0] * size
    for y in range(size):
        for x in range(size):
            current = int(int(x / np.sqrt(size)) + int(y / np.sqrt(size)) * np.sqrt(size))
            Box[current, numerator[current]] = grid[x, y]
            numerator[current] += 1

    if not check_if_range(Box, size):
        return False

    return True


def sudoku_generator(size=9):
    if int(np.sqrt(size)) % 1 != 0:
        print("Not valid size!")
        return False

    last_grid = np.zeros([size, size])
    grid = np.zeros([size, size])

    basic_coordinates, transcription_matrix = sudoku_matrix_representation(last_grid)
    basic_solution = np.array(algorithm_x_first_solution(transcription_matrix))
    np.random.shuffle(basic_solution)

    while True:
        last_grid = grid
        basic_solution = basic_solution[1:]
        grid = grid_from_coordinates(basic_solution, basic_coordinates, size_of_grid=size)
        coordinates, transcription_matrix = sudoku_matrix_representation(grid)
        if len(algorithm_x(transcription_matrix)) > 1:
            break
    return last_grid
