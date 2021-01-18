import numpy as np


def Algorithm_X(Given_matrix):
    # input:
    # Given_matrix = np.array([x,y]) containing only ones and zeros

    def Sub_alg(A, d):
        if A.size == 0:
            return [-1]

        # Sort columns by number of ones
        A = (A.T[np.argsort(A.sum(axis=0))]).T

        rows = A[A[:, 0] == 1]
        objective_row_numbers = d[A[:, 0] == 1]

        if rows.size == 0:
            return []

        solution = []
        for i, objective_row_number in zip(range(rows.shape[0]), objective_row_numbers):
            B = A.copy()
            d_c = d.copy()
            for j in np.arange(B.shape[1])[rows[i] == 1][::-1]:
                # removing each row which intersection with rows[i] is not empty
                k = B[:, j] == 0
                B = B[k]
                d_c = d_c[k]
                # removing each column that rows[i] covers
                B = np.delete(B, j, axis=1)

            previous_solution = Sub_alg(B, d_c)

            if previous_solution == [-1]:
                solution.append([objective_row_number])

            elif previous_solution:
                solution += [element + [objective_row_number] for element in previous_solution]

        return solution

    # output:
    # List of lists containing indexes of rows of given matrix,
    # which are good solution of problem.
    # If there is no good solution algorythm return empty [[]].
    return Sub_alg(Given_matrix, np.arange(Given_matrix.shape[0]))


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
    indexes_of_columns_to_delete = []

    index_of_constraint_matrix = np.array(
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

    index_of_constraint_matrix = np.delete(index_of_constraint_matrix, indexes_of_rows_to_delete, axis=0)
    Row_column = np.delete(Row_column, indexes_of_rows_to_delete, axis=0)
    Row_number = np.delete(Row_number, indexes_of_rows_to_delete, axis=0)
    Column_number = np.delete(Column_number, indexes_of_rows_to_delete, axis=0)
    Box_number = np.delete(Box_number, indexes_of_rows_to_delete, axis=0)

    # # removing redundant columns
    # Row_column = np.delete(Row_column,[int(x*size+y) for x,y,z in coordinates_and_values], axis=1)
    # Row_number = np.delete(Row_number,[int(x*size+z-1) for x,y,z in coordinates_and_values], axis=1)
    # Column_number = np.delete(Column_number, [int(y*size+z-1) for x,y,z in coordinates_and_values], axis=1)
    # Box_number = np.delete(Box_number,[int(int(int(y/np.sqrt(size))+int(x/np.sqrt(size))*np.sqrt(size))*size+z-1) for x,y,z in coordinates_and_values], axis=1)


    # output:
    # tuple with two elements
    # first: np.array([n,3]), each row [x,y,z] represents coordinates (x,y) of number z
    # second: np.array([n,m]), matrix of constraints
    return index_of_constraint_matrix, np.hstack([Row_column, Row_number, Column_number, Box_number])

    # return index_of_constraint_matrix, Row_column, Row_number, Column_number, Box_number


def print_sudoku(grid=np.empty([0, 0]), row_numbers=None, indexes=None):
    # input:
    # 1.
    # grid = np.array([n,n]) full of intigers from 1 to n or 0 representing empty square
    # 2.
    # row_numbers = np.array([n**2]) or list - numbers of rows
    # indexes = np.array([m,3]) - each row [x,y,z] is list of coordinates (x,y) of value z

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
        X = []
        Y = []
        Z = []
        for i in row_numbers:
            x, y, z = indexes[i]
            X.append(x)
            Y.append(y)
            Z.append(z)
        size = int(np.sqrt(len(X)))
        grid = np.zeros([size, size])
        for x, y, z in zip(X, Y, Z):
            grid[x, y] = z
        print_sudoku(grid=grid)

def sudoku_solution_checker(index_of_constraint_matrix, indexes):
    # input:
    # index_of_constraint_matrix = np.array([m,3]) - each row [x,y,z] is
    # a list of coordiantes (x,y) and value z
    # indexes = np.array([n]) or list - list of indexes of rows
    # from index_of_constraint_matrix

    X = []
    Y = []
    Z = []
    for id in indexes:
        x,y,z = index_of_constraint_matrix[id]
        X.append(x)
        Y.append(y)
        Z.append(z)
    if np.sqrt(len(X))%1!=0:
        return False
    size = int(np.sqrt(len(X)))
    grid = np.empty([size,size])
    for x,y,z in zip(X,Y,Z):
        grid[x,y]=z

    # check row-number
    for l in grid:
        line = sorted(l)
        if line[0] != 1:
            return False
        for i in range(1, size):
            if line[i - 1] + 1 != line[i]:
                return False

    # check column number
    for l in grid.T:
        line = sorted(l)
        if line[0] != 1:
            return False
        for i in range(1, size):
            if line[i - 1] + 1 != line[i]:
                return False

    # check box-number
    Box = np.zeros([size, size])
    numerator = [0] * size
    for y in range(size):
        for x in range(size):
            current = int(int(x / np.sqrt(size)) + int(y / np.sqrt(size)) * np.sqrt(size))
            Box[current, numerator[current]] = grid[x, y]
            numerator[current] += 1
    for box in Box:
        line = sorted(box)
        if line[0] != 1:
            return False
        for i in range(1, size):
            if line[i - 1] + 1 != line[i]:
                return False
    return True