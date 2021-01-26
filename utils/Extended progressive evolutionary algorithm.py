import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


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
        x, y, z = index_of_constraint_matrix[id]
        X.append(x)
        Y.append(y)
        Z.append(z)
    if np.sqrt(len(X)) % 1 != 0:
        return False
    size = int(np.sqrt(len(X)))
    grid = np.empty([size, size])
    for x, y, z in zip(X, Y, Z):
        grid[x, y] = z

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


def fittnes_function(chromosome, evaluation_matrix,
                     model=lambda once, more: more - once):
    # input:
    # chromosome = 1-D array(n)
    # evaluation_matrix = np.array(n,m) of zeros and ones
    # model = lambda int, int: int; also float type numbers will work well
    # model - describes dependency of coverd elements and coverd more than once on fittnes function
    # for default model minimum of fittnes function is always zero

    filter = chromosome.astype(bool)
    solution = evaluation_matrix[filter]
    flattened = solution.sum(axis=0)
    covered_more_than_once = np.sum(flattened > 1)
    covered_once = np.sum(flattened == 1)

    # output:
    # int/float
    return model(covered_once, covered_more_than_once) + flattened.shape[0]


def mutation_one(chromosome):
    # input:
    # chromosome = 1-D array

    index_of_chosen_one = np.random.choice(np.argwhere(chromosome == 1).flatten())
    index_of_chosen_zero = np.random.choice(np.argwhere(chromosome == 0).flatten())

    # swap
    new = chromosome.copy()
    new[index_of_chosen_one] = 0
    new[index_of_chosen_zero] = 1

    # output:
    # new = 1-D array
    return new


def Progressive_evolutionary_algorithm_1(evaluation_matrix, size_of_population, initial_number_of_subsets, max_iter):
    # initialization of first generation
    sol = np.hstack(
        [np.ones(initial_number_of_subsets), np.zeros(evaluation_matrix.shape[0] - initial_number_of_subsets)])
    sol = sol.astype(int)
    np.random.shuffle(sol)
    population = sol
    for _ in range(size_of_population - 1):
        np.random.shuffle(sol)
        population = np.vstack([population, sol])

    chromosome_fitness_tracking = np.zeros([size_of_population, max_iter])
    number_of_ones_tracking = np.zeros([size_of_population, max_iter])

    last_added_subset = np.ones(size_of_population) * (-1)

    founded = False
    winning_chromosome = np.empty([])

    for i in tqdm(range(max_iter)):
        for j in range(size_of_population):
            current_fitness = fittnes_function(population[j], evaluation_matrix)
            chromosome_fitness_tracking[j, i] = current_fitness
            number_of_ones_tracking[j, i] = np.sum(population[j])
            if current_fitness == 0:
                print("I found it!")
                founded = True
                winning_chromosome = population[j]
            else:
                if np.sum(evaluation_matrix[population[j].astype(bool)].sum(axis=0) > 1) == 0:
                    last_added_subset[j] = np.random.choice(np.argwhere(population[j] == 0).flatten())
                    population[j, int(last_added_subset[j])] = 1
                else:
                    if last_added_subset[j] != -1:
                        index_of_chosen_zero = np.random.choice(np.argwhere(population[j] == 0).flatten())
                        population[j, index_of_chosen_zero] = 1
                        population[j, int(last_added_subset[j])] = 0
                        last_added_subset[j] = index_of_chosen_zero
                    else:
                        index_of_chosen_zero = np.random.choice(np.argwhere(population[j] == 0).flatten())
                        index_of_chosen_one = np.random.choice(np.argwhere(population[j] == 1).flatten())
                        population[j, index_of_chosen_one] = 0
                        population[j, index_of_chosen_zero] = 1
                        last_added_subset[j] = index_of_chosen_one
        if founded:
            return chromosome_fitness_tracking[:, :i + 1], number_of_ones_tracking[:, :i + 1], winning_chromosome
    return chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome


def plot_PEA_solution_1(chromosome_fitness_tracking, number_of_ones_tracking):
    print("Fitness value")
    print("Best:", np.min(chromosome_fitness_tracking))
    print("Mean:", np.mean(chromosome_fitness_tracking))
    print("Worst:", np.max(chromosome_fitness_tracking))

    print("\nNumber of filled squares")
    print("Max:", np.max(number_of_ones_tracking))
    print("Mean:", np.mean(number_of_ones_tracking))
    print("Min:", np.min(number_of_ones_tracking))

    N = np.arange(chromosome_fitness_tracking.shape[1])
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Progressive evolutionary algorithm')
    axs[0].set_title('Fitness value')
    axs[1].set_title('Number of covered squares')
    for line1, line2 in zip(chromosome_fitness_tracking, number_of_ones_tracking):
        axs[0].plot(N, line1)
        axs[1].plot(N, line2)
    plt.show()


if __name__ == "__main__":
    # grid = np.array([[0, 1, 0, 0],
    #                  [0, 2, 0, 0],
    #                  [0, 0, 1, 0],
    #                  [0, 0, 4, 0]])
    # np.random.seed(2)

    grid = np.array([[0, 0, 0, 0, 3, 7, 6, 0, 0],
                     [0, 0, 0, 6, 0, 0, 0, 9, 0],
                     [0, 0, 8, 0, 0, 0, 0, 0, 4],
                     [0, 9, 0, 0, 0, 0, 0, 0, 1],
                     [6, 0, 0, 0, 0, 0, 0, 0, 9],
                     [3, 0, 0, 0, 0, 0, 0, 4, 0],
                     [7, 0, 0, 0, 0, 0, 8, 0, 0],
                     [0, 1, 0, 0, 0, 9, 0, 0, 0],
                     [0, 0, 2, 5, 4, 0, 0, 0, 0]])

    grid2 = np.array([[1, 0, 0, 9, 0, 4, 0, 8, 2],
                      [0, 5, 2, 6, 8, 0, 3, 0, 0],
                      [8, 6, 4, 2, 0, 0, 9, 1, 0],
                      [0, 1, 0, 0, 4, 9, 8, 0, 6],
                      [4, 9, 8, 3, 0, 0, 7, 0, 1],
                      [6, 0, 7, 0, 1, 0, 0, 9, 3],
                      [0, 8, 6, 0, 3, 5, 2, 0, 9],
                      [5, 0, 9, 0, 0, 2, 1, 2, 0],
                      [0, 3, 0, 4, 9, 7, 0, 0, 8]])

    grid_hard = np.array([[0, 0, 0, 8, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 4, 3],
                          [5, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 7, 0, 8, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 2, 0, 0, 3, 0, 0, 0, 0],
                          [6, 0, 0, 0, 0, 0, 0, 7, 5],
                          [0, 0, 3, 4, 0, 0, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0, 6, 0, 0]])

    # np.random.seed(2)


    no_of_empty_squares = 16
    rows, evaluation_matrix = sudoku_matrix_representation(grid)

    chromosome_fitness_tracking, number_of_ones_tracking, winning_chromosome = Progressive_evolutionary_algorithm_1(
        evaluation_matrix, 15, 1, 10000)
    plot_PEA_solution_1(chromosome_fitness_tracking, number_of_ones_tracking)

    if winning_chromosome.size > 0:
        print_sudoku(row_numbers=np.arange(evaluation_matrix.shape[0])[winning_chromosome.astype(bool)], indexes=rows)
