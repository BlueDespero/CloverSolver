import numpy as np


def algorithm_x(Given_matrix):
    return sub_alg(Given_matrix, np.arange(Given_matrix.shape[0]))


def remove_intersections(sets, sets_ids, selected_rows, id):
    B = sets.copy()
    d_c = sets_ids.copy()
    for j in np.arange(B.shape[1])[selected_rows[id] == 1][::-1]:
        # removing each row which intersection with rows[i] is not empty
        k = B[:, j] == 0
        B = B[k]
        d_c = d_c[k]
        # removing each column that rows[i] covers
        B = np.delete(B, j, axis=1)

    return B, d_c


def sub_alg(A, d):
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
        B, d_c = remove_intersections(A, d, rows, i)

        previous_solution = sub_alg(B, d_c)

        if previous_solution == [-1]:
            solution.append([objective_row_number])

        elif previous_solution:
            solution += [element + [objective_row_number] for element in previous_solution]

    return solution


def check_solvable(Given_matrix):
    def solve(A, d):
        if A.size == 0:
            return True

        # Sort columns by number of ones
        A = (A.T[np.argsort(A.sum(axis=0))]).T

        rows = A[A[:, 0] == 1]
        objective_row_numbers = d[A[:, 0] == 1]

        if rows.size == 0:
            return False

        for i, objective_row_number in zip(range(rows.shape[0]), objective_row_numbers):
            B, d_c = remove_intersections(A, d, rows, i)

            if solve(B, d_c):
                return True

        return False

    return solve(Given_matrix, np.arange(Given_matrix.shape[0]))
