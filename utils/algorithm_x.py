import numpy as np


def remove_intersections(sets, sets_ids, selected_rows, id):
    sets_c = sets.copy()
    d_c = sets_ids.copy()
    for j in np.arange(sets_c.shape[1])[selected_rows[id] == 1][::-1]:
        # removing each row which intersection with rows[i] is not empty
        k = sets_c[:, j] == 0
        sets_c = sets_c[k]
        d_c = d_c[k]
        # removing each column that rows[i] covers
        sets_c = np.delete(sets_c, j, axis=1)

    return sets_c, d_c


def algorithm_x(given_matrix):
    def sub_alg(sets, d):
        if sets.size == 0:
            return [-1]

        # Sort columns by number of ones
        sets = (sets.T[np.argsort(sets.sum(axis=0))]).T

        rows = sets[sets[:, 0] == 1]
        objective_row_numbers = d[sets[:, 0] == 1]

        if rows.size == 0:
            return []

        solution = []
        for i, objective_row_number in zip(range(rows.shape[0]), objective_row_numbers):
            sets_c, d_c = remove_intersections(sets, d, rows, i)

            previous_solution = sub_alg(sets_c, d_c)

            if previous_solution == [-1]:
                solution.append([objective_row_number])

            elif previous_solution:
                solution += [element + [objective_row_number] for element in previous_solution]

        return solution

    return sub_alg(given_matrix, np.arange(given_matrix.shape[0]))


def check_solvable(given_matrix):
    def solve(sets, d):
        if sets.size == 0:
            return True

        # Sort columns by number of ones
        sets = (sets.T[np.argsort(sets.sum(axis=0))]).T

        rows = sets[sets[:, 0] == 1]
        objective_row_numbers = d[sets[:, 0] == 1]

        if rows.size == 0:
            return False

        for i, objective_row_number in zip(range(rows.shape[0]), objective_row_numbers):
            sets_c, d_c = remove_intersections(sets, d, rows, i)

            if solve(sets_c, d_c):
                return True

        return False

    return solve(given_matrix, np.arange(given_matrix.shape[0]))