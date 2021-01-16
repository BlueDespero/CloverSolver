import numpy as np


def Algorithm_X(Given_matrix):
    result = Sub_alg(Given_matrix, np.arange(Given_matrix.shape[0]))
    if result["found"]:
        return result["answer"]
    return "No solution found"


def Sub_alg(A, d):
    if A.size == 0:
        return dict(found=True, answer=[])

    # Sort columns by number of ones
    A = (A.T[np.argsort(A.sum(axis=0))]).T

    rows = A[A[:, 0] == 1]
    objective_row_numbers = d[A[:, 0] == 1]

    if rows.size == 0:
        return dict(found=False)

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

        solution = Sub_alg(B, d_c)

        if solution["found"]:
            solution["answer"].append(objective_row_number)
            return solution

    return solution


if __name__ == "__main__":
    '''
    np.random.seed(5)
    A_example = np.random.rand(9, 6) * 1.2
    A_example = np.floor(A_example)
    A_example = np.vstack([A_example, np.array([0., 0., 0., 0., 0., 1.])])
    '''
    wiki_example = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 1]
    ])

    print(wiki_example, "\n")
    print("Result: ", Algorithm_X(wiki_example))

    no_answer_example = np.array([
        [0, 0, 0, 0, 0, 0, 0],
    ])

    print(no_answer_example, "\n")
    print("Result: ", Algorithm_X(no_answer_example))
