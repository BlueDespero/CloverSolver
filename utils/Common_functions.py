import numpy as np



def fitness_function(chromosome, evaluation_matrix,
                     model=lambda once, more: more - once):
    # input:
    # chromosome = 1-D array(n)
    # evaluation_matrix = np.array(n,m) of zeros and ones
    # model = lambda int, int: int; also float type numbers will work well
    # model - describes dependency of covered elements and covered more than once on fitness function
    # for default model minimum of fitness function is always zero

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


def ids_to_binary_list(ids, length):
    # convert list of number to binary chromosome
    output = np.zeros(length)
    for id in ids:
        output[id] = 1
    return output


def binary_list_to_ids(binary_list):
    # binary chromosome to convert list of number
    output = []
    for i, j in enumerate(binary_list):
        if j:
            output.append(i)
    return np.array(output)