import numpy as np
from copy import deepcopy


def crossover_population(population, fitness, crossover_operator, selection_method, number_of_children):
    children_set = []
    for i in range(0, number_of_children, 2):
        parent_1 = selection_method(set_o=deepcopy(population), weights=deepcopy(fitness))
        parent_2 = selection_method(set_o=deepcopy(population), weights=deepcopy(fitness), except_for=[parent_1])

        child_1, child_2 = crossover_operator(parent_1, parent_2)
        children_set += [child_1, child_2]

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
