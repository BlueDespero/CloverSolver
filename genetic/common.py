import numpy as np
from copy import deepcopy


def sort_population_by_fitness(population, fitness):
    p = np.array(fitness.argsort())
    return np.array([fitness[i] for i in p]), np.array([population[i] for i in p])


def roulette_wheel_selection(set_o, weights, except_for=[]):
    set_c, weights_c = np.reshape(deepcopy(set_o), set_o.shape), np.reshape(deepcopy(weights), weights.shape)
    for e in except_for:
        del_indexes = np.where(set_c == e)
        set_c = np.delete(set_c, del_indexes)
        weights_c = np.delete(weights_c, del_indexes)

    w_sum = np.sum(weights_c)
    border = np.random.uniform(0, w_sum)
    current = 0
    for i, s in enumerate(set_c):
        current += weights_c[i]
        if current > border:
            return s


def lambda_coma_mu(parents, children):
    return children


def lambda_plus_mu(parents, children):
    return np.concatenate((parents, children))
