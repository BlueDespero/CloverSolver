import numpy as np


def sort_population_by_fitness(population, fitness):
    p = np.array(fitness.argsort())
    return np.array([fitness[i] for i in p]), np.array([population[i] for i in p])


def roulette_wheel_selection(set_o, weights, except_for=[]):
    for e in except_for:
        del_indexes = [i for i, s in enumerate(set_o) if list(s) == list(e)]
        set_o = np.delete(set_o, del_indexes, axis=0)
        weights = np.delete(weights, del_indexes)

    w_sum = np.sum(weights)
    border = np.random.uniform(0, w_sum)
    current = 0
    for i, s in enumerate(set_o):
        current += weights[i]
        if current > border:
            return s
    return except_for[0]


def lambda_coma_mu(parents, children):
    return children


def lambda_plus_mu(parents, children):
    return np.concatenate((parents, children))
