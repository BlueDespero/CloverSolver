import numpy as np


def get_population_fitness(population, sets, function):
    return np.array([function(get_solution_cover(p, sets)) for p in population])


def get_individual_fitness(solution, sets, function):
    return function(get_solution_cover(solution, sets))


def get_solution_cover(individual, sets):
    cover = np.full(shape=sets.shape[1], fill_value=-1)
    for id, i in enumerate(individual):
        cover += sets[id] * i
    return cover


def linear_fitness(cover):
    cover = np.array([abs(v) for v in cover])
    return np.sum(cover)


def quadratic_fitness(cover):
    return np.sum(cover ** 2)
