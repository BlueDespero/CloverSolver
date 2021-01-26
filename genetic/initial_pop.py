import numpy as np


def uniform_initial_population(population_size, chromosome_size):
    population = []
    for i in range(population_size):
        pop = np.array([round(np.random.uniform(0, 1)) for _ in range(chromosome_size)])
        population.append(pop)

    return population
