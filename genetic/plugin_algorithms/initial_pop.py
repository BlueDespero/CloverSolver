import numpy as np


def uniform_initial_population(population_size, chromosome_size, initial_situation):
    population = []
    for i in range(population_size):
        pop = np.array([round(np.random.uniform(0, 1)) for _ in range(chromosome_size)])
        pop = np.bitwise_or(pop, initial_situation)
        population.append(pop)

    return population
