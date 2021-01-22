import numpy as np


def mutate_population(popultaion, mutation_operator, mutation_rate):
    for i, p in enumerate(popultaion):
        if np.random.uniform(0, 1) > mutation_rate:
            popultaion[i] = mutation_operator(p)

    return popultaion


def reverse_bit_mutation(individual):
    index = np.random.sample(range(individual.shape[0]))
    individual[index] ^= 1

    return individual
