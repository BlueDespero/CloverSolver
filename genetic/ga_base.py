import numpy as np

from genetic.common import roulette_wheel_selection
from genetic.common import sort_population_by_fitness
from genetic.crossover import crossover_population
from genetic.fitness import get_population_fitness
from genetic.mutation import mutate_population


def SGA(initial_population_generation,
        fitness_function,
        mutation_operator,
        crossover_operator,
        sets,
        termination_condition,
        population_merge_function,
        iterations=10000,
        population_size=100,
        number_of_children=50,
        mutation_rate=0.05
        ):
    best_solution, best_solution_fitness = 0, np.inf

    population = initial_population_generation(population_size, sets.shape[0])
    population_fitness = get_population_fitness(population=population, sets=sets, function=fitness_function)
    population_fitness, population = sort_population_by_fitness(population=population, fitness=population_fitness)

    for _ in range(iterations):
        children = crossover_population(population=population, fitness=population_fitness,
                                        crossover_operator=crossover_operator,
                                        selection_method=roulette_wheel_selection,
                                        number_of_children=number_of_children)
        children = mutate_population(popultaion=children, mutation_operator=mutation_operator,
                                     mutation_rate=mutation_rate)

        population = population_merge_function(population, children)
        population_fitness = get_population_fitness(population=population, sets=sets, function=fitness_function)
        population_fitness, population = sort_population_by_fitness(population=population, fitness=population_fitness)
        population_fitness, population = population_fitness[:population_size], population[:population_size]

        if best_solution_fitness > population_fitness[0]:
            best_solution_fitness = population_fitness[0]
            best_solution = population[0]

        if termination_condition(population_fitness):
            break

    return best_solution, best_solution_fitness
