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
        mutation_rate=0.05,
        lookup=False,
        lookup_every=100,
        lookup_top=5
        ):
    best_solution, best_solution_fitness = 0, np.inf

    population = initial_population_generation(population_size, sets.shape[0])
    population_fitness = get_population_fitness(population=population, sets=sets, function=fitness_function)
    population_fitness, population = sort_population_by_fitness(population=population, fitness=population_fitness)

    for i in range(iterations):
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

        #TODO: make this log into a function and use logger
        if lookup and i % lookup_every == 0:
            print("Iteration {} results".format(i))
            print("Best solution {s}  |  Best fitness {f}".format(s=best_solution, f=best_solution_fitness))
            for j in range(1, lookup_top + 1):
                print("    {iter}: solution {s} | fitness {f}".format(iter=i, s=population[j], f=population_fitness[j]))
            print("############################")

        if termination_condition(population_fitness):
            if lookup:
                print("Iteration {} results".format(i))
                print("Best solution {s}  |  Best fitness {f}".format(s=best_solution, f=best_solution_fitness))
                for j in range(1, lookup_top + 1):
                    print("    {iter}: solution {s} | fitness {f}".format(iter=i, s=population[j], f=population_fitness[j]))
                print("############################")
            break

    return best_solution, best_solution_fitness
