import numpy as np

from genetic.common import sort_population_by_fitness, roulette_wheel_selection, sudoku_full_constraints_set, cube, \
    print_current_results
from genetic.plugin_algorithms.crossover import crossover_population
from genetic.plugin_algorithms.fitness import get_population_fitness
from genetic.plugin_algorithms.mutation import mutate_population


def SGA(initial_population_generation,
        fitness_function,
        mutation_operator,
        crossover_operator,
        initial_state,
        termination_condition,
        population_merge_function,
        iterations=10000,
        population_size=100,
        number_of_children=50,
        mutation_rate=0.05,
        lookup=False,
        lookup_every=0,
        lookup_top=5
        ):
    best_solution, best_solution_fitness = 0, np.inf

    fitness_record = []
    distance_record = []

    full_set_of_constraints = sudoku_full_constraints_set(round(cube(initial_state.shape[0])))

    population = initial_population_generation(population_size, initial_state.shape[0], initial_state)
    population_fitness = get_population_fitness(population=population, sets=full_set_of_constraints,
                                                function=fitness_function)
    population_fitness, population = sort_population_by_fitness(population=population, fitness=population_fitness)

    for i in range(iterations):
        children = crossover_population(population=population, fitness=population_fitness,
                                        crossover_operator=crossover_operator,
                                        selection_method=roulette_wheel_selection,
                                        number_of_children=number_of_children,
                                        initial_situation=initial_state)
        children = mutate_population(popultaion=children, mutation_operator=mutation_operator,
                                     mutation_rate=mutation_rate,
                                     initial_situation=initial_state)

        population = population_merge_function(population, children)
        population_fitness = get_population_fitness(population=population, sets=full_set_of_constraints,
                                                    function=fitness_function)
        population_fitness, population = sort_population_by_fitness(population=population, fitness=population_fitness)
        population_fitness, population = population_fitness[:population_size], population[:population_size]

        fitness_for_the_record = np.copy(population_fitness)
        fitness_record.append(
            [np.max(fitness_for_the_record), np.min(fitness_for_the_record), np.mean(fitness_for_the_record)])

        if best_solution_fitness > population_fitness[0]:
            best_solution_fitness = population_fitness[0]
            best_solution = population[0]

        if lookup and lookup_every != 0 and i % lookup_every == 0:
            print_current_results(iteration=i, best_solution=best_solution, best_solution_fitness=best_solution_fitness,
                                  population=population[:lookup_top],
                                  population_fitness=population_fitness[:lookup_top])

        if termination_condition(population_fitness):
            break

    if lookup:
        print_current_results(iteration=i, best_solution=best_solution, best_solution_fitness=best_solution_fitness,
                              population=population[:lookup_top], population_fitness=population_fitness[:lookup_top])

    return best_solution, best_solution_fitness, fitness_record
