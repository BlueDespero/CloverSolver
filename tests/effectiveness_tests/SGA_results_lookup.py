from genetic.common import lambda_plus_mu
from genetic.crossover import single_point_crossover
from genetic.fitness import linear_fitness, get_individual_fitness
from genetic.ga_base import SGA
from genetic.initial_pop import uniform_initial_population
from genetic.mutation import reverse_bit_mutation
from tests.test_common import get_default_set, default_termination_condition

result, fitness = SGA(
    initial_population_generation=uniform_initial_population,
    population_size=10,
    number_of_children=6,
    fitness_function=linear_fitness,
    mutation_operator=reverse_bit_mutation,
    mutation_rate=0.5,
    crossover_operator=single_point_crossover,
    sets=get_default_set(),
    termination_condition=default_termination_condition,
    population_merge_function=lambda_plus_mu,
    iterations=501,
    lookup=True,
    lookup_top=5,
    lookup_every=100
)

#print(get_individual_fitness([0, 1, 0, 1, 0, 1], get_default_set(), linear_fitness))
