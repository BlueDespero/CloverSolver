import itertools
from unittest import TestCase

from ddt import ddt, data, unpack

from genetic.common import lambda_plus_mu, lambda_coma_mu
from genetic.crossover import single_point_crossover, double_point_crossover
from genetic.fitness import linear_fitness, quadratic_fitness
from genetic.ga_base import SGA
from genetic.initial_pop import uniform_initial_population
from genetic.mutation import reverse_bit_mutation
from tests.test_common import get_default_set, default_termination_condition


@ddt
class TestSGA(TestCase):

    @data(*itertools.product([linear_fitness, quadratic_fitness], [single_point_crossover, double_point_crossover],
                             [lambda_plus_mu, lambda_coma_mu], [10, 50, 100]))
    @unpack
    def test_SGA_1(self, fitness_function, crossover_operator, population_merge_function, iterations):
        bug = False
        try:
            _ = SGA(
                initial_population_generation=uniform_initial_population,
                fitness_function=fitness_function,
                mutation_operator=reverse_bit_mutation,
                crossover_operator=crossover_operator,
                sets=get_default_set(),
                termination_condition=default_termination_condition,
                population_merge_function=population_merge_function,
                iterations=iterations
            )
        except:
            bug = True

        self.assertFalse(bug)
