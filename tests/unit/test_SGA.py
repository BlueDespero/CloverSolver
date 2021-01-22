from unittest import TestCase

from genetic.common import lambda_plus_mu
from genetic.crossover import single_point_crossover
from genetic.fitness import linear_fitness
from genetic.ga_base import SGA
from genetic.initial_pop import uniform_initial_population
from genetic.mutation import reverse_bit_mutation
from tests.test_common import get_default_set, default_termination_condition


class TestSGA(TestCase):

    def test_SGA_1(self):
        result = SGA(
            initial_population_generation=uniform_initial_population,
            fitness_function=linear_fitness,
            mutation_operator=reverse_bit_mutation,
            crossover_operator=single_point_crossover,
            sets=get_default_set(),
            termination_condition=default_termination_condition,
            population_merge_function=lambda_plus_mu
        )

        print(result)
        self.assertTrue(True)
