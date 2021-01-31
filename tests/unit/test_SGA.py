import itertools
from unittest import TestCase

from ddt import ddt, data, unpack

from genetic.common import lambda_plus_mu, lambda_coma_mu
from genetic.plugin_algorithms.crossover import single_point_crossover, double_point_crossover, exchange_two_boxes_crossover, \
    exchange_two_columns_crossover, exchange_two_rows_crossover
from genetic.plugin_algorithms.fitness import linear_fitness, quadratic_fitness
from genetic.plugin_algorithms.ga_base import SGA
from genetic.plugin_algorithms.initial_pop import uniform_initial_population
from genetic.plugin_algorithms.mutation import reverse_bit_mutation, shuffle_row_mutation, shuffle_column_mutation, shuffle_box_mutation
from tests.test_common import default_termination_condition, get_default_sudoku_4x4, get_default_sudoku_9x9


@ddt
class TestSGA_Basic(TestCase):

    @data(*itertools.product([linear_fitness, quadratic_fitness], [single_point_crossover, double_point_crossover],
                             [lambda_plus_mu, lambda_coma_mu], [10, 50, 100]))
    @unpack
    def test_SGA_sudoku_4x4_basic(self, fitness_function, crossover_operator, population_merge_function, iterations):
        bug = False
        try:
            result, fitness = SGA(
                initial_population_generation=uniform_initial_population,
                fitness_function=fitness_function,
                mutation_operator=reverse_bit_mutation,
                crossover_operator=crossover_operator,
                initial_state=get_default_sudoku_4x4(),
                termination_condition=default_termination_condition,
                population_merge_function=population_merge_function,
                iterations=iterations
            )
        except:
            bug = True

        self.assertFalse(bug)

    @data(*itertools.product([linear_fitness, quadratic_fitness], [single_point_crossover, double_point_crossover],
                             [lambda_plus_mu, lambda_coma_mu], [10, 50, 100]))
    @unpack
    def test_SGA_sudoku_9x9_basic(self, fitness_function, crossover_operator, population_merge_function, iterations):
        bug = False
        try:
            result, fitness = SGA(
                initial_population_generation=uniform_initial_population,
                fitness_function=fitness_function,
                mutation_operator=reverse_bit_mutation,
                crossover_operator=crossover_operator,
                initial_state=get_default_sudoku_9x9(),
                termination_condition=default_termination_condition,
                population_merge_function=population_merge_function,
                iterations=iterations
            )
        except:
            bug = True

        self.assertFalse(bug)


@ddt
class TestSGA_PhenotypeCrossoverMutation(TestCase):

    @data(*itertools.product([linear_fitness, quadratic_fitness],
                             [exchange_two_boxes_crossover, exchange_two_columns_crossover,
                              exchange_two_rows_crossover],
                             [shuffle_column_mutation, shuffle_box_mutation, shuffle_row_mutation],
                             [lambda_plus_mu, lambda_coma_mu], [10]))
    @unpack
    def test_SGA_sudoku_4x4_phenotype(self, fitness_function, crossover_operator, mutation_operator,
                                      population_merge_function, iterations):
        bug = False
        try:
            result, fitness = SGA(
                initial_population_generation=uniform_initial_population,
                fitness_function=fitness_function,
                mutation_operator=mutation_operator,
                crossover_operator=crossover_operator,
                initial_state=get_default_sudoku_4x4(),
                termination_condition=default_termination_condition,
                population_merge_function=population_merge_function,
                iterations=iterations,
                # lookup=True
            )
        except:
            bug = True

        self.assertFalse(bug)

    @data(*itertools.product([linear_fitness, quadratic_fitness],
                             [exchange_two_boxes_crossover, exchange_two_columns_crossover,
                              exchange_two_rows_crossover],
                             [shuffle_column_mutation, shuffle_box_mutation, shuffle_row_mutation],
                             [lambda_plus_mu, lambda_coma_mu], [10, 50, 100]))
    @unpack
    def test_SGA_sudoku_9x9_phenotype(self, fitness_function, crossover_operator, mutation_operator,
                                      population_merge_function, iterations):
        bug = False
        try:
            result, fitness = SGA(
                initial_population_generation=uniform_initial_population,
                fitness_function=fitness_function,
                mutation_operator=mutation_operator,
                crossover_operator=crossover_operator,
                initial_state=get_default_sudoku_4x4(),
                termination_condition=default_termination_condition,
                population_merge_function=population_merge_function,
                iterations=iterations
            )
        except:
            bug = True

        self.assertFalse(bug)
