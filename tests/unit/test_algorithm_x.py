from unittest import TestCase

import numpy as np

from utils.algorithm_x import algorithm_x, check_solvable


class TestAlgorithmX(TestCase):

    def test_wiki_example(self):
        example = np.array([
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1]
        ])

        self.assertEqual(algorithm_x(example), [[1, 3, 5]])

    def test_no_answer_example(self):
        example = np.array([
            [0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertEqual(algorithm_x(example), [])

    def test_multiple_answer_example(self):
        example = np.array([
            [1, 1],
            [1, 1]
        ])

        self.assertEqual(algorithm_x(example), [[0], [1]])


class TestCheckSolvable(TestCase):

    def test_wiki_example(self):
        example = np.array([
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1]
        ])

        self.assertEqual(True, check_solvable(example))

    def test_no_answer_example(self):
        example = np.array([
            [0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertEqual(False, check_solvable(example))

    def test_multiple_answer_example(self):
        example = np.array([
            [1, 1],
            [1, 1]
        ])

        self.assertEqual(True, check_solvable(example))
