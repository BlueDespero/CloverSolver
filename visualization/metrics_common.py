import itertools


def list_conjunction(A, B):
    return [a & b for a, b in zip(A, B)]


def list_alternative(A, B):
    return [a | b for a, b in zip(A, B)]


def jaccard_distance(A, B):
    return 1 - (sum(list_conjunction(A, B)) / sum(list_alternative(A, B)))


def dice_distance(A, B):
    return 1 - (2 * sum(list_conjunction(A, B)) / (sum(A) + sum(B)))


def get_population_distance(population, distance_function):
    all_pairs = itertools.product(population, population)
    distance_sum = sum([distance_function(a, b) for a, b in all_pairs])
    return distance_sum
