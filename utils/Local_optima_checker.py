import numpy as np
import matplotlib.pyplot as plt

from utils.Sudoku_transcription import *

def Local_optima_checker(size=4):
    for i in range(100):
        grid = sudoku_generator(size)
        # grid = np.array([[0, 0, 0, 0],
        #                  [0, 0, 1, 3],
        #                  [4, 0, 0, 0],
        #                  [0, 0, 0, 1]])
        coordinates, transcription_matrix = sudoku_matrix_representation(grid)

        solution = algorithm_x_first_solution(transcription_matrix)
        if len(solution)!=16:
            print(solution)
        # full_grid = grid_from_coordinates(chromosome=solution,coordinates=coordinates,size_of_grid=size)

if __name__=="__main__":
    np.random.seed(1)
    Local_optima_checker()
    print(np.zeros(3))
    # sol = [49, 44, 39, 35, 32, 29, 26, 25, 24, 23, 17, 13, 11, 4, 2]
    # [[1. 3. 2. 4.]
    #  [2. 4. 1. 3.]
    #  [4. 2. 3. 1.]
    #  [3. 1. 4. 0.]]
    #
    # [[1. 3. 2. 4.]
    #  [2. 4. 1. 3.]
    #  [4. 1. 3. 2.]
    #  [3. 2. 4. 1.]]