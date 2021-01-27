import numpy as np

from utils.Sudoku_transcription import grid_from_coordinates


def print_sudoku(grid=np.empty([0, 0]), chromosome=None, coordinates=None):
    # input optional:
    # 1.
    # grid = np.array([n,n]) full of intigers from 1 to n or 0 representing empty square
    # 2.
    # solution = np.array([n**2]) or list - list of numbers of rows of transcription matrix
    # coordinates = np.array([m,3]) - each row [x,y,z] is list of coordinates (x,y) of value z

    if grid.size > 0:
        for y in range(grid.shape[1]):
            s = ""
            for x in range(grid.shape[0]):
                if grid[y, x] != 0:
                    s += str(int(grid[y, x]))
                else:
                    s += "_"
                if x % np.sqrt(grid.shape[0]) == np.sqrt(grid.shape[0]) - 1 and x != grid.shape[0] - 1:
                    s += "|"
            if y % np.sqrt(grid.shape[1]) == np.sqrt(grid.shape[1]) - 1 and y != grid.shape[1] - 1:
                s += "\n" + (int(np.sqrt(grid.shape[0])) * "-" + "+") * int(np.sqrt(grid.shape[0]) - 1) + int(
                    np.sqrt(grid.shape[0])) * "-"
            print(s)
    else:
        print_sudoku(grid=grid_from_coordinates(chromosome, coordinates))
