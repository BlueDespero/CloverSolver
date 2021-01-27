import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.widgets import Slider, Button

from utils.Sudoku_transcription import sudoku_matrix_representation
from utils.algorithm_x import remove_intersections, algorithm_x_first_solution


def binary_list_to_ids(binary_list):
    output = []
    for i, j in enumerate(binary_list):
        if j:
            output.append(i)
    return np.array(output)


def ids_to_binary_list(ids, length):
    output = np.zeros(length)
    for id in ids:
        output[id] = 1
    return output


def observe_number_of_possible_rows(numbers_of_rows, transcription_matrix):
    queue = numbers_of_rows.copy()
    np.random.shuffle(queue)

    t_matrix = transcription_matrix.copy()

    number_of_possibilities = []

    while queue.size > 0:
        size = t_matrix.shape[0]
        number_of_possibilities.append(size)
        chromosome = ids_to_binary_list(queue, size)

        t_matrix, chromosome = remove_intersections(t_matrix, chromosome, t_matrix, queue[0])

        queue = binary_list_to_ids(chromosome)

    return np.array(number_of_possibilities)


def cumulate_data(size_of_sample, size_of_sudoku):
    coordinates, transcription_matrix = sudoku_matrix_representation(np.zeros([size_of_sudoku, size_of_sudoku]))
    solution = np.array(algorithm_x_first_solution(transcription_matrix))
    temp = observe_number_of_possible_rows(solution, transcription_matrix)
    initial = temp[0]
    current_stage = np.empty(temp.shape[0])
    for k, value in enumerate(temp):
        current_stage[k] = value / (initial - k)
    number_of_possibilities = current_stage
    for i in tqdm(range(size_of_sample)):
        solution = np.array(algorithm_x_first_solution(transcription_matrix))
        temp = observe_number_of_possible_rows(solution, transcription_matrix)
        initial = temp[0]
        data = np.empty(temp.shape[0])
        for k, value in enumerate(temp):
            data[k] = value / (initial - k)

        current_stage += data

        number_of_possibilities = np.vstack([number_of_possibilities, current_stage / (i + 2)])

    return number_of_possibilities


def slider_demo(size_of_sample, size_of_sudoku=9):
    number_of_possibilities = cumulate_data(size_of_sample, size_of_sudoku)
    f = lambda i: number_of_possibilities[int(i)]
    x = np.linspace(0, 1, number_of_possibilities.shape[1])
    t = np.arange(number_of_possibilities.shape[1])

    # Define initial parameters
    init = 1

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = plt.plot(x, f(1), lw=2)
    ax.set_xlabel('Percent of filling a sudoku')
    ax.set_ylabel('Probability')

    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    freq_slider = Slider(
        ax=axfreq,
        label='Size of sample',
        valmin=1,
        valmax=number_of_possibilities.shape[0] - 1,
        valinit=init,
        valfmt="%1i"
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(f(freq_slider.val))
        fig.canvas.draw_idle()

    # register the update function with each slider
    freq_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        freq_slider.reset()

    button.on_clicked(reset)

    plt.show()
