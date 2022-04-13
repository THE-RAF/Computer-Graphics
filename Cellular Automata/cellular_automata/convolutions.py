from numba import cuda
from math import *
import random


@cuda.jit
def pathways(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * 1
            convolution_sum += neighbour1 * 1
            convolution_sum += neighbour2 * 0
            convolution_sum += neighbour3 * 1
            convolution_sum += neighbour4 * 0
            convolution_sum += neighbour5 * 1
            convolution_sum += neighbour6 * 0
            convolution_sum += neighbour7 * 1
            convolution_sum += neighbour8 * 0

            convolution_sum = 1/(2**((convolution_sum-3.5)**2))

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def upper_wind(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (
                    colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * 1
            convolution_sum += neighbour1 * -0.1
            convolution_sum += neighbour2 * 0.3
            convolution_sum += neighbour3 * -0.3
            convolution_sum += neighbour4 * 0.2
            convolution_sum += neighbour5 * -0.1
            convolution_sum += neighbour6 * -0.5
            convolution_sum += neighbour7 * 0
            convolution_sum += neighbour8 * -0.1

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def stars(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (
                colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * 0.627
            convolution_sum += neighbour1 * -0.716
            convolution_sum += neighbour2 * 0.565
            convolution_sum += neighbour3 * -0.759
            convolution_sum += neighbour4 * 0.565
            convolution_sum += neighbour5 * -0.716
            convolution_sum += neighbour6 * 0.565
            convolution_sum += neighbour7 * -0.759
            convolution_sum += neighbour8 * 0.565

            convolution_sum = abs(convolution_sum)

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def worms(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (
                colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * -0.66
            convolution_sum += neighbour1 * -0.9
            convolution_sum += neighbour2 * 0.68
            convolution_sum += neighbour3 * -0.9
            convolution_sum += neighbour4 * 0.68
            convolution_sum += neighbour5 * -0.9
            convolution_sum += neighbour6 * 0.68
            convolution_sum += neighbour7 * -0.9
            convolution_sum += neighbour8 * 0.68

            convolution_sum = 1 - 1 / (2**(0.6*(convolution_sum**2)))

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def mold(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (
                colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * -0.233
            convolution_sum += neighbour1 * 0.787
            convolution_sum += neighbour2 * -0.349
            convolution_sum += neighbour3 * 0.787
            convolution_sum += neighbour4 * -0.349
            convolution_sum += neighbour5 * 0.787
            convolution_sum += neighbour6 * -0.349
            convolution_sum += neighbour7 * 0.787
            convolution_sum += neighbour8 * -0.349

            convolution_sum = 1 - 1 / (2**(0.6*(convolution_sum**2)))

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def waves(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (
                colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * 0.627
            convolution_sum += neighbour1 * -0.716
            convolution_sum += neighbour2 * 0.565
            convolution_sum += neighbour3 * -0.716
            convolution_sum += neighbour4 * 0.565
            convolution_sum += neighbour5 * -0.716
            convolution_sum += neighbour6 * 0.565
            convolution_sum += neighbour7 * -0.716
            convolution_sum += neighbour8 * 0.565

            convolution_sum = abs(1.2*convolution_sum)

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def emulsion(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * -0.059
            convolution_sum += neighbour1 * -0.465
            convolution_sum += neighbour2 * 0.585
            convolution_sum += neighbour3 * -0.465
            convolution_sum += neighbour4 * 0.585
            convolution_sum += neighbour5 * -0.465
            convolution_sum += neighbour6 * 0.585
            convolution_sum += neighbour7 * -0.465
            convolution_sum += neighbour8 * 0.585

            convolution_sum = 1 - exp(-convolution_sum)

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def pores(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            convolution_sum = cell * 0.319
            convolution_sum += neighbour1 * 0.972
            convolution_sum += neighbour2 * 0.407
            convolution_sum += neighbour3 * 0.972
            convolution_sum += neighbour4 * 0.407
            convolution_sum += neighbour5 * 0.972
            convolution_sum += neighbour6 * 0.407
            convolution_sum += neighbour7 * 0.972
            convolution_sum += neighbour8 * 0.407

            convolution_sum = 1 - exp(cos(convolution_sum))

            if convolution_sum < 0:
                convolution_sum = 0

            if convolution_sum > 1:
                convolution_sum = 1

            output_array[i, j] = value_to_rgb(convolution_sum)

@cuda.jit
def game_of_life(array, output_array, colors, greater_rgb_distance_index):
    def value_to_rgb(value):
        r = (colors[1][0] - colors[0][0]) * value + colors[0][0]
        g = (colors[1][1] - colors[0][1]) * value + colors[0][1]
        b = (colors[1][2] - colors[0][2]) * value + colors[0][2]

        return r, g, b

    def rgb_to_value(rgb):
        value = (rgb[greater_rgb_distance_index] - colors[0][greater_rgb_distance_index]) / (
                colors[1][greater_rgb_distance_index] - colors[0][greater_rgb_distance_index])

        return value

    i, j = cuda.grid(2)

    if i < output_array.shape[0] and j < output_array.shape[1]:
        if i == 0 or i == output_array.shape[0]-1 or j == 0 or j == output_array.shape[1]-1:
            output_array[i][j][0] = colors[0][0]
            output_array[i][j][1] = colors[0][1]
            output_array[i][j][2] = colors[0][2]

        else:
            cell = rgb_to_value(array[i][j])
            neighbour1 = rgb_to_value(array[i - 1][j])
            neighbour2 = rgb_to_value(array[i - 1][j + 1])
            neighbour3 = rgb_to_value(array[i][j + 1])
            neighbour4 = rgb_to_value(array[i + 1][j + 1])
            neighbour5 = rgb_to_value(array[i + 1][j])
            neighbour6 = rgb_to_value(array[i + 1][j - 1])
            neighbour7 = rgb_to_value(array[i][j - 1])
            neighbour8 = rgb_to_value(array[i - 1][j - 1])

            live_neighbours = 0
            live_neighbours += neighbour1
            live_neighbours += neighbour2
            live_neighbours += neighbour3
            live_neighbours += neighbour4
            live_neighbours += neighbour5
            live_neighbours += neighbour6
            live_neighbours += neighbour7
            live_neighbours += neighbour8

            if cell == 1:
                if live_neighbours < 2:
                    r, g, b = value_to_rgb(0)
                    output_array[i, j, 0] = r
                    output_array[i, j, 1] = g
                    output_array[i, j, 2] = b

                if live_neighbours == 2 or live_neighbours == 3:
                    r, g, b = value_to_rgb(1)
                    output_array[i, j, 0] = r
                    output_array[i, j, 1] = g
                    output_array[i, j, 2] = b

                if live_neighbours > 3:
                    r, g, b = value_to_rgb(0)
                    output_array[i, j, 0] = r
                    output_array[i, j, 1] = g
                    output_array[i, j, 2] = b

            else:
                if live_neighbours == 3:
                    r, g, b = value_to_rgb(1)
                    output_array[i, j, 0] = r
                    output_array[i, j, 1] = g
                    output_array[i, j, 2] = b
