import numpy as np
from numba import cuda
from copy import deepcopy
from math import *
import pygame


class Conv3x3:
    def __init__(self, tile_density, initial_live_proportion, convolution, colors, skip_odd=False):
        self.TILE_DENSITY = tile_density
        self.skip_odd = skip_odd

        self.convolution = convolution

        self.colors = np.array(colors)
        self.greater_rgb_distance_index = np.argmax([self.colors[1][i] - self.colors[0][i] for i in range(3)])

        self.game_array = np.random.choice(
            [0, 1],
            p=[1-initial_live_proportion, initial_live_proportion],
            size=(self.TILE_DENSITY, int((9 / 16) * self.TILE_DENSITY)),
        )

        game_array_r = deepcopy(self.game_array)
        game_array_r[game_array_r == 0] = self.colors[0][0]
        game_array_r[game_array_r == 1] = self.colors[1][0]

        game_array_g = deepcopy(self.game_array)
        game_array_g[game_array_g == 0] = self.colors[0][1]
        game_array_g[game_array_g == 1] = self.colors[1][1]

        game_array_b = deepcopy(self.game_array)
        game_array_b[game_array_b == 0] = self.colors[0][2]
        game_array_b[game_array_b == 1] = self.colors[1][2]

        self.game_array = np.stack((game_array_r, game_array_g, game_array_b), axis=2)
        self.plot_array = self.game_array

        self.skip_index = 0

    def step(self):
        grid_array = cuda.to_device(self.game_array)
        out_array = cuda.to_device(self.game_array)
        colors = cuda.to_device(self.colors)
        greater_rgb_distance_index = cuda.to_device(self.greater_rgb_distance_index)

        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(out_array.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(out_array.shape[1] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        self.convolution[blocks_per_grid, threads_per_block](grid_array, out_array, colors, greater_rgb_distance_index)

        self.game_array = out_array.copy_to_host()

    def value_to_rgb(self, value):
        r = (self.colors[1][0] - self.colors[0][0]) * value + self.colors[0][0]
        g = (self.colors[1][1] - self.colors[0][1]) * value + self.colors[0][1]
        b = (self.colors[1][2] - self.colors[0][2]) * value + self.colors[0][2]

        return r, g, b

    def on_click(self, mouse_x_index, mouse_y_index):
        self.game_array[mouse_x_index][mouse_y_index] = self.value_to_rgb(1)
        self.game_array[mouse_x_index+1][mouse_y_index] = self.value_to_rgb(1)
        self.game_array[mouse_x_index][mouse_y_index+1] = self.value_to_rgb(1)
        self.game_array[mouse_x_index+1][mouse_y_index+1] = self.value_to_rgb(1)
        self.game_array[mouse_x_index-1][mouse_y_index] = self.value_to_rgb(1)
        self.game_array[mouse_x_index][mouse_y_index-1] = self.value_to_rgb(1)
        self.game_array[mouse_x_index-1][mouse_y_index-1] = self.value_to_rgb(1)
        self.game_array[mouse_x_index+1][mouse_y_index-1] = self.value_to_rgb(1)
        self.game_array[mouse_x_index-1][mouse_y_index+1] = self.value_to_rgb(1)

    def generate_array(self):
        self.step()
        if self.skip_odd:
            if self.skip_index == 0:
                self.plot_array = self.game_array
                self.skip_index = 1

            else:
                self.skip_index = 0

        else:
            self.plot_array = self.game_array

        return self.plot_array
