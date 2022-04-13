import numpy as np
from numba import cuda
from copy import deepcopy
import pygame


class RandomArrayGenerator:
    def __init__(self, tile_density):
        self.TILE_DENSITY = tile_density

    def generate_array(self):
        return np.random.randint(low=0, high=255, size=(self.TILE_DENSITY, int((9 / 16) * self.TILE_DENSITY), 3))

    def on_click(self, mouse_x_index, mouse_y_index):
        pass
