import numpy as np
import pygame

from cellular_automata.random_array_generator import RandomArrayGenerator
from cellular_automata.general_conv_3x3 import Conv3x3
from cellular_automata.convolutions import *

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = (9/16) * WINDOW_WIDTH
TILE_DENSITY = int(1920/2)


def mouse_position_index(position):
    x, y = position
    x_index = int((x / WINDOW_WIDTH) * TILE_DENSITY)
    y_index = int((y / WINDOW_HEIGHT) * TILE_DENSITY * (9/16))

    return x_index, y_index

def main_loop(array_generator):
    pygame.init()
    pygame.display.set_caption('Cellular Automata')
    pygame.display.set_icon(pygame.image.load('icon.png'))
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    mouse_down = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True

            if event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False

            if event.type == pygame.QUIT:
                running = False

        gridarray = array_generator.generate_array()

        if mouse_down:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_x_index, mouse_y_index = mouse_position_index((mouse_x, mouse_y))
            array_generator.on_click(mouse_x_index, mouse_y_index)

        surface = pygame.surfarray.make_surface(gridarray)
        surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(600)


if __name__ == '__main__':
    array_generator = Conv3x3(
        tile_density=TILE_DENSITY,
        initial_live_proportion=0.6,
        convolution=worms,
        colors=[[0, 0, 0], [255, 50, 0]],
        skip_odd=True
    )

    # array_generator = RandomArrayGenerator(tile_density=TILE_DENSITY)
    main_loop(array_generator=array_generator)
