from game.tron import Direction

import pygame
from pygame.locals import *

LENGTH = 800
WIDTH = 800
SPEED = 500

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
BLUE = [14, 35, 173]
PURPLE = [255, 0, 255]
RED = [173, 3, 23]
GREEN = [3, 173, 49]

WALLS = [89, 89, 89]

COLORS = [BLUE, PURPLE, RED, BLACK]
#
# pygame.init()
# screen = pygame.display.set_mode([LENGTH, WIDTH])
# clock = pygame.time.Clock()

gui_initialized = False

def init_utility_gui(game_dimension):
    global screen_factor, screen, clock
    screen_factor = LENGTH / game_dimension
    pygame.init()
    screen = pygame.display.set_mode([LENGTH, WIDTH])
    clock = pygame.time.Clock()


def show_game_state(game):
    global gui_initialized

    if not gui_initialized:
        init_utility_gui(len(game.grid))
        gui_initialized = True

    screen.fill([25, 25, 25])
    for player in game.players:
        for row in range(len(game.grid)):
            for col in range(len(game.grid[0])):

                if game.grid[row][col]:
                    x, y = col * screen_factor, row * screen_factor
                    screen.fill(WALLS, pygame.Rect(x, y, screen_factor, screen_factor))

    for (i, player) in enumerate(game.players):

        color = GREEN if i == 0 else RED
        headx, heady = player.col * screen_factor, player.row * screen_factor
        screen.fill(color, pygame.Rect(headx, heady, screen_factor, screen_factor))

    pygame.event.pump()
    pygame.display.flip()
    clock.tick(20)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    return None
                if event.key == K_w:
                    return Direction.UP
                if event.key == K_a:
                    return Direction.LEFT
                if event.key == K_s:
                    return Direction.DOWN
                if event.key == K_d:
                    return Direction.RIGHT


