import pygame


LENGTH = 800
WIDTH = 800
DIMENSION = 40
SCREEN_FACTOR = LENGTH / DIMENSION

SPEED = 500

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
BLUE = [0, 0, 255]
PURPLE = [255, 0, 255]
RED = [255, 0, 0]

COLORS = [BLUE, PURPLE, RED]

pygame.init()
screen = pygame.display.set_mode([LENGTH, WIDTH])
clock = pygame.time.Clock()

print("everything initialized")


def show_game_state(game):
    
    screen.fill([0, 255, 0])
    for racer in game.players:
        for i in range(game.dimension):
            for j in range(game.dimension):

                player_num = game.collision_table[i][j]
                x, y = i * SCREEN_FACTOR, j * SCREEN_FACTOR
                if player_num == 1:
                    color = COLORS[0]
                    screen.fill(color, pygame.Rect(x, y, SCREEN_FACTOR, SCREEN_FACTOR))

                if player_num == 2:
                    color = COLORS[1]
                    screen.fill(color, pygame.Rect(x, y, SCREEN_FACTOR, SCREEN_FACTOR))

    pygame.display.flip()
    clock.tick(SPEED)