import pygame

LENGTH = 800
WIDTH = 800
SPEED = 500

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
BLUE = [0, 0, 255]
PURPLE = [255, 0, 255]
RED = [255, 0, 0]

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
        init_utility_gui(game.dimension)
        gui_initialized = True

    screen.fill([0, 255, 0])
    for racer in game.players:
        for i in range(game.dimension):
            for j in range(game.dimension):

                player_num = game.collision_table[i][j]
                x, y = i * screen_factor, j * screen_factor
                if player_num == 1:
                    color = COLORS[0]
                    screen.fill(color, pygame.Rect(x, y, screen_factor, screen_factor))

                if player_num == 2:
                    color = COLORS[1]
                    screen.fill(color, pygame.Rect(x, y, screen_factor, screen_factor))

    for racer in game.players:
        headx, heady = racer.head[0] * screen_factor, racer.head[1] * screen_factor
        screen.fill(BLACK, pygame.Rect(headx, heady, screen_factor, screen_factor))

    pygame.event.pump()
    pygame.display.flip()
    clock.tick(20)

