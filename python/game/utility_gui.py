from typing import Optional
from game.tron import Direction
from ai.minimax import MinimaxDebugState

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


def show_game_state(game, minimax_debug_state: Optional[MinimaxDebugState] = None):
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

    for i, player in enumerate(game.players):

        color = GREEN if i == 0 else RED
        headx, heady = player.col * screen_factor, player.row * screen_factor
        screen.fill(color, pygame.Rect(headx, heady, screen_factor, screen_factor))

    if minimax_debug_state is not None:
        show_minimax_debug_info(minimax_debug_state)

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


def show_minimax_debug_info(minimax_debug_state: MinimaxDebugState):
    font = pygame.font.Font(None, 36)  # 'None' uses the default font; size is 36

    # Render the text
    text = font.render(
        f"Maximizing Player?  {minimax_debug_state.is_maximizing_player}",
        True,
        WHITE,
    )  # True enables anti-aliasing
    # Get the rectangle for positioning
    text_rect = text.get_rect(center=(400, 80))  # Centered at (400, 300)
    # Draw the text
    screen.blit(text, text_rect)

    # Render the text
    text = font.render(
        f"Maximizing Player Move:  {minimax_debug_state.maximizing_player_move}",
        True,
        WHITE,
    )  # True enables anti-aliasing
    # Get the rectangle for positioning
    text_rect = text.get_rect(center=(400, 110))  # Centered at (400, 300)
    # Draw the text
    screen.blit(text, text_rect)

    # Render the text
    text = font.render(
        f"Depth:  {minimax_debug_state.depth}, Alpha:  {minimax_debug_state.alpha}, Beta:  {minimax_debug_state.beta}",
        True,
        WHITE,
    )  # True enables anti-aliasing

    # Get the rectangle for positioning
    text_rect = text.get_rect(center=(400, 50))  # Centered at (400, 300)

    # Draw the text
    screen.blit(text, text_rect)
