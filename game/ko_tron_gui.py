import pygame
import pickle
import random
from pathlib import Path
from typing import Optional
from AI.Bot import RandomBot, ReinforcementBot, TronBot
from game.ko_tron import KOTron, Directions
from copy import deepcopy


class KOTronGUI:
    ANIMATION_OFFSETS = [[0, 10], [-1, 0], [0, -1], [10, 0]]

    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]
    BLUE = [0, 0, 255]
    PURPLE = [255, 0, 255]
    RED = [255, 0, 0]

    COLORS = [BLUE, PURPLE, RED]

    LENGTH = 800
    WIDTH = 800
    game_speed = 500

    def __init__(self, game: KOTron, bot: Optional[TronBot]=None, game_speed=500, collect_trn_data=False):

        pygame.init()
        self.collect_trn_data = collect_trn_data
        # self.bot = ImitationBot(self.game) if is_bot else None
        self.game = game
        self.bot = bot
        self.game_speed = game_speed
        self.SCORES = [0, 0, 0, 0]
        self.COLORS = [self.PURPLE, self.BLUE, self.RED, self.WHITE]
        self.screen_factor = self.LENGTH / self.game.dimension
        self.screen = pygame.display.set_mode([self.LENGTH, self.WIDTH])
        self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.step_through = False

    def main(self):

        kylan_moves = []
        # self.background = pygame.image.load("IMAGES\\background.jpg")

        self.clock = pygame.time.Clock()
        self.score_updated = False

        go_again = True

        while (go_again):

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    go_again = False
                    break

                if event.type == pygame.KEYDOWN:
                    self.handle_keydown(event.key)

            if not self.game.winner_found:

                if self.collect_trn_data:
                    kylan_moves.append((deepcopy(self.game.collision_table),
                                        self.game.players[0].head,
                                        self.game.players[0].direction.value))

                if not self.step_through:
                    self.animate_move()
                    self.game.move_racers()
                    if self.bot is not None:
                        self.bot.bot_move()

                self.game.check_for_winner()

                if self.game.winner_found and not self.score_updated:
                    self.update_score()
                    self.score_updated = True

            self.clock_tick(self.game_speed)

        if self.collect_trn_data:
            raise NotImplementedError()
            #collect_feat(kylan_moves)

    def update_score(self):
        # Handle tie
        if self.game.winner_player_num != -1:
            self.SCORES[self.game.winner_player_num] += 1

    def handle_keydown(self, key):

        if key is pygame.K_w:
            self.game.update_direction(0, Directions.up)

        if key is pygame.K_a:
            self.game.update_direction(0, Directions.left)

        if key is pygame.K_s:
            self.game.update_direction(0, Directions.down)

        if key is pygame.K_d:
            self.game.update_direction(0, Directions.right)

        if key is pygame.K_i:
            self.game.update_direction(1, Directions.up)

        if key is pygame.K_j:
            self.game.update_direction(1, Directions.left)

        if key is pygame.K_k:
            self.game.update_direction(1, Directions.down)

        if key is pygame.K_l:
            self.game.update_direction(1, Directions.right)
        if key is pygame.K_t:
            self.game.update_direction(2, Directions.up)

        if key is pygame.K_f:
            self.game.update_direction(2, Directions.left)

        if key is pygame.K_g:
            self.game.update_direction(2, Directions.down)

        if key is pygame.K_h:
            self.game.update_direction(2, Directions.right)
        if key is pygame.K_r:
            self.restart()
        if key is pygame.K_e:
            self.reset_score()
        if key is pygame.K_z:
            self.step_through = not self.step_through
        if key is pygame.K_x:
            self.game.move_racers()
            self.bot.bot_move()
            self.draw_body()
        if key is pygame.K_c:
            outfile = open("SaveState", 'w+b')
            pickle.dump(self.game, outfile)
            outfile.close()
        if key is pygame.K_v:
            infile = open("SaveState", 'rb')
            self.game = pickle.load(infile)
            infile.close()
            self.bot = TronBot(self.game)

    def clock_tick(self, fps):
        pygame.display.flip()
        self.clock.tick(fps)

    # Change so no hardcoded values
    def restart(self):
        self.game = KOTron(2, dimension=10, random_starts=True)

        if self.bot:
            self.bot.game = self.game

        self.score_updated = False

    def reset_score(self):

        self.SCORES = [0, 0, 0, 0]

    def draw_body(self):

        for racer in self.game.players:
            for i in range(self.game.dimension):
                for j in range(self.game.dimension):

                    player_num = self.game.collision_table[i][j]
                    x, y = i * self.screen_factor, j * self.screen_factor
                    if player_num == 1:
                        color = self.COLORS[0]
                        self.screen.fill(color, pygame.Rect(x, y, self.screen_factor, self.screen_factor))

                    if player_num == 2:
                        color = self.COLORS[1]
                        self.screen.fill(color, pygame.Rect(x, y, self.screen_factor, self.screen_factor))

    def draw_score(self):

        locations = [[20, 20], [225, 20], [475, 20], [725, 20]]

        for i in range(3):
            self.screen.fill(self.COLORS[i], pygame.Rect(locations[i][0], locations[i][1],
                                                         self.LENGTH / 50, self.LENGTH / 50))
            score_string = str(self.SCORES[i])
            text = self.font.render(score_string, True, self.BLACK)
            self.screen.blit(text, [locations[i][0] + 20, locations[i][1]])

    def get_animation_sizes(self, directions):
        animation_sizes = []

        for racer in self.game.players:
            if racer.direction == 0 or racer.direction == 2:
                animation_sizes.append([10, 1])

            else:
                animation_sizes.append([1, 10])

        return animation_sizes

    def get_offsets(self):

        offsets = []

        for racer in self.game.players:
            offsets.append(self.ANIMATION_OFFSETS[racer.direction.value])

        return offsets

    def animate_move(self):

        # self.game.print_info()
        self.screen.fill([0, 255, 0])

        self.draw_body()
        self.draw_score()

        heads = self.multiply_coords_by_screen_factor(self.game.get_heads())
        directions = self.game.get_directions()
        animation_sizes = self.get_animation_sizes(directions)
        offsets = self.get_offsets()

        square_size = self.LENGTH / self.game.dimension

        animation_steps = 20

        for k in range(animation_steps):

            for i in range(len(self.game.players)):
                if self.game.players[i].can_move:

                    pixels_moved = int((k/animation_steps) * square_size)
                    newx, newy = [heads[i][0] + pixels_moved * directions[i][0], heads[i][1] + pixels_moved * directions[i][1]]

                    #color = self.COLORS[i]
                    color = [255, 0, 0]



                    rect_width = square_size if directions[i][0] == 0 else square_size / animation_steps
                    rect_height = square_size if directions[i][1] == 0 else square_size / animation_steps


                    # TODO: use directions Enum here
                    if directions[i][0] == -1:
                        offset_x = 0
                        offset_y = 0
                    elif directions[i][0] == 1:
                        offset_x = square_size - (square_size / animation_steps)
                        offset_y = 0
                    elif directions[i][1] == -1:
                        offset_x = 0
                        offset_y = 0
                    elif directions[i][1] == 1:
                        offset_x = 0
                        offset_y = -(square_size / animation_steps) + square_size


                    self.screen.fill(color, pygame.Rect(newx + offset_x, newy + offset_y,
                                                                          rect_width,
                                                                          rect_height))
                    #
                    # self.screen.fill(color, pygame.Rect(newx + offsets[i][0], newy + offsets[i][1],
                    #                                                       animation_sizes[i][0], animation_sizes[i][1]))

            self.clock_tick(self.game_speed)

    def get_random_color(self):
        return [random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)]

    def is_out_of_bounds(self, x, y):

        return True if (x < 0 or x > self.LENGTH or y < 0 or y > self.WIDTH) else False

    def multiply_coords_by_screen_factor(self, coords):

        new_coords = []

        for coord in coords:
            new_coords.append([coord[0] * self.screen_factor, coord[1] * self.screen_factor])

        return new_coords


if __name__ == "__main__":
    game = KOTron(2, dimension=10, random_starts=True)

    # Allows us to load any PyTorch model from 'model_architectures.py'
    import sys
    sys.path.append('../AI')

    model_path = Path("../AI/model_checkpoints/20240725_v1/909.pt")

    bot = ReinforcementBot(game, "../AI/model_checkpoints/20240725_v1/909")

    game = KOTronGUI(game, bot, game_speed=50, collect_trn_data=False)
    game.main()



