from typing import Optional

import pygame
import pickle
from Bot import *
from BMTron import *
from models import collect_feat
from copy import deepcopy

from models import DirectionNetConv



class BMTronGUI:
    ANIMATION_OFFSETS = [[0, 10], [-1, 0], [0, -1], [10, 0]]

    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]
    BLUE = [0, 0, 255]
    PURPLE = [255, 0, 255]
    RED = [255, 0, 0]

    COLORS = [BLUE, PURPLE, RED]

    LENGTH = 800
    WIDTH = 800
    SPEED = 500

    def __init__(self, game: BMTron, bot: Optional[TronBot], collect_trn_data=False):

        pygame.init()
        self.collect_trn_data = collect_trn_data
        # self.bot = ImitationBot(self.game) if is_bot else None
        self.game = game
        self.bot = bot
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
                    #print("APPENDING MOVE")
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

            self.clock_tick(self.SPEED) \
                # MOVE TO BMTRON GAME

        if self.collect_trn_data:
            collect_feat(kylan_moves)

    #CHANGE TO USE "GET_WINNER FUNCTION"
    def update_score(self):
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

    def restart(self):
        self.game = BMTron(2)

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

        for k in range(20):

            for i in range(len(self.game.players)):
                if self.game.players[i].can_move:
                    newx, newy = [heads[i][0] + k * directions[i][0], heads[i][1] + k * directions[i][1]]

                    self.screen.fill(self.get_random_color(), pygame.Rect(newx, newy,
                                                                          self.LENGTH / self.game.dimension,
                                                                          self.LENGTH / self.game.dimension))

                    self.screen.fill(self.get_random_color(), pygame.Rect(newx + offsets[i][0], newy + offsets[i][1],
                                                                          animation_sizes[i][0], animation_sizes[i][1]))

            self.clock_tick(self.SPEED)

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
    game = BMTron(2)
    bot = ReinforcementBot(game, "reinforcement_model.pth")

    game = BMTronGUI(game, bot, collect_trn_data=True)
    game.main()
