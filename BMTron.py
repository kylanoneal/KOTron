import random
from enum import Enum


class Racer:

    def __init__(self, head, direction, can_move):
        self.head = head
        assert isinstance(direction, Directions)
        self.direction = direction
        self.can_move = can_move


class Directions(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


class BMTron:
    DIRECTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]]

    def __init__(self, num_players, random_starts=False):

        self.dimension = 40
        self.collision_table = BMTron.build_collision_table(self.dimension)

        self.players = []
        self.build_racers(num_players, self.get_starting_positions(num_players, random_starts))
        self.game_running = True
        # self.print_collision_table()
        # self.print_info()

    def get_starting_positions(self, num_players, random_starts):

        starts = []

        if random_starts:
            for i in range(num_players):
                starts.append([random.randrange(0, self.dimension), random.randrange(0, self.dimension)])
        else:

            half_dim = int(self.dimension * .5)
            if num_players == 2:
                starts.append((int(self.dimension * .25), half_dim))
                starts.append((int(self.dimension * .75), half_dim))

        return starts

    def build_racers(self, num_players, starts):

        for i in range(num_players):

            if i % 2 == 0:
                self.players.append(Racer(starts[i], Directions.right, True))
            else:
                self.players.append(Racer(starts[i], Directions.left, True))

            self.collision_table[starts[i][0]][starts[i][1]] = i + 1

    def get_heads(self):

        heads = []

        for racer in self.players:
            heads.append(racer.head)

        return heads

    def get_directions(self):

        directions = []

        for racer in self.players:
            directions.append(self.DIRECTIONS[racer.direction.value])

        return directions

    @staticmethod
    def build_collision_table(dimension):

        collision_table = []

        for i in range(dimension):
            collision_table.append([])

            for j in range(dimension):
                collision_table[i].append(0)

        return collision_table

    def get_next_square(self, racer):
        x = racer.head[0] + self.DIRECTIONS[racer.direction.value][0]
        y = racer.head[1] + self.DIRECTIONS[racer.direction.value][1]
        return x, y

    def move_racers(self):
        i = 0

        for racer in self.players:
            if racer.can_move:
                next_square = self.get_next_square(racer)
                # racer.head.append()

                collision_value = self.collision_table[next_square[0]][next_square[1]] \
                    if self.in_bounds(next_square) else 1

                if collision_value > 0:

                    racer.can_move = False


                else:

                    racer.head = next_square

                    self.collision_table[next_square[0]][next_square[1]] = i + 1

            i += 1

    def in_bounds(self, coords):
        return 0 <= coords[0] < self.dimension and 0 <= coords[1] < self.dimension

    def update_direction(self, player_num, direction):
        assert (isinstance(direction, Directions))
        if not direction is self.get_opposite_direction(self.players[player_num - 1].direction):
            self.players[player_num - 1].direction = direction

    def get_opposite_direction(self, direction):
        new_direction = direction.value + 2

        new_direction = new_direction - 4 if new_direction > 3 else new_direction

        return Directions(new_direction)

    def check_for_winner(self):
        i = 0

        for racer in self.players:

            if racer.can_move:
                i += 1

        if i == 1 or i == 0:
            self.game_running = False
            return True

        return False

    def print_info(self):
        print(self.players)
        for i in range(len(self.players)):
            print("Player ", i + 1, " :")
            print("Head: ", self.players[i].head)
            print("Direction: ", self.players[i].direction)

            print("Player can move?  : ", self.players[i].can_move)

    def print_collision_table(self):
        for i in range(self.dimension):
            print(self.collision_table[i])
