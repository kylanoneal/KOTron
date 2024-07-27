import random
from enum import Enum


class Racer:

    def __init__(self, head, direction, can_move):
        self.head = head
        assert isinstance(direction, Directions)
        self.direction = direction
        self.can_move = can_move


DIRECTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]]


class Directions(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


def get_readable_direction(direction: Directions):
    assert (isinstance(direction, Directions))
    if direction == Directions.up:
        return "UP"
    if direction == Directions.right:
        return "RIGHT"
    if direction == Directions.down:
        return "DOWN"
    if direction == Directions.left:
        return "LEFT"


def are_opposite_directions(d1, d2):
    new_direction = d1.value + 2
    new_direction = new_direction - 4 if new_direction > 3 else new_direction
    new_direction = Directions(new_direction)
    return new_direction == d2


class KOTron:

    def __init__(self, num_players=2, dimension=40, random_starts=False):

        self.num_players = 2
        self.dimension = dimension
        self.random_starts = random_starts

        self.new_game_state()



    def get_starting_positions(self, num_players, random_starts):

        starts = []

        if random_starts:
            i = 0
            while i < num_players:
                random_start = (random.randrange(1, self.dimension - 1), random.randrange(1, self.dimension-1))
                if random_start not in starts:
                    starts.append(random_start)
                    i += 1
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

        return [player.head for player in self.players]

    def get_directions(self):

        directions = []

        for racer in self.players:
            directions.append(DIRECTIONS[racer.direction.value])

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
        dx, dy = DIRECTIONS[racer.direction.value]
        return racer.head[0] + dx, racer.head[1] + dy

    def move_racers(self):

        for player_num, racer in enumerate(self.players):
            if racer.can_move:
                new_x, new_y = self.get_next_square(racer)
                has_collided = not self.is_in_bounds_and_empty(new_x, new_y)

                if has_collided:
                    racer.can_move = False
                else:
                    racer.head = (new_x, new_y)

        for player_num, racer in enumerate(self.players):
            if racer.can_move:
                head_x, head_y = racer.head
                if self.collision_table[head_x][head_y] != 0:
                    racer.can_move = False
                    self.players[self.collision_table[head_x][head_y] - 1].can_move = False

                self.collision_table[head_x][head_y] = player_num + 1

    def in_bounds(self, x, y):
        return 0 <= x < self.dimension and 0 <= y < self.dimension

    def is_in_bounds_and_empty(self, x, y):
        return self.in_bounds(x, y) and self.collision_table[x][y] == 0

    def update_direction(self, player_num, direction):

        assert (isinstance(direction, Directions))
        if not are_opposite_directions(self.players[player_num].direction, direction):
            self.players[player_num].direction = direction

    def get_possible_directions(self, player_num):
        available_directions = []
        head_x, head_y = self.players[player_num].head
        for i, (dx, dy) in enumerate(DIRECTIONS):
            new_x, new_y = head_x + dx, head_y + dy
            if self.is_in_bounds_and_empty(new_x, new_y):
                available_directions.append(Directions(i))

        return available_directions

    def check_for_winner(self):
        i = 0

        for racer in self.players:
            if racer.can_move:
                i += 1


        # More than 1 player can move, game continues
        if i > 1:
            return
        else:
            self.winner_found = True
            if i == 1:
                for player_num in range(len(self.players)):
                    if self.players[player_num].can_move:
                        self.winner_player_num = player_num
            # If 0 players can move, game is a tie
            else:
                self.winner_player_num = -1

    def new_game_state(self):

        self.collision_table = KOTron.build_collision_table(self.dimension)
        self.players = []
        self.build_racers(self.num_players, self.get_starting_positions(self.num_players, self.random_starts))
        self.winner_found = False
        self.winner_player_num = None


    def print_player_info(self):
        for i in range(len(self.players)):
            print("Player ", i + 1, " :")
            print("Head: ", self.players[i].head)
            print("Direction: ", self.players[i].direction)
            print("Player can move?  : ", self.players[i].can_move)

    def print_collision_table(self):
        for i in range(self.dimension):
            print(self.collision_table[i])
