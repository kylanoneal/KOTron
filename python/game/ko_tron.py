import random
from enum import Enum
import json


class Racer:

    def __init__(self, head, direction, can_move):
        self.head = head
        assert isinstance(direction, Directions)
        self.direction = direction
        self.can_move = can_move



class Directions(Enum):
    """(row, col) from top left"""
    up = (-1, 0)     
    right = (0, 1)   
    down = (1, 0)    
    left = (0, -1)   


def get_readable_direction(direction: Directions):
    assert isinstance(direction, Directions)
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
                random_start = (
                    random.randrange(1, self.dimension - 1),
                    random.randrange(1, self.dimension - 1),
                )
                if random_start not in starts:
                    starts.append(random_start)
                    i += 1
        else:

            half_dim = int(self.dimension * 0.5)
            if num_players == 2:
                starts.append((int(self.dimension * 0.25), half_dim))
                starts.append((int(self.dimension * 0.75), half_dim))

        return starts

    def build_racers(self, num_players, starts):

        for i in range(num_players):

            if i % 2 == 0:
                self.players.append(Racer(starts[i], Directions.right, True))
            else:
                self.players.append(Racer(starts[i], Directions.left, True))

            self.grid[starts[i][0]][starts[i][1]] = i + 1

    def get_heads(self):

        return [player.head for player in self.players]

    def get_directions(self):

        directions = []

        for racer in self.players:
            directions.append(DIRECTIONS[racer.direction.value])

        return directions

    @staticmethod
    def build_grid(dimension):

        grid = []

        for i in range(dimension):
            grid.append([])

            for j in range(dimension):
                grid[i].append(0)

        return grid

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
                if self.grid[head_x][head_y] != 0:
                    racer.can_move = False
                    self.players[self.grid[head_x][head_y] - 1].can_move = False

                self.grid[head_x][head_y] = player_num + 1

    def in_bounds(self, x, y):
        return 0 <= x < self.dimension and 0 <= y < self.dimension

    def is_in_bounds_and_empty(self, x, y):
        return self.in_bounds(x, y) and self.grid[x][y] == 0

    def update_direction(self, player_num, direction):

        assert isinstance(direction, Directions)
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

        self.grid = KOTron.build_grid(self.dimension)
        self.players = []
        self.build_racers(
            self.num_players,
            self.get_starting_positions(self.num_players, self.random_starts),
        )
        self.winner_found = False
        self.winner_player_num = None

    """
    Get a JSON string representation of the current game state
    """

    def to_json(self) -> str:

        player_list = []

        for i, player in enumerate(self.players):
            player_list.append(
                {
                    "player_num": i + 1,
                    "direction": player.direction.value,
                    "head_pos": player.head,
                }
            )

        json_dict = {
            "grid": self.grid,
            "players": player_list
        }

        return json.dumps(json_dict)

    def __repr__(self):
        repr_str = ""
        for y in range(self.dimension):
            for x in range(self.dimension):
                repr_str += f"{self.grid[x][y]} "
            repr_str += "\n"

        repr_str += (
            f"\n{'Player Num':<15}{'Head (x, y)':<15}{'Direction':<15}{'Can Move':<15}"
        )
        for i, player in enumerate(self.players):
            repr_str += f"\n{i:<15}{str(player.head):<15}{player.direction.name:<15}{str(player.can_move):<15}"

        return repr_str
