import random
from enum import Enum, StrEnum, auto
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


class Direction(Enum):
    """(row, col) from top left"""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @staticmethod
    def are_opposite_directions(d1: "Direction", d2: "Direction") -> bool:
        dr1, dc1 = d1.value
        dr2, dc2 = d2.value
        return (dr1 + dr2 == 0) and (dc1 + dc2 == 0)

    @staticmethod
    def get_random_direction() -> "Direction":
        return random.choice(list(Direction))


class Player:

    def __init__(self, row: int, col: int, direction: Direction, can_move: bool):
        self.row = row
        self.col = col
        self.direction = direction
        self.can_move = can_move

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return (
            self.row == other.row
            and self.col == other.col
            and self.direction == other.direction
            and self.can_move == other.can_move
        )

    def __hash__(self):
        return hash(
            (hash(self.row), hash(self.col), hash(self.direction), hash(self.can_move))
        )


class GameStatus(Enum):

    IN_PROGRESS = auto()
    TIE = auto()
    P1_WIN = auto()
    P2_WIN = auto()
    P3_WIN = auto()
    P4_WIN = auto()

    # TODO: Maybe this should be an instance method
    @staticmethod
    def index_of_winner(status: "GameStatus"):
        if status == GameStatus.P1_WIN:
            return 0
        if status == GameStatus.P2_WIN:
            return 1
        else:
            raise NotImplementedError()


@dataclass
class DirectionUpdate:
    direction: Direction
    player_index: int

    def __eq__(self, other):
        if not isinstance(other, DirectionUpdate):
            return False
        return (
            self.direction == other.direction
            and self.player_index == other.player_index
        )

    def __hash__(self):
        return hash((self.direction, self.player_index))


@dataclass
class DefaultStart:
    row_fraction: float
    col_fraction: float
    direction: Direction


class Tron:

    TWO_PLAYER_DEFAULT_STARTS = [
        DefaultStart(row_fraction=0.5, col_fraction=0.25, direction=Direction.RIGHT),
        DefaultStart(row_fraction=0.5, col_fraction=0.75, direction=Direction.LEFT),
    ]
    # TODO:
    # THREE_PLAYER_DEFAULT_STARTS = [...]
    # FOUR_PLAYER_DEFAULT_STARTS = [...]


    # TODO: Change to @classmethod, "with_random_starts" or something like that
    def __init__(self, num_players=2, num_rows=10, num_cols=10, random_starts=False):
        """
        Init game without pre-initialized players.
        """

        self.grid = np.zeros((num_rows, num_cols), dtype=bool)
        self.status = GameStatus.IN_PROGRESS

        # Becomes self.players Tuple after Player objects are appended
        player_list = []

        starts = set()
        if random_starts:
            i = 0
            while i < num_players:
                random_start = (
                    random.randrange(1, num_rows - 1),
                    random.randrange(1, num_cols - 1),
                )
                if random_start not in starts:
                    starts.add(random_start)
                    player_list.append(
                        Player(
                            random_start[0],
                            random_start[1],
                            direction=Direction.get_random_direction(),
                            can_move=True,
                        )
                    )
                    i += 1
        else:

            if num_players == 2:
                default_starts = Tron.TWO_PLAYER_DEFAULT_STARTS
            else:
                raise NotImplementedError()

            for i in range(num_players):

                player_list.append(
                    Player(
                        row=int(default_starts[i].row_fraction * num_rows),
                        col=int(default_starts[i].col_fraction * num_cols),
                        direction=default_starts[i].direction,
                        can_move=True,
                    )
                )

        self.players = tuple(player_list)

        for player in self.players:
            self.grid[player.row, player.col] = True

    
    # TODO: Change to @classmethod, "from players" or something like that
    # def __init__(self, players: tuple[Player], num_rows=10, num_cols=10):
    #     """
    #     Init game with pre-initialized players.
    #     """

    #     self.grid = np.zeros((num_rows, num_cols), dtype=bool)
    #     self.status = GameStatus.IN_PROGRESS

    #     assert isinstance(players, tuple)

    #     for i in range(len(players)):

    #         p1 = players[i]

    #         assert isinstance(p1, Player)
    #         assert Tron.in_bounds(self.grid, p1.row, p1.col)

    #         for j in range(i+1, len(players)):
    #             p2 = players[j]
    #             assert not (p1.row == p2.row and p1.col == p2.col)

    #     self.players = players

    #     for player in self.players:
    #         self.grid[player.row, player.col] = True

    def __eq__(self, other):
        if not isinstance(other, Tron):
            return False
        return (
            np.array_equal(self.grid, other.grid)
            and self.status == other.status
            and self.players == other.players
        )

    def __hash__(self):

        return hash((self.grid.tobytes(), self.status, self.players))

    @lru_cache(maxsize=None)
    def lru_cache_next(game: "Tron", direction_updates: DirectionUpdate):
        return Tron.next(game, direction_updates)

    @staticmethod
    def next(game: "Tron", direction_updates: tuple[DirectionUpdate]) -> "Tron":

        assert game.status == GameStatus.IN_PROGRESS
        assert isinstance(direction_updates, tuple)

        # TODO: Make this quicker and avoid deepcopy somehow
        next_game_state = deepcopy(game)

        # NOTE: It would be nice to not have to modify any Player objects
        # Instead, create the new objects with the updated direction
        # NOTE: Should players even have a direction?
        for dir_update in direction_updates:
            next_game_state.players[dir_update.player_index].direction = (
                dir_update.direction
            )

        for player in next_game_state.players:
            if player.can_move:
                dr, dc = player.direction.value
                new_row, new_col = player.row + dr, player.col + dc
                if (
                    Tron.in_bounds(next_game_state.grid, new_row, new_col)
                    and not next_game_state.grid[new_row, new_col]
                ):
                    player.row = new_row
                    player.col = new_col
                else:
                    player.can_move = False

        # Case where players attempt to occupy same square
        for i in range(len(next_game_state.players)):

            pi = next_game_state.players[i]

            if pi.can_move:
                for j in range(i + 1, len(next_game_state.players)):
                    pj = next_game_state.players[j]

                    if pj.can_move:
                        if pi.row == pj.row and pi.col == pj.col:
                            pi.can_move = False
                            pj.can_move = False

            next_game_state.grid[pi.row, pi.col] = True

        next_game_state.status = Tron.get_status(next_game_state.players)

        return next_game_state

    @staticmethod
    def in_bounds(grid: np.ndarray, row: int, col: int):
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

    @staticmethod
    def get_status(players: list[Player]):

        num_players_can_move = 0

        for player in players:
            if player.can_move:
                num_players_can_move += 1

        if num_players_can_move > 1:
            return GameStatus.IN_PROGRESS
        elif num_players_can_move == 0:
            return GameStatus.TIE

        elif num_players_can_move == 1:
            # Assuming 2 players only
            if players[0].can_move:
                return GameStatus.P1_WIN
            elif players[1].can_move:
                return GameStatus.P2_WIN
            else:
                raise NotImplementedError()

    @staticmethod
    def get_possible_directions(game: "Tron", player_index):
        available_directions = []
        player = game.players[player_index]

        for dir in Direction:

            dr, dc = dir.value
            new_row, new_col = player.row + dr, player.col + dc

            if (
                Tron.in_bounds(game.grid, new_row, new_col)
                and not game.grid[new_row, new_col]
            ):
                available_directions.append(dir)

        return available_directions

    # TODO: Where should this live?
    # """
    # Get a JSON string representation of the current game state
    # """

    # def to_json(self) -> str:

    #     player_list = []

    #     for i, player in enumerate(self.players):
    #         player_list.append(
    #             {
    #                 "player_num": i + 1,
    #                 "direction": player.direction.value,
    #                 "head_pos": player.head,
    #             }
    #         )

    #     json_dict = {"grid": self.grid, "players": player_list}

    #     return json.dumps(json_dict)
