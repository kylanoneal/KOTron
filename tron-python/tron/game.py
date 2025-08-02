import sys
import random
from enum import Enum, StrEnum, auto
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
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

@dataclass(frozen=True)
class Player:

    row: int
    col: int
    can_move: bool

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return (
            self.row == other.row
            and self.col == other.col
            and self.can_move == other.can_move
        )

    def __hash__(self):
        return hash(
            (hash(self.row), hash(self.col), hash(self.can_move))
        )


@dataclass(frozen=True)
class GameState:
    grid: np.ndarray[bool]
    players: tuple[Player]


    def __post_init__(self):
        # 1. Validate players container
        if not isinstance(self.players, tuple):
            raise TypeError(
                f"self.players must be a tuple of Player instances, "
                f"got {type(self.players).__name__}"
            )

        # 2. Validate grid type, dtype, and dimensions
        if not isinstance(self.grid, np.ndarray):
            raise TypeError(
                f"grid must be a numpy.ndarray, got {type(self.grid).__name__}"
            )
        if self.grid.dtype != bool:
            raise TypeError(
                f"grid.dtype must be bool, got {self.grid.dtype}"
            )
        if self.grid.ndim != 2:
            raise ValueError(
                f"grid must be 2-dimensional, got {self.grid.ndim} dimensions"
            )

        num_rows, num_cols = self.grid.shape

        # 3. Validate each player
        for idx, player in enumerate(self.players):
            # 3a. Type check
            if not isinstance(player, Player):
                raise TypeError(
                    f"Element {idx} of self.players must be Player, "
                    f"got {type(player).__name__}"
                )

            # 3b. Bounds check
            if not (0 <= player.row < num_rows):
                raise IndexError(
                    f"Player {idx} row index out of bounds: "
                    f"{player.row} not in [0, {num_rows - 1}]"
                )
            if not (0 <= player.col < num_cols):
                raise IndexError(
                    f"Player {idx} col index out of bounds: "
                    f"{player.col} not in [0, {num_cols - 1}]"
                )

            # 3c. Grid occupancy check
            if not self.grid[player.row, player.col]:
                raise ValueError(
                    f"grid at position ({player.row}, {player.col}) "
                    f"must be True for a player head"
                )
            

            for j in range(idx + 1, len(self.players)):
                pj: Player = self.players[j]

                if player.row == pj.row and player.col == pj.col:
                    if player.can_move or pj.can_move:
                        raise ValueError("Active players occupying same square")


    def __str__(self):

        repr_str = ""

        for row in self.grid.tolist():
            repr_str += str(row)
            repr_str += "\n"

        repr_str += "\n"

        for i, player in enumerate(self.players):
            repr_str += f"Player {i + 1}: ({player.row}, {player.col})"

        return repr_str
    
    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return (
            np.array_equal(self.grid, other.grid)
            and self.players == other.players
        )

    def __hash__(self):

        return hash((self.grid.tobytes(), self.players))


    @staticmethod
    def new_game(
        num_players: int = 2,
        num_rows: int = 10,
        num_cols: int = 10,
        random_starts: bool = False,
        neutral_starts: bool = False,
        obstacle_density: float = 0.0
    ) -> 'GameState':
        """
        Init game without pre-initialized players.
        """

        if obstacle_density > 0.8:
            raise ValueError("Too many obstacles.")
        
        if num_rows * num_cols < num_players:
            raise ValueError("Too many players for grid size.")

        grid = np.zeros((num_rows, num_cols), dtype=bool)

        if obstacle_density > 0.0:
            # Calculate the total number of True values based on density
            num_obstacles = int(num_rows * num_cols * obstacle_density)
            
            # Randomly select indices to set to True
            true_indices = np.random.choice(num_rows * num_cols, num_obstacles, replace=False)
            
            # Convert flat indices to row, column coordinates and set corresponding cells to True
            grid.ravel()[true_indices] = True


        if random_starts and not neutral_starts:

            random_starts_flat = np.random.choice(num_rows * num_cols, size=num_players, replace=False)
            random_rows, random_cols = np.unravel_index(random_starts_flat, grid.shape)
                
            players = tuple([Player(row, col, can_move=True) for row, col in zip(random_rows, random_cols)])

        elif random_starts and neutral_starts:
            # Neutral start (symmetric over horizontal, vertical, or diagonal axis)
            if num_players != 2:
                raise NotImplementedError()
        
            retries = 200
            for _ in range(retries):
                rand_row = random.randrange(0, num_rows)
                rand_col = random.randrange(0, num_cols)

                rot_grid = np.zeros_like(grid)
                rot_grid[rand_row][rand_col] = True


                do_flip = random.random() > 0.5
                rot_grid = np.fliplr(rot_grid) if do_flip else rot_grid
                
                n_rot_90 = random.randrange(0, 4) if do_flip else random.randrange(1, 4)
                rot_grid = np.rot90(rot_grid, k=n_rot_90)

                oppo_row, oppo_col = np.argwhere(rot_grid).squeeze()

                if not (rand_row == oppo_row and rand_col == oppo_col):
                    break

            else:
                raise RuntimeError(f"Neutral start not found after {retries}.")
            
            players = (Player(rand_row, rand_col, can_move=True), Player(oppo_row, oppo_col, can_move=True))

        else:
            # Default starts
            raise NotImplementedError()

        for player in players:
            grid[player.row, player.col] = True

        assert type(players) == tuple
        assert len(players) == num_players
        return GameState(grid, players)
    
    @staticmethod
    def from_players(players: tuple[Player], num_rows=10, num_cols=10) -> 'GameState':
        """
        Create game with pre-initialized players.
        """

        grid = np.zeros((num_rows, num_cols), dtype=bool)

        assert isinstance(players, tuple)

        for i in range(len(players)):

            p1 = players[i]

            assert isinstance(p1, Player)
            assert in_bounds(grid, p1.row, p1.col)
            assert p1.can_move

            for j in range(i + 1, len(players)):
                p2 = players[j]
                assert not (p1.row == p2.row and p1.col == p2.col)

        for player in players:
            grid[player.row, player.col] = True


        return GameState(grid, players)


class GameStatus(Enum):

    IN_PROGRESS = auto()
    TIE = auto()
    WINNER = auto()

@dataclass
class StatusInfo:
    status: GameStatus
    winner_index: Optional[int] = None


def get_status(game: GameState) -> StatusInfo:

    num_players_can_move = 0
    winner_index = None

    for i, player in enumerate(game.players):
        if player.can_move:
            num_players_can_move += 1
            winner_index = i

    if num_players_can_move == 0:
        return StatusInfo(GameStatus.TIE)
    elif num_players_can_move == 1:
        return StatusInfo(GameStatus.WINNER, winner_index)
    else:
        return StatusInfo(GameStatus.IN_PROGRESS)
    
def in_bounds(grid: np.ndarray, row: int, col: int):
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]


def get_possible_directions(game: GameState, player_index):
    available_directions = []
    player = game.players[player_index]

    for dir in Direction:

        dr, dc = dir.value
        new_row, new_col = player.row + dr, player.col + dc

        if (
            in_bounds(game.grid, new_row, new_col)
            and not game.grid[new_row][new_col]
        ):
            available_directions.append(dir)

    return available_directions


# TODO - should this be somewhere else?
# 10 mil in da cache
# @lru_cache(maxsize=int(1e7))
# def lru_cache_next(game: "GameState", direction_updates: DirectionUpdate):
#     return GameState.next(game, direction_updates)


@staticmethod
def next(game: GameState, directions: tuple[Direction]) -> GameState:

    assert len(directions) == len(game.players)

    next_grid = game.grid.copy()

    next_players = []
    for player, direction in zip(game.players, directions):
        next_row, next_col, next_can_move = player.row, player.col, player.can_move

        if player.can_move:
            dr, dc = direction.value
            new_row, new_col = player.row + dr, player.col + dc
            if (
                in_bounds(game.grid, new_row, new_col)
                and not game.grid[new_row, new_col]
            ):
                next_row, next_col = new_row, new_col
            else:
                next_can_move = False

        next_players.append(Player(next_row, next_col, next_can_move))

    # Update grid and handle case where 2 or more players try to occupy the same square
    for i in range(len(next_players)):

        pi: Player = next_players[i]
        next_grid[pi.row, pi.col] = True

        if pi.can_move:
            for j in range(i + 1, len(next_players)):
                pj: Player = next_players[j]

                if pj.can_move:
                    if pi.row == pj.row and pi.col == pj.col:
                        next_players[i] = Player(pi.row, pi.col, can_move=False)
                        next_players[j] = Player(pj.row, pj.col, can_move=False)

    return GameState(next_grid, tuple(next_players))


# @staticmethod
# def old_next(game: "GameState", direction_updates: tuple[DirectionUpdate]) -> "GameState":

#     assert game.status == GameStatus.IN_PROGRESS
#     assert isinstance(direction_updates, tuple)

#     # TODO: Make this quicker and avoid deepcopy somehow
#     next_game_state = deepcopy(game)

#     # NOTE: It would be nice to not have to modify any Player objects
#     # Instead, create the new objects with the updated direction
#     # NOTE: Should players even have a direction?
#     for dir_update in direction_updates:
#         next_game_state.players[dir_update.player_index].direction = (
#             dir_update.direction
#         )

#     for player in next_game_state.players:
#         if player.can_move:
#             dr, dc = player.direction.value
#             new_row, new_col = player.row + dr, player.col + dc
#             if (
#                 GameState.in_bounds(next_game_state.grid, new_row, new_col)
#                 and not next_game_state.grid[new_row, new_col]
#             ):
#                 player.row = new_row
#                 player.col = new_col
#             else:
#                 player.can_move = False

#     # Case where players attempt to occupy same square
#     for i in range(len(next_game_state.players)):

#         pi = next_game_state.players[i]

#         if pi.can_move:
#             for j in range(i + 1, len(next_game_state.players)):
#                 pj = next_game_state.players[j]

#                 if pj.can_move:
#                     if pi.row == pj.row and pi.col == pj.col:
#                         pi.can_move = False
#                         pj.can_move = False

#         next_game_state.grid[pi.row, pi.col] = True

#     next_game_state.status = GameState.get_status(next_game_state.players)

#     return next_game_state





