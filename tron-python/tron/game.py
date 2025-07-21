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
        
        grid = np.zeros((num_rows, num_cols), dtype=bool)

        if obstacle_density > 0.0:
            # Calculate the total number of True values based on density
            num_obstacles = int(num_rows * num_cols * obstacle_density)
            
            # Randomly select indices to set to True
            true_indices = np.random.choice(num_rows * num_cols, num_obstacles, replace=False)
            
            # Convert flat indices to row, column coordinates and set corresponding cells to True
            grid.ravel()[true_indices] = True
            
        # Becomes self.players Tuple after Player objects are appended
        player_list = []

        starts = set()

        if random_starts:


            if not neutral_starts:
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
                                can_move=True,
                            )
                        )
                        i += 1
            # Neutral start (symmetric over horizontal, vertical, or diagonal axis)
            else:
                if num_players > 2:
                    raise NotImplementedError()
                
                retries = 0

                print(f"neutral starts implementation is kidna cringe.")

                # TODO: This retries stuff is jank. Also not sure if this is always 100% neutral
                while retries < 100:
                    player_list = []
                
                    rand_row = random.randrange(1, num_rows - 1)
                    rand_col = random.randrange(1, num_cols - 1)

                    player_list.append(
                        Player(
                            rand_row,
                            rand_col,
                            can_move=True,
                        )
                    )

                    c_row = (num_rows - 1) / 2
                    c_col = (num_cols - 1) / 2

                    def rotate_pos(row, col, angle):

                        if angle == 0:
                            return row, col
                        elif angle == 90:  # 90 degrees clockwise
                            row_new = int(c_col - (col - c_col))
                            col_new = int(row - c_row + c_row)
                        elif angle == 180:  # 180 degrees
                            row_new = int(2 * c_row - row)
                            col_new = int(2 * c_col - col)
                        elif angle == 270:  # 270 degrees clockwise
                            row_new = int(col - c_col + c_row)
                            col_new = int(c_row - (row - c_row))
                        else:
                            raise ValueError("Angle must be 90, 180, or 270 degrees")
                        return (row_new, col_new)

                    neutral_oppo_row = rand_row
                    neutral_oppo_col = rand_col

                    do_flip = random.random() > 0.5

                    if do_flip:
                        neutral_oppo_col = num_cols - rand_col
                        neutral_oppo_row, neutral_oppo_col = rotate_pos(neutral_oppo_row, neutral_oppo_col, random.choice([0, 90, 180, 270]))
                    else:
                        neutral_oppo_row, neutral_oppo_col = rotate_pos(neutral_oppo_row, neutral_oppo_col, random.choice([90, 180, 270]))

                    player_list.append(
                        Player(
                            neutral_oppo_row,
                            neutral_oppo_col,
                            can_move=True,
                        )
                    )
                    if not (neutral_oppo_row == rand_row and neutral_oppo_col == rand_col):
                        break
                    else:
                        retries += 1
                else:
                    raise RuntimeError("Extremely unlucky.")

        else:

            raise NotImplementedError("cringe.")

            if num_players == 2:
                default_starts = GameState.TWO_PLAYER_DEFAULT_STARTS
            else:
                raise NotImplementedError()

            for i in range(num_players):

                player_list.append(
                    Player(
                        row=int(default_starts[i].row_fraction * num_rows),
                        col=int(default_starts[i].col_fraction * num_cols),
                        can_move=True,
                    )
                )

        players = tuple(player_list)

        for player in players:
            grid[player.row, player.col] = True

        return GameState(grid, players)

@dataclass
class DefaultStart:
    row_fraction: float
    col_fraction: float
    direction: Direction


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
            and not game.grid[new_row, new_col]
        ):
            available_directions.append(dir)

    return available_directions




def from_players(players: tuple[Player], num_rows=10, num_cols=10) -> GameState:
    """
    Init game with pre-initialized players.
    """

    grid = np.zeros((num_rows, num_cols), dtype=bool)

    assert isinstance(players, tuple)

    for i in range(len(players)):

        p1 = players[i]

        assert isinstance(p1, Player)
        assert GameState.in_bounds(grid, p1.row, p1.col)
        assert p1.can_move

        for j in range(i + 1, len(players)):
            p2 = players[j]
            assert not (p1.row == p2.row and p1.col == p2.col)

    for player in players:
        grid[player.row, player.col] = True

    status = GameStatus.IN_PROGRESS

    return GameState(grid, players)



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





