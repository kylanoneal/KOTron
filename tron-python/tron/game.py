import numpy as np

from typing import Optional
from dataclasses import dataclass

from tron.game_2d import GameState2D, Player2D
from tron.enums import Direction, GameStatus


MAX_ROWS = 50
MAX_COLS = 50

BIT_MASKS = [1 << i for i in range(MAX_ROWS * MAX_COLS)]


@dataclass(frozen=True)
class Player:
    idx: int
    can_move: bool


@dataclass(frozen=True)
class GameState:
    num_rows: int
    num_cols: int
    board: int
    players: tuple[Player]

    
@dataclass
class PovGameState:
    game_state: GameState
    hero_index: int
    opponent_index: int

    # def __eq__(self, other):
    #     if not isinstance(other, PovGameState):
    #         return False
    #     return self.hero_index == other.hero_index and self.game_state == other.game_state

    # def __hash__(self):

    #     return hash((self.game_state, self.hero_index))



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

def get_bit(board: int, idx: int) -> bool:
    return (board & BIT_MASKS[idx]) != 0

def get_wall_indices(game_state: GameState) -> list[int]:
    return [i for i in range(game_state.num_rows * game_state.num_cols) if (get_bit(game_state.board, i))]


def get_next_position(game: GameState, player_index: int, direction: Direction) -> tuple[int, bool]:

    player = game.players[player_index]

    oob = False

    if direction == Direction.UP:
        if player.idx < game.num_cols:
            oob = True

        next_idx = player.idx - game.num_cols

    elif direction == Direction.DOWN:
        if player.idx >= game.num_cols * (game.num_rows - 1):
            oob = True

        next_idx = player.idx + game.num_cols

    elif direction == Direction.LEFT:
        if player.idx % game.num_cols == 0:
            oob = True

        next_idx = player.idx - 1

    elif direction == Direction.RIGHT:
        if player.idx % game.num_cols == game.num_cols - 1:
            oob = True

        next_idx = player.idx + 1

    else:
        raise ValueError("Invalid direction.")

    return next_idx, oob


def get_next_player(
    game:GameState, player_index: int, direction: Direction
) -> Player:

    player = game.players[player_index]

    next_idx, next_can_move = player.idx, player.can_move

    if player.can_move:

        _new_idx, _oob = get_next_position(game, player_index, direction)

        if _oob:
            next_can_move = False
        else:
            next_idx = _new_idx

    if next_can_move:
        if get_bit(game.board, next_idx):
            next_can_move = False
            next_idx = player.idx

    return Player(next_idx, next_can_move)


def next(game: GameState, directions: tuple[Direction]) -> GameState:

    next_players = tuple(
        get_next_player(game, i, d) for i, d in enumerate(directions)
    )

    next_board = game.board

    # Update board and handle case where 2 or more players try to occupy the same square
    for i in range(len(next_players)):

        pi: Player = next_players[i]
        next_board = next_board | BIT_MASKS[pi.idx]

        if pi.can_move:
            for j in range(i + 1, len(next_players)):
                pj: Player = next_players[j]

                if pj.can_move:
                    if pi.idx == pj.idx:

                        next_players = tuple(
                            Player(
                                p.idx, can_move=False if p.idx == pi.idx else p.can_move
                            )
                            for p in next_players
                        )
                        break

    return GameState(game.num_rows, game.num_cols, next_board, next_players)


def get_possible_directions(
    game: GameState,
    player_index: int,
):

    available_directions = []

    for dir in Direction:

        new_idx, is_oob = get_next_position(game, player_index, dir)

        if not is_oob and not get_bit(game.board, new_idx):
            available_directions.append(dir)

    return available_directions


def from_2d_game_state(game: GameState2D) -> GameState:

    board = 0

    num_rows, num_cols = game.grid.shape

    for row in range(num_rows):
        for col in range(num_cols):
            if game.grid[row, col]:
                idx = row * num_cols + col
                board |= BIT_MASKS[idx]

    # 2. Convert players
    players = tuple(
        Player(
            idx=p.row * num_cols + p.col,
            can_move=p.can_move,
        )
        for p in game.players
    )

    return GameState(num_rows, num_cols, board=board, players=players)



def from_bitboard(game: GameState) -> GameState2D:
    # 1. Reconstruct grid
    grid = np.zeros((game.num_rows, game.num_cols), dtype=bool)

    board = game.board
    while board:
        lsb = board & -board
        idx = lsb.bit_length() - 1

        row = idx // game.num_cols
        col = idx % game.num_cols

        grid[row, col] = True

        board ^= lsb  # remove bit

    # 2. Reconstruct players
    players = tuple(
        Player2D(
            row=p.idx // game.num_cols,
            col=p.idx % game.num_cols,
            can_move=p.can_move,
        )
        for p in game.players
    )

    return GameState2D(grid=grid, players=players)

