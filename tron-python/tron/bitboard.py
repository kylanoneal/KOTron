from dataclasses import dataclass

from tron import GameState, Player


NUM_ROWS = 5
NUM_COLS = 5

BIT_MASKS = [1 << i for i in range(NUM_ROWS * NUM_COLS)]


@dataclass(frozen=True)
class BitBoardPlayer:
    idx: int
    can_move: bool


@dataclass(frozen=True)
class BitBoardGameState:
    board: int
    players: tuple[BitBoardPlayer]

    
@dataclass
class PovBitBoardGameState:
    game_state: BitBoardGameState
    hero_index: int
    opponent_index: int

    # def __eq__(self, other):
    #     if not isinstance(other, PovGameState):
    #         return False
    #     return self.hero_index == other.hero_index and self.game_state == other.game_state

    # def __hash__(self):

    #     return hash((self.game_state, self.hero_index))


def get_bit(board: int, idx: int) -> bool:
    return (board & BIT_MASKS[idx]) != 0

def get_wall_indices(board: int) -> list[int]:
    return [i for i in range(NUM_ROWS * NUM_COLS) if (get_bit(board, i))]


def get_next_position(player: BitBoardPlayer, direction: Direction) -> tuple[int, bool]:

    oob = False

    if direction == Direction.UP:
        if player.idx < NUM_COLS:
            oob = True

        next_idx = player.idx - NUM_COLS

    elif direction == Direction.DOWN:
        if player.idx >= NUM_COLS * (NUM_ROWS - 1):
            oob = True

        next_idx = player.idx + NUM_COLS

    elif direction == Direction.LEFT:
        if player.idx % NUM_COLS == 0:
            oob = True

        next_idx = player.idx - 1

    elif direction == Direction.RIGHT:
        if player.idx % NUM_COLS == NUM_COLS - 1:
            oob = True

        next_idx = player.idx + 1

    else:
        raise ValueError("Invalid direction.")

    return next_idx, oob


def get_next_player(
    board: int, player: BitBoardPlayer, direction: Direction
) -> BitBoardPlayer:

    next_idx, next_can_move = player.idx, player.can_move

    if player.can_move:

        _new_idx, _oob = get_next_position(player, direction)

        if _oob:
            next_can_move = False
        else:
            next_idx = _new_idx

    if next_can_move:
        if get_bit(board, next_idx):
            next_can_move = False
            next_idx = player.idx

    return BitBoardPlayer(next_idx, next_can_move)


def next(game: BitBoardGameState, directions: tuple[Direction]) -> BitBoardGameState:

    next_players = tuple(
        get_next_player(game.board, p, d) for p, d in zip(game.players, directions)
    )

    next_board = game.board

    # Update board and handle case where 2 or more players try to occupy the same square
    for i in range(len(next_players)):

        pi: BitBoardPlayer = next_players[i]
        next_board = next_board | BIT_MASKS[pi.idx]

        if pi.can_move:
            for j in range(i + 1, len(next_players)):
                pj: BitBoardPlayer = next_players[j]

                if pj.can_move:
                    if pi.idx == pj.idx:

                        next_players = tuple(
                            BitBoardPlayer(
                                p.idx, can_move=False if p.idx == pi.idx else p.can_move
                            )
                            for p in next_players
                        )
                        break

    return BitBoardGameState(next_board, next_players)


def get_possible_directions(
    game: BitBoardGameState,
    player_index: int,
):

    available_directions = []
    player = game.players[player_index]

    for dir in Direction:

        new_idx, is_oob = get_next_position(player, dir)

        if not is_oob and not get_bit(game.board, new_idx):
            available_directions.append(dir)

    return available_directions


def from_2d_game_state(game: GameState) -> BitBoardGameState:

    board = 0

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            if game.grid[row, col]:
                idx = row * NUM_COLS + col
                board |= BIT_MASKS[idx]

    # 2. Convert players
    players = tuple(
        BitBoardPlayer(
            idx=p.row * NUM_COLS + p.col,
            can_move=p.can_move,
        )
        for p in game.players
    )

    return BitBoardGameState(board=board, players=players)


import numpy as np


def from_bitboard(game: BitBoardGameState) -> GameState:
    # 1. Reconstruct grid
    grid = np.zeros((NUM_ROWS, NUM_COLS), dtype=bool)

    board = game.board
    while board:
        lsb = board & -board
        idx = lsb.bit_length() - 1

        row = idx // NUM_COLS
        col = idx % NUM_COLS

        grid[row, col] = True

        board ^= lsb  # remove bit

    # 2. Reconstruct players
    players = tuple(
        Player(
            row=p.idx // NUM_COLS,
            col=p.idx % NUM_COLS,
            can_move=p.can_move,
        )
        for p in game.players
    )

    return GameState(grid=grid, players=players)

