from dataclasses import dataclass
from tron import Direction


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


def get_bit(board: int, idx: int) -> bool:
    return (board & BIT_MASKS[idx]) != 0


def get_next_player(board: int, player: BitBoardPlayer, direction: Direction) -> BitBoardPlayer:

    next_idx, next_can_move = player.idx, player.can_move

    if player.can_move:
        if direction == Direction.UP:
            if player.idx < NUM_COLS:
                next_can_move = False
            else:
                next_idx = player.idx - NUM_COLS

        elif direction == Direction.DOWN:
            if player.idx >= NUM_COLS * (NUM_ROWS - 1):
                next_can_move = False
            else:
                next_idx = player.idx + NUM_COLS

        elif direction == Direction.LEFT:
            if player.idx % NUM_COLS == 0:
                next_can_move = False
            else:
                next_idx = player.idx - 1

        elif direction == Direction.RIGHT:
            if player.idx % NUM_COLS == NUM_COLS - 1:
                next_can_move = False
            else:
                next_idx = player.idx + 1

        else:
            raise ValueError("Invalid direction.")

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
    for i in range(len(next_players) - 1):

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
