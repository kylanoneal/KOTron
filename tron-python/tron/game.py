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

    # def canonical_key(self):

    #     assert len(self.players) == 2
    #     assert self.players[0].can_move and self.players[1].can_move

    #     player_tup = (
    #         (self.players[0].idx, self.players[1].idx)
    #         if self.players[0].idx > self.players[1].idx
    #         else (self.players[1].idx, self.players[0].idx)
    #     )

    #     return (
    #         self.num_rows,
    #         self.num_cols,
    #         self.board,
    #         player_tup,
    #     )

    # def __hash__(self):
    #     return hash(self.canonical_key())

    # def __eq__(self, other):
    #     return self.canonical_key() == other.canonical_key()

    # NOTE: Could add if __debug__ to skip this
    def __post_init__(self):

        # 1. Validate players container
        if not isinstance(self.players, tuple):
            raise TypeError(
                f"self.players must be a tuple of Player instances, "
                f"got {type(self.players).__name__}"
            )

        # 2. Validate board shape/type equivalent
        if not isinstance(self.num_rows, int):
            raise TypeError(
                f"num_rows must be an int, got {type(self.num_rows).__name__}"
            )
        if not isinstance(self.num_cols, int):
            raise TypeError(
                f"num_cols must be an int, got {type(self.num_cols).__name__}"
            )
        if self.num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {self.num_rows}")
        if self.num_cols <= 0:
            raise ValueError(f"num_cols must be positive, got {self.num_cols}")

        num_cells = self.num_rows * self.num_cols
        if num_cells > len(BIT_MASKS):
            raise ValueError(
                f"board has {num_cells} cells, but only {len(BIT_MASKS)} bit masks exist"
            )

        if not isinstance(self.board, int):
            raise TypeError(f"board must be an int, got {type(self.board).__name__}")
        if self.board < 0:
            raise ValueError(f"board must be non-negative, got {self.board}")
        if self.board >= (1 << num_cells):
            raise ValueError(
                f"board has bits set outside the {self.num_rows}x{self.num_cols} grid"
            )

        # 3. Validate each player
        for idx, player in enumerate(self.players):
            # 3a. Type check
            if not isinstance(player, Player):
                raise TypeError(
                    f"Element {idx} of self.players must be Player, "
                    f"got {type(player).__name__}"
                )

        for idx, player in enumerate(self.players):
            # 3b. Bounds check
            if not (0 <= player.idx < num_cells):
                raise IndexError(
                    f"Player {idx} index out of bounds: "
                    f"{player.idx} not in [0, {num_cells - 1}]"
                )

            # 3c. Board occupancy check
            if not (self.board & BIT_MASKS[player.idx]):
                raise ValueError(
                    f"board bit at index {player.idx} must be set for a player head"
                )

            for j in range(idx + 1, len(self.players)):
                pj: Player = self.players[j]

                if player.idx == pj.idx:
                    if player.can_move or pj.can_move:
                        raise ValueError("Active players occupying same square")

        if len(self.players) != 2:
            raise NotImplementedError()

    @staticmethod
    def new_game(
        num_players: int = 2,
        num_rows: int = 10,
        num_cols: int = 10,
        random_starts: bool = False,
        neutral_starts: bool = False,
        obstacle_density: float = 0.0,
    ) -> "GameState":

        return from_2d_game_state(
            GameState2D.new_game(
                num_players=num_players,
                num_rows=num_rows,
                num_cols=num_cols,
                random_starts=random_starts,
                neutral_starts=neutral_starts,
                obstacle_density=obstacle_density,
            )
        )

    @staticmethod
    def transform(
        game_state: "GameState",
        do_lr_flip: bool,
        n_rot_90: int,
    ) -> "GameState":

        return from_2d_game_state(
            GameState2D.transform(
                from_bitboard(game_state), do_lr_flip=do_lr_flip, n_rot_90=n_rot_90
            )
        )


@dataclass(frozen=True)
class PovGameState:
    game_state: GameState
    hero_index: int
    opponent_index: int

    def __post_init__(self):

        assert (0 <= self.hero_index < 2) and (0 <= self.opponent_index < 2)

        assert self.hero_index != self.opponent_index


@dataclass(frozen=True)
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
    return [
        i
        for i in range(game_state.num_rows * game_state.num_cols)
        if (get_bit(game_state.board, i))
    ]


def get_next_position(
    game: GameState, player_index: int, direction: Direction
) -> tuple[int, bool]:

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


def get_next_player(game: GameState, player_index: int, direction: Direction) -> Player:

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

    next_players = tuple(get_next_player(game, i, d) for i, d in enumerate(directions))

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

def percent_board_filled(game: GameState) -> float:
    num_cells = game.num_rows * game.num_cols
    num_occupied = game.board.bit_count()
    
    return num_occupied / num_cells

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
