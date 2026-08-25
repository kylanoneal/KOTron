import random
import imageio
import numpy as np

from math import comb
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import deque
from itertools import combinations
from collections import OrderedDict


import tron
from tron.game import (
    GameState,
    GameStatus,
    StatusInfo,
    Direction,
    GameState2D,
    Player2D,
    PovGameState,
    get_possible_directions,
    from_2d_game_state,
    from_bitboard,
)

from tron.ai.tron_model import DummyTronModel


from tron.ai.minimax_oracle_pessimistic import (
    MinimaxContext,
    OracleInfo,
    oracle_minimax,
)

DEPTH = 100


def expected_num_gamestates(grid_dim):

    n = grid_dim * grid_dim
    player_combos = comb(n, 2)
    obstacle_combos = 2 ** (n - 2)
    total_states = player_combos * obstacle_combos

    print(
        f"{grid_dim}x{grid_dim}: "
        f"N={n}, "
        f"players={player_combos:,}, "
        f"obstacles={obstacle_combos:,}, "
        f"total={total_states:,}"
    )


def label_every_gamestate(
    grid_dim: int,
):

    if grid_dim > 6 or grid_dim < 0:

        raise ValueError("Not happening...")

    model = DummyTronModel()

    n_squares = grid_dim * grid_dim

    oracle_table = OrderedDict()

    for i in tqdm(range(n_squares)):

        for j in tqdm(range(i + 1, n_squares)):

            assert j != i

            i_row, i_col = i // grid_dim, i % grid_dim
            j_row, j_col = j // grid_dim, j % grid_dim

            grid = [[0] * grid_dim for _ in range(grid_dim)]

            grid[i_row][i_col] = 1

            assert grid[j_row][j_col] == 0
            grid[j_row][j_col] = 1

            zero_coords = []

            for _row_idx, _row in enumerate(grid):
                for _col_idx, _value in enumerate(_row):
                    if _value == 0:
                        zero_coords.append((_row_idx, _col_idx))

            zero_coord_combos = [
                combo
                for r in range(1, len(zero_coords) + 1)
                for combo in combinations(zero_coords, r)
            ]

            zero_coord_combos.insert(0, tuple([]))

            for obstacle_coords in zero_coord_combos:

                obstacle_grid = deepcopy(grid)

                for _r, _c in obstacle_coords:
                    assert obstacle_grid[_r][_c] == 0
                    obstacle_grid[_r][_c] = 1

                for k in range(2):

                    if k == 0:
                        players = (
                            Player2D(i_row, i_col, True),
                            Player2D(j_row, j_col, True),
                        )
                    else:
                        players = (
                            Player2D(j_row, j_col, True),
                            Player2D(i_row, i_col, True),
                        )

                    game = from_2d_game_state(
                        GameState2D(
                            grid=np.array(
                                obstacle_grid,
                                dtype=bool,
                            ),
                            players=players,
                        )
                    )

                    mm_context = MinimaxContext(
                        model,
                        hero_index=0,
                        opponent_index=1,
                        oracle_table=oracle_table,
                    )

                    root_oracle_info: OracleInfo = oracle_minimax(
                        game, depth=DEPTH, is_hero=True, context=mm_context
                    )

                    # new_gs_explored = len(oracle_table) - last_oracle_len

                    # if new_gs_explored > 2:
                    #     print(f"{new_gs_explored=}")

                    # last_oracle_len = len(oracle_table)

    print(f"Created oracle table with {len(oracle_table)} total entries.")
    print(f"Expected for grid dim {grid_dim}:")
    expected_num_gamestates(grid_dim)

    return oracle_table


def pad_gamestate(
    game: GameState, num_rows: int, num_cols: int, obstacle_density: float = 0.5
):

    old_rows, old_cols = game.num_rows, game.num_cols

    if num_rows <= old_rows or num_cols <= old_cols:
        raise ValueError()

    row_offset = random.randint(0, num_rows - old_rows)
    col_offset = random.randint(0, num_cols - old_cols)

    game_2d = from_bitboard(game)

    # Put walls around grid
    one_padded_grid = np.ones((old_rows + 2, old_cols + 2), dtype=np.bool)
    one_padded_grid[
        1:-1,
        1:-1,
    ] = game_2d.grid

    # Handle corners
    one_padded_grid[0, 0] = np.random.rand(1) < obstacle_density
    one_padded_grid[0, -1] = np.random.rand(1) < obstacle_density
    one_padded_grid[-1, 0] = np.random.rand(1) < obstacle_density
    one_padded_grid[-1, -1] = np.random.rand(1) < obstacle_density

    # Initialize new grid with underlying pattern
    pattern_grid = np.random.rand(num_rows + 2, num_cols + 2) < obstacle_density

    # Paste original game grid
    pattern_grid[row_offset : row_offset + old_rows + 2, col_offset + old_cols + 2] = (
        one_padded_grid
    )

    # Remove outerwalls
    new_grid = pattern_grid[1:-1, 1:-1]

    new_players = [
        Player2D(p.row + row_offset, p.col + col_offset, p.can_move)
        for p in game_2d.players
    ]

    return from_2d_game_state(GameState2D(new_grid, new_players))
