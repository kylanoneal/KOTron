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
    from_2d_game_state,
    GameState2D,
    Player2D,
    PovGameState,
    get_possible_directions,
)

from tron.ai.tron_model import DummyTronModel
from tron.ai.MINIMAX_ORACLE_SIMUL_MOVES import (
    MinimaxContext,
    OracleInfo,
    oracle_minimax,
)
from tron.ai.training import (
    LabeledExample,
    make_dataset,
    get_label_magnitude,
    ModelExample,
)

from tron.ai.benchmarks import Tactic

from tron.utils.viz_utils import render_model_example_image

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
) -> dict[GameState, OracleInfo]:

    if grid_dim > 6 or grid_dim < 0:

        raise ValueError("Not happening...")

    model = DummyTronModel()

    n_squares = grid_dim * grid_dim

    unique_gamestates = set()

    oracle_table = OrderedDict()

    last_oracle_len = 0

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

                game = from_2d_game_state(
                    GameState2D(
                        grid=np.array(
                            obstacle_grid,
                            dtype=bool,
                        ),
                        players=(
                            Player2D(i_row, i_col, True),
                            Player2D(j_row, j_col, True),
                        ),
                    )
                )

                if game in unique_gamestates:
                    continue

                p1_mm_context = MinimaxContext(
                    model,
                    hero_index=0,
                    opponent_index=1,
                    oracle_table=oracle_table,
                )

                root_oracle_info: OracleInfo = oracle_minimax(
                    game, depth=DEPTH, context=p1_mm_context
                )

                # new_gs_explored = len(oracle_table) - last_oracle_len

                # if new_gs_explored > 2:
                #     print(f"{new_gs_explored=}")

                # last_oracle_len = len(oracle_table)

    print(f"Created oracle table with {len(oracle_table)} total entries.")
    print(f"Expected for grid dim {grid_dim}:")
    expected_num_gamestates(grid_dim)

    return oracle_table
