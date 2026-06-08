import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from itertools import combinations

import tron
from tron.game import (
    GameStatus,
    StatusInfo,
    Direction,
    from_2d_game_state,
    GameState2D,
    Player2D,
)

from tron.ai.tron_model import DummyTronModel
from tron.ai.minimax import basic_minimax, MinimaxContext, MinimaxResult
from tron.ai.training import LabeledExample, make_dataset


def label_every_gamestate(grid_dim: int) -> list[LabeledExample]:

    DEPTH = 100

    if grid_dim > 6 or grid_dim < 0:

        raise ValueError("Not happening...")

    model = DummyTronModel()

    n_squares = grid_dim * grid_dim

    games = []

    for i in tqdm(range(n_squares)):

        for j in tqdm(range(n_squares)):

            if j == i:
                continue

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

            zero_coord_combos.append(tuple([]))

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

                game_status: StatusInfo = tron.get_status(game)

                curr_game = [deepcopy(game)]

                while game_status.status == GameStatus.IN_PROGRESS:

                    p1_mm_result: MinimaxResult = basic_minimax(
                        game,
                        depth=DEPTH,
                        is_hero=True,
                        context=MinimaxContext(
                            model, hero_index=0, opponent_index=1
                        ),
                    )

                    p2_mm_result: MinimaxResult = basic_minimax(
                        game,
                        depth=DEPTH,
                        is_hero=True,
                        context=MinimaxContext(
                            model, hero_index=1, opponent_index=0
                        ),
                    )

                    p1_dir = (
                        Direction.UP
                        if p1_mm_result.principal_variation is None
                        else p1_mm_result.principal_variation
                    )
                    p2_dir = (
                        Direction.UP
                        if p2_mm_result.principal_variation is None
                        else p2_mm_result.principal_variation
                    )

                    game = tron.next(game, directions=(p1_dir, p2_dir))

                    curr_game.append(game)

                    game_status = tron.get_status(game)

                games.append(curr_game)

    print(f"Total games: {len(games)}")

    print(f"Total gamestates: {sum([len(g) for g in games])}")

    for full_game in games:

        for k, gstate in enumerate(full_game):

            if k == len(full_game) - 1:
                assert tron.get_status(gstate).status != GameStatus.IN_PROGRESS
            else:
                assert tron.get_status(gstate).status == GameStatus.IN_PROGRESS


    unique_game_states = set()

    for game in games:

        assert tron.get_status(game[-1]).status != GameStatus.IN_PROGRESS
        for gs in game[:-1]:

            assert tron.get_status(gs).status == GameStatus.IN_PROGRESS
            unique_game_states.add(gs)

    print(f"Num unique game states: {len(unique_game_states)}")

    full_dataset = make_dataset(games, shuffle=False, keep_rate=1.1, do_affine=False)

    print(f"Total examples: {len(full_dataset)}")

    unique_examples = set()

    unique_game_states = set()

    for example in full_dataset:

        unique_examples.add(example)

        gs = example.pov_game_state.game_state

        assert gs.players[0].can_move and gs.players[1].can_move

        unique_game_states.add(gs)

    print(f"Num unique examples: {len(unique_examples)}")

    print(f"Num unique game states: {len(unique_game_states)}")

    return list(unique_examples)
