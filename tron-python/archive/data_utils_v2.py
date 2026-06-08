import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import deque
from itertools import combinations


import tron
from tron.game import (
    GameStatus,
    StatusInfo,
    Direction,
    from_2d_game_state,
    GameState2D,
    Player2D,
    PovGameState,
)

from tron.ai.tron_model import DummyTronModel
from tron.ai.MINIMAX_PROTOTYPE_TREE import basic_minimax, MinimaxContext, MinimaxResult
from tron.ai.training import LabeledExample, make_dataset, get_label_magnitude

DEPTH = 100


def get_best_mm_direction(mm_results: list[MinimaxResult]):

    best_eval = float("-inf")
    best_dir = None

    for mm_result in mm_results:

        if mm_result.evaluation > best_eval:
            best_dir = mm_result.dir

    return best_dir


def get_unique_from_mm_root(
    root_mm_results: list[MinimaxResult], mm_context: MinimaxContext
):

    queue = deque([root_mm_results])
    unique_examples = set()
    unique_gamestates = set()

    while queue:
        node = queue.popleft()

        assert all(node[0].game == n.game for n in node)
        assert all(node[0].is_hero == n.is_hero for n in node)


        for r in node:
            if r.sub_results is not None:

                queue.append(r.sub_results)

        if not node[0].is_hero:
            continue

        game = node[0].game
        depth = node[0].depth

        best_eval = float("-inf")

        for r in node:
            best_eval = max(best_eval, r.evaluation)


        if tron.get_status(game).status != GameStatus.IN_PROGRESS:
            continue

        if best_eval == 0:
            hero_label = opponent_label = 0
        else:
            # NOTE: Jank
            _steps_to_end, remainder = divmod(best_eval, mm_context.win_magnitude)

            assert remainder == 0
            _steps_to_end = int(abs(_steps_to_end))

            steps_to_end = depth - _steps_to_end + 1

            label_magnitude = get_label_magnitude(steps_to_end)

            hero_label = label_magnitude if best_eval > 0 else -label_magnitude
            opponent_label = -hero_label

        hero_pov_game_state = PovGameState(
            game, mm_context.hero_index, opponent_index=mm_context.opponent_index
        )
        hero_example = LabeledExample(hero_pov_game_state, hero_label)

        oppo_pov_game_state = PovGameState(
            game, mm_context.opponent_index, opponent_index=mm_context.hero_index
        )
        oppo_example = LabeledExample(oppo_pov_game_state, opponent_label)


        if game not in unique_gamestates:

            assert hero_example not in unique_examples
            assert oppo_example not in unique_examples
        else:
            assert hero_example in unique_examples
            assert oppo_example in unique_examples

        
        unique_gamestates.add(game)
        unique_examples.add(oppo_example)
        unique_examples.add(hero_example)



    return unique_gamestates, unique_examples


def label_every_gamestate(grid_dim: int) -> list[LabeledExample]:

    if grid_dim > 6 or grid_dim < 0:

        raise ValueError("Not happening...")

    model = DummyTronModel()

    n_squares = grid_dim * grid_dim

    unique_examples = set()
    unique_gamestates = set()

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

                if game in unique_gamestates:
                    continue

                p1_mm_context = MinimaxContext(model, hero_index=0, opponent_index=1)

                p1_mm_results: list[MinimaxResult] = basic_minimax(
                    game,
                    depth=DEPTH,
                    is_hero=True,
                    context=p1_mm_context
                )

                game_unique_gamestates, game_unique_examples = get_unique_from_mm_root(p1_mm_results, p1_mm_context)

                assert len(game_unique_examples) == 2 * len(game_unique_gamestates)

                unique_examples.update(game_unique_examples)
                unique_gamestates.update(game_unique_gamestates)


    print(f"Num unique examples: {len(unique_examples)}")
    print(f"Num unique gamestates: {len(unique_gamestates)}")


    return list(unique_examples)
