import random
import imageio
import numpy as np

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
from tron.ai.MINIMAX_THAT_BUILDS_ORACLE_INFO import (
    oracle_minimax,
    MinimaxContext,
    OracleInfo,
    GameResult,
    SpecialCase,
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


def label_every_gamestate(grid_dim: int) -> list[LabeledExample]:

    if grid_dim > 6 or grid_dim < 0:

        raise ValueError("Not happening...")

    model = DummyTronModel()

    n_squares = grid_dim * grid_dim

    unique_gamestates = set()

    h0_oracle_table = OrderedDict()
    h1_oracle_table = OrderedDict()

    last_oracle_len = 0

    for hero_idx, otable in enumerate([h0_oracle_table, h1_oracle_table]):

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
                        hero_index=hero_idx,
                        opponent_index=0 if hero_idx == 1 else 1,
                        oracle_table=otable,
                    )

                    root_oracle_info: OracleInfo = oracle_minimax(
                        game, depth=DEPTH, is_hero=True, context=p1_mm_context
                    )

                    # new_gs_explored = len(oracle_table) - last_oracle_len

                    # if new_gs_explored > 2:
                    #     print(f"{new_gs_explored=}")

                    # last_oracle_len = len(oracle_table)

    assert len(h0_oracle_table) == len(h1_oracle_table)

    diff_examples = []

    n_diff_result = 0
    n_steps_to_result_diff = 0

    for h0_gs, h0_oi in h0_oracle_table.items():

        h1_oi = h1_oracle_table[h0_gs]

        assert h1_oi.hero_player == h0_oi.oppo_player
        assert h0_oi.hero_player == h1_oi.oppo_player

        n_ties = sum([oi.result == GameResult.TIE for oi in [h1_oi, h0_oi]])

        if h1_oi.steps_to_result != h0_oi.steps_to_result:
            n_steps_to_result_diff += 1

        if n_ties == 2:
            pass
        elif (
            (n_ties == 1)
            or (h1_oi.result == h0_oi.result)
            or (h1_oi.steps_to_result != h0_oi.steps_to_result)
        ):

            h0_label = h0_oi.steps_to_result * (
                1 if h0_oi.result == GameResult.HERO_WIN else -1
            )

            h1_label = h1_oi.steps_to_result * (
                1 if h1_oi.result == GameResult.HERO_WIN else -1
            )

            h0_label = h0_label * 1000 if h0_oi.result == GameResult.TIE else h0_label
            h1_label = h1_label * 1000 if h1_oi.result == GameResult.TIE else h1_label

            model_example = ModelExample(
                LabeledExample(
                    PovGameState(h0_gs, 0, 1),
                    h0_label,
                ),
                prediction=h1_label,
            )

            diff_examples.append(model_example)

            n_diff_result += 1

            if n_ties == 1:

                assert (
                    h0_oi.result != GameResult.HERO_WIN
                    and h1_oi.result != GameResult.HERO_WIN
                )

                h0_is_tie = h0_oi.result == GameResult.TIE

                if h0_is_tie:
                    assert h1_oi.result == GameResult.OPPO_WIN
                else:
                    assert h0_oi.result == GameResult.OPPO_WIN

                winning_steps_to_result = (
                    h1_oi.steps_to_result if h0_is_tie else h0_oi.steps_to_result
                )

                h0_perspective_winner = (
                    GameResult.HERO_WIN if h0_is_tie else GameResult.OPPO_WIN
                )

                h0_oi.special_case = SpecialCase.ONE_TIE_ONE_WIN
                h0_oi.steps_to_result = winning_steps_to_result
                h0_oi.result = h0_perspective_winner

            elif h1_oi.result != h0_oi.result:

                h0_oi.special_case = SpecialCase.DIFF_STEPS_TO_SAME_RESULT
                h0_oi.steps_to_result = (
                    h1_oi.steps_to_result
                    if h1_oi.steps_to_result > h0_oi.steps_to_result
                    else h0_oi.steps_to_result
                )

            elif h1_oi.result == h0_oi.result:

                h0_oi.special_case = SpecialCase.OPPOSITE_RESULT
            else:
                raise AssertionError()

    out_dir = Path(r"C:\Users\kylan\code\KOTron\tron-python\scripts\y2026\m05\viz_diff")

    print(f"{n_diff_result=} out of {len(h0_oracle_table)} total game states")
    print(f"{n_steps_to_result_diff=}")

    grid_dim = int(len(diff_examples) ** 0.5) + 1
    # random.shuffle(diff_examples)
    viz = render_model_example_image(
        diff_examples, num_rows=grid_dim, num_cols=grid_dim
    )

    imageio.imwrite(out_dir / "diff_viz.png", viz)

    return h0_oracle_table


def make_tactics_from_oracle(
    oracle_table: dict[GameState, OracleInfo], n_tactics: int = 10
):

    # Find situations where one move is correct

    tactics: list[Tactic] = []
    used_up_gamestates = set()

    for gs, _oracle_info in oracle_table.items():

        if _oracle_info.special_case is not None:
            print(f"Skipping special case: {_oracle_info.special_case}")
            continue

        curr_oracle_info = _oracle_info
        curr_gs = gs

        assert tron.get_status(curr_gs).status == GameStatus.IN_PROGRESS

        opposing_dirs = []
        hero_dirs = []

        while True:

            more_than_one_pv = len(curr_oracle_info.pvs) > 1

            n_possible_moves = (
                len(curr_oracle_info.pvs)
                + len(curr_oracle_info.non_pvs)
                + len(curr_oracle_info.slower_pvs)
            )

            if (
                more_than_one_pv
                or (n_possible_moves == 1)
                or (curr_gs in used_up_gamestates)
            ):
                break

            used_up_gamestates.add(curr_gs)

            pv_move = curr_oracle_info.pvs[0]

            h_dir = pv_move.dir

            print(f"{pv_move.response=}")
            # TODO: could use multiple equal responses from opponent
            oppo_dir = pv_move.response.pvs[0]

            curr_gs = tron.next(curr_gs, (h_dir, oppo_dir))

            if tron.get_status(curr_gs).status != GameStatus.IN_PROGRESS:
                break

            hero_dirs.append(h_dir)
            opposing_dirs.append(oppo_dir)

            curr_oracle_info = oracle_table[curr_gs]

        hero_index = 0 if _oracle_info.hero_player == gs.players[0] else 1
        oppo_index = 1 if _oracle_info.oppo_player == gs.players[1] else 0

        assert hero_index == 0, "Oracle table shouldn't contain perspective swaps"
        assert gs.players[oppo_index] == _oracle_info.oppo_player
        assert gs.players[hero_index] == _oracle_info.hero_player

        if len(opposing_dirs) > 0:
            tactics.append(
                Tactic(
                    PovGameState(gs, hero_index, oppo_index),
                    opposing_dirs=opposing_dirs,
                    expected_hero_dirs=hero_dirs,
                )
            )

        if len(tactics) >= n_tactics:
            break

    return tactics
