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
    grid_dim: int, hero_index: int
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
                    hero_index=hero_index,
                    opponent_index=0 if hero_index == 1 else 1,
                    oracle_table=oracle_table,
                )

                root_oracle_info: OracleInfo = oracle_minimax(
                    game, depth=DEPTH, is_hero=True, context=p1_mm_context
                )

                # new_gs_explored = len(oracle_table) - last_oracle_len

                # if new_gs_explored > 2:
                #     print(f"{new_gs_explored=}")

                # last_oracle_len = len(oracle_table)

    print(f"Created oracle table with {len(oracle_table)} total entries.")
    print(f"Expected for grid dim {grid_dim}:")
    expected_num_gamestates(grid_dim)

    return oracle_table


def disambiguate_oracle_tables(h0_oracle_table, h1_oracle_table):
    assert len(h0_oracle_table) == len(h1_oracle_table)

    diff_examples = []

    n_diff_result = 0
    n_steps_to_result_diff = 0

    disambiguated_oracle_table = {}

    special_cases = {
        SpecialCase.ONE_TIE_ONE_WIN: [],
        SpecialCase.DIFF_STEPS_TO_SAME_DECISIVE_RESULT: [],
        SpecialCase.OPPOSITE_DECISIVE_RESULT: [],
        SpecialCase.GOING_FIRST_GOOD: [],
    }

    for h0_gs, h0_oi in tqdm(h0_oracle_table.items(), "Disambiguating..."):

        h1_oi = h1_oracle_table[h0_gs]

        assert h1_oi.hero_player == h0_oi.oppo_player
        assert h0_oi.hero_player == h1_oi.oppo_player

        n_ties = sum([oi.result == GameResult.TIE for oi in [h1_oi, h0_oi]])

        # Special cases:

        is_one_tie_one_win = n_ties == 1

        is_opposite_decisive_result = (n_ties == 0) and (h0_oi.result == h1_oi.result)

        is_diff_steps_to_same_decisive_result = (
            (n_ties == 0)
            and (h0_oi.result != h1_oi.result)
            and (h0_oi.steps_to_result != h1_oi.steps_to_result)
        )

        is_diff_steps_to_tie = (n_ties == 2) and (
            h0_oi.steps_to_result != h1_oi.steps_to_result
        )

        if is_one_tie_one_win:

            ##########################################################
            # Disambiguate case of one tie and one win:
            #
            #   - Assign speical case of one tie one win
            #   - Store amount of steps to winning result
            #   - Store h0 perspective winnner
            #
            # This special case indicates that 50% of the time this
            # position results in a win and 50% of the time it results
            # in a tie. To label this position, could divide win/loss
            # label by two, but might confuse the model since there
            # are such few cases of this.
            ##########################################################

            #############################
            # Going first is good?? Huh??
            #############################

            if (
                h0_oi.result == GameResult.HERO_WIN
                or h1_oi.result == GameResult.HERO_WIN
            ):

                special_cases[SpecialCase.GOING_FIRST_GOOD].append(
                    (h0_gs, h0_oi, h1_oi)
                )

                # NOTE: Don't know what's going on so just doing this for now:

                disambiguated_oracle_table[h0_gs] = h0_oi
            else:

                ##########################################################
                # Disambiguate and create new Oracle Info
                ##########################################################

                h0_is_tie = h0_oi.result == GameResult.TIE

                winning_steps_to_result = (
                    h1_oi.steps_to_result if h0_is_tie else h0_oi.steps_to_result
                )

                h0_perspective_winner = (
                    GameResult.HERO_WIN if h0_is_tie else GameResult.OPPO_WIN
                )

                new_oi = OracleInfo(
                    result=h0_perspective_winner,
                    steps_to_result=winning_steps_to_result,
                    hero_player=h0_oi.hero_player,
                    oppo_player=h0_oi.oppo_player,
                    special_case=SpecialCase.ONE_TIE_ONE_WIN,
                )

                disambiguated_oracle_table[h0_gs] = new_oi
                special_cases[SpecialCase.ONE_TIE_ONE_WIN].append((h0_gs, h0_oi, h1_oi))

        elif is_opposite_decisive_result:

            ##########################################################
            # Disambiguate case of opposite decisive results
            #
            #   - Assign speical case of OPPOSITE_DECISIVE_RESULT
            #
            # Label this position with 0.0 since half the time you
            # win and half the time you lose?
            ##########################################################

            new_oi = OracleInfo(
                result=h0_oi.result,
                steps_to_result=h0_oi.steps_to_result,
                hero_player=h0_oi.hero_player,
                oppo_player=h0_oi.oppo_player,
                special_case=SpecialCase.OPPOSITE_DECISIVE_RESULT,
            )
            disambiguated_oracle_table[h0_gs] = new_oi

            special_cases[SpecialCase.OPPOSITE_DECISIVE_RESULT].append(
                (h0_gs, h0_oi, h1_oi)
            )

        elif is_diff_steps_to_same_decisive_result:

            ##########################################################
            # Disambiguate case of different number of steps to the
            # same decisive result
            #
            #   - Assign special case
            #   - Store the greater steps to result
            #
            # This special case indicates that the same outcome is
            # achieved but it might take longer to achieve it depending
            # on who has the misfortune of going first. If a win can
            # still be guaranteed when going first, that is the
            # principal line which should take more steps
            ##########################################################

            new_oi = OracleInfo(
                result=h0_oi.result,
                steps_to_result=(
                    h1_oi.steps_to_result
                    if h1_oi.steps_to_result > h0_oi.steps_to_result
                    else h0_oi.steps_to_result
                ),
                hero_player=h0_oi.hero_player,
                oppo_player=h0_oi.oppo_player,
                special_case=SpecialCase.DIFF_STEPS_TO_SAME_DECISIVE_RESULT,
            )

            disambiguated_oracle_table[h0_gs] = new_oi
            special_cases[SpecialCase.DIFF_STEPS_TO_SAME_DECISIVE_RESULT].append(
                (h0_gs, h0_oi, h1_oi)
            )

        else:

            if is_diff_steps_to_tie:

                ##########################################################
                # Ties are always labeled 0, so doesn't matter how long
                # it takes to tie. Minimax also has no preference on
                # when it ties (unlike losses which it wants to delay, and
                # wins it wants to expediate).
                ##########################################################

                pass

            disambiguated_oracle_table[h0_gs] = h0_oi

    assert len(disambiguated_oracle_table) == len(h0_oracle_table)
    assert len(disambiguated_oracle_table) == len(h1_oracle_table)

    n_each_special_case = {k: len(special_cases[k]) for k in special_cases.keys()}
    print(f"{n_each_special_case=}")

    return disambiguated_oracle_table, special_cases


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
