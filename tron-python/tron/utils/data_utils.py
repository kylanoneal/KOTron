import copy
import random
import imageio
import numpy as np


from math import comb
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import deque, defaultdict
from itertools import combinations, product
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
from tron.ai.benchmarks import Tactic


from tron.ai.minimax_oracle_pessimistic import (
    MinimaxContext,
    ResultComparison,
    OracleGameState,
    oracle_minimax,
    SpecialCase,
    GameResult,
    compare_results,
)

DEPTH = 100
DUMMY_MODEL = DummyTronModel()


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


def make_oracle_data(game: GameState) -> dict[GameState, OracleGameState]:

    h0_oracle_table = {}

    mm_context = MinimaxContext(
        DUMMY_MODEL,
        hero_index=0,
        opponent_index=1,
        oracle_table=h0_oracle_table,
    )

    # Populate h0 table
    oracle_minimax(game, depth=DEPTH, is_hero=True, context=mm_context)

    game2 = swap_two_player_game(game)

    h1_oracle_table = {}

    mm_context = MinimaxContext(
        DUMMY_MODEL,
        hero_index=0,
        opponent_index=1,
        oracle_table=h1_oracle_table,
    )

    # Populate h1 table
    oracle_minimax(game2, depth=DEPTH, is_hero=True, context=mm_context)

    special_cases = find_special_cases(h0_oracle_table, h1_oracle_table)

    for special_case in special_cases.values():

        for h0_gs, h1_gs in special_case:

            del h0_oracle_table[h0_gs]

    return h0_oracle_table, special_cases


# def label_every_gamestate(
#     grid_dim: int,
# ):

#     if grid_dim > 6 or grid_dim < 0:

#         raise ValueError("Not happening...")


#     n_squares = grid_dim * grid_dim

#     oracle_table = OrderedDict()

#     for i in tqdm(range(n_squares)):

#         for j in tqdm(range(i + 1, n_squares)):

#             assert j != i

#             i_row, i_col = i // grid_dim, i % grid_dim
#             j_row, j_col = j // grid_dim, j % grid_dim

#             grid = [[0] * grid_dim for _ in range(grid_dim)]

#             grid[i_row][i_col] = 1

#             assert grid[j_row][j_col] == 0
#             grid[j_row][j_col] = 1

#             zero_coords = []

#             for _row_idx, _row in enumerate(grid):
#                 for _col_idx, _value in enumerate(_row):
#                     if _value == 0:
#                         zero_coords.append((_row_idx, _col_idx))

#             zero_coord_combos = [
#                 combo
#                 for r in range(1, len(zero_coords) + 1)
#                 for combo in combinations(zero_coords, r)
#             ]

#             zero_coord_combos.insert(0, tuple([]))

#             for obstacle_coords in zero_coord_combos:

#                 obstacle_grid = deepcopy(grid)

#                 for _r, _c in obstacle_coords:
#                     assert obstacle_grid[_r][_c] == 0
#                     obstacle_grid[_r][_c] = 1

#                 for k in range(2):

#                     if k == 0:
#                         players = (
#                             Player2D(i_row, i_col, True),
#                             Player2D(j_row, j_col, True),
#                         )
#                     else:
#                         players = (
#                             Player2D(j_row, j_col, True),
#                             Player2D(i_row, i_col, True),
#                         )

#                     game = from_2d_game_state(
#                         GameState2D(
#                             grid=np.array(
#                                 obstacle_grid,
#                                 dtype=bool,
#                             ),
#                             players=players,
#                         )
#                     )

#                     make_oracle_table(game, oracle_table)

#                     # new_gs_explored = len(oracle_table) - last_oracle_len

#                     # if new_gs_explored > 2:
#                     #     print(f"{new_gs_explored=}")

#                     # last_oracle_len = len(oracle_table)

#     print(f"Created oracle table with {len(oracle_table)} total entries.")
#     print(f"Expected for grid dim {grid_dim}:")
#     expected_num_gamestates(grid_dim)

#     return oracle_table


def swap_two_player_game(game: GameState) -> GameState:

    assert len(game.players) == 2

    return GameState(
        game.num_rows, game.num_cols, game.board, (game.players[1], game.players[0])
    )


def find_special_cases(h0_oracle_table, h1_oracle_table):

    assert len(h0_oracle_table) == len(h1_oracle_table)

    special_cases = defaultdict(list)

    for h0_gs, h0_oi in h0_oracle_table.items():

        h1_gs = swap_two_player_game(h0_gs)

        h1_oi = h1_oracle_table[h1_gs]

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

            special_cases[SpecialCase.ONE_TIE_ONE_WIN].append((h0_gs, h1_gs))

        elif is_opposite_decisive_result:

            ##########################################################
            # Disambiguate case of opposite decisive results
            #
            #   - Assign speical case of OPPOSITE_DECISIVE_RESULT
            #
            # Label this position with 0.0 since half the time you
            # win and half the time you lose?
            ##########################################################

            special_cases[SpecialCase.OPPOSITE_DECISIVE_RESULT].append((h0_gs, h1_gs))

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

            special_cases[SpecialCase.DIFF_STEPS_TO_SAME_DECISIVE_RESULT].append(
                (h0_gs, h1_gs)
            )

        else:

            if is_diff_steps_to_tie:

                ##########################################################
                # Ties are always labeled 0, so doesn't matter how long
                # it takes to tie. Minimax also has no preference on
                # when it ties (unlike losses which it wants to delay, and
                # wins it wants to expediate).
                ##########################################################

                special_cases[SpecialCase.DIFF_STEPS_TO_TIE].append((h0_gs, h1_gs))

    n_each_special_case = {k: len(special_cases[k]) for k in special_cases.keys()}
    print(f"{n_each_special_case=}")

    return special_cases


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


def make_tactics_from_oracle(
    oracle_table: dict[GameState, OracleGameState], special_cases, n_tactics: int = 10
):

    # Flatten special cases

    special_cases_set = set()

    for special_instances in special_cases.values():

        for instance_pair in special_instances:
            special_cases_set.update(instance_pair)

    # Find situations where one move is correct

    tactics: list[Tactic] = []

    items = list(oracle_table.items())
    random.shuffle(items)

    for gs, oracle_info in items:

        if len(tactics) >= n_tactics:
            break

        expected_outcome = oracle_info.result

        assert tron.get_status(gs).status == GameStatus.IN_PROGRESS

        p1_possible_dirs = tron.get_possible_directions(gs, 0)
        p2_possible_dirs = tron.get_possible_directions(gs, 1)

        if len(p1_possible_dirs) < 2 or len(p2_possible_dirs) == 0:
            continue

        p1_moves_that_achieve_outcome = []
        p2_responses = []

        special_case_found = False

        for p1_dir in p1_possible_dirs:

            best_p2_outcome = None
            best_p2_dir = None

            for p2_dir in p2_possible_dirs:

                next_gs = tron.next(gs, (p1_dir, p2_dir))

                if not next_gs in oracle_table:

                    if next_gs in special_cases_set:
                        print(f"Found a special case, do not use this tactic")
                        special_case_found = True
                        break
                    if next_gs.players[0] == next_gs.players[1]:
                        print(f"Players colliding was only move here, skipping!")
                        continue
                    else:
                        raise ValueError(f"wtf? {next_gs.players=}")

                next_outcome = oracle_table[next_gs]

                if best_p2_outcome is None:
                    best_p2_outcome = next_outcome
                    best_p2_dir = p2_dir
                else:
                    compare_result = compare_results(
                        best_p2_outcome, next_outcome, is_p1=False
                    )

                    if compare_result == ResultComparison.BETTER:
                        best_p2_outcome = next_outcome
                        best_p2_dir = p2_dir

            # Should only be None in the case of
            # players colliding tie edge case or special cases
            if best_p2_outcome is not None:
                if best_p2_outcome.result == expected_outcome:
                    p1_moves_that_achieve_outcome.append(p1_dir)
                    p2_responses.append(best_p2_dir)

        if len(p1_moves_that_achieve_outcome) == 1 and not special_case_found:

            tactics.append(
                Tactic(
                    PovGameState(gs, 0, 1),
                    opposing_dirs=p2_responses,
                    expected_hero_dirs=p1_moves_that_achieve_outcome,
                )
            )

    return tactics


def subsample_and_augment(
    oracle_examples: list[OracleGameState], keep_rate: float = 0.01
):

    shallow_copy = copy.copy(oracle_examples)
    random.shuffle(shallow_copy)

    keep_step = len(oracle_examples) // max(1, int((len(oracle_examples) * keep_rate)))

    subsampled = shallow_copy[::keep_step]

    for i in range(len(subsampled)):

        augmented = GameState.transform(
            subsampled[i].game,
            do_lr_flip=random.random() > 0.5,
            n_rot_90=random.randrange(0, 4),
        )

        subsampled[i] = OracleGameState(
            augmented,
            result=subsampled[i].result,
            steps_to_result=subsampled[i].steps_to_result,
        )

    return subsampled
