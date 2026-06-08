import pickle
import numpy as np

from pathlib import Path

from tron.game import from_2d_game_state, from_bitboard

from tron.game_2d import GameState2D, Player2D


from tron.ai.algos import choose_direction_basic_minimax
from tron.ai.MINIMAX_ORACLE_SIMUL_MOVES import (
    MinimaxContext,
    solve_zero_sum_matrix_game,
)


from tron.ai.tron_model import DummyTronModel

from tron.utils.data_utils import label_every_gamestate


def main():

    NUM_ROWS = NUM_COLS = 4

    dummy_model = DummyTronModel()

    # Here opponent loses if it must go first, can tie if hero must go first
    # Basic minimax agrees

    gs1 = from_2d_game_state(
        GameState2D(
            grid=np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0],
                ],
                dtype=bool,
            ),
            players=(Player2D(1, 3, True), Player2D(2, 1, True)),
        )
    )

    gs2 = from_2d_game_state(
        GameState2D(
            grid=np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0],
                ],
                dtype=bool,
            ),
            players=(Player2D(1, 2, True), Player2D(1, 1, True)),
        )
    )

    # for gs in [gs1, gs2]:

    #     print("New gs \n\n")
    #     for hero_index in range(2):

    #         print(f"Hero is: {hero_index} \n")

    #         opponent_index = 0 if hero_index == 1 else 1
    #         context = MinimaxContext(DummyTronModel(), hero_index, opponent_index)
    #         oracle_context = OracleMinimaxContext(
    #             DummyTronModel(), hero_index, opponent_index, {}
    #         )

    #         result = basic_minimax(
    #             gs,
    #             50,
    #             is_hero=True,
    #             context=context,
    #         )

    #         print(f"{result.evaluation=}")

    #         oracle_result = oracle_minimax(gs, 50, True, context=oracle_context)

    #         print(
    #             f"{oracle_result.result=}\n{oracle_result.steps_to_result=}\n{oracle_result.hero_player=}\n{oracle_result.oppo_player=}"
    #         )

    # matrices = [
    #     [[1, -1], [-1, 1]],
    #     [[1, 1], [-1, 1]],
    #     [[0.0, -1], [-1.0, 0.0]],
    #     [[0.5, -1], [-1, 0.25]],
    #     [[-1, 0.0], [0.0, -1]],
    #     [[1.0, -0.2], [-0.4, -0.1]],
    #     [[0.6, -0.3], [-0.8, 0.8]],
    #     [[-0.6, 0.3], [0.8, -0.8]],
    # ]

    matrices = [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.5, -0.25], [-0.25, 0.5]],
        [[0, -1], [0, -1]],
    ]

    # p_move chosen * steps_to_win

    # (4 * 0.25) + (2 * 0.25) =

    # avg 3 steps to win * p_tie

    for matrix in matrices:
        print(solve_zero_sum_matrix_game(matrix))

    # script_dir = Path(__file__).resolve().parent

    # out_dir = script_dir / "oracle_data_simul"
    # out_dir.mkdir(exist_ok=True)

    # grid_dim = 3

    # oracle_table = label_every_gamestate(grid_dim)

    # with (out_dir / f"{grid_dim}x{grid_dim}_pessimistic.pkl").open("wb") as f:
    #     pickle.dump(oracle_table, f)

    # zero_evals = win_evals = other = 0
    # for gs, oi in oracle_table.items():

    #     abs_val = abs(oi.value)

    #     if abs_val == 1:
    #         win_evals += 1
    #     elif abs_val == 0.0:
    #         zero_evals += 1
    #     elif 0 < abs_val < 1.0:
    #         other += 1

    #         gs_2d = from_bitboard(gs)
    #         print(f"\nother: {oi.value}\n{gs_2d}")
    #     else:
    #         raise AssertionError()

    # print(f"{zero_evals=}, {win_evals=}, {other=}")

    # gs = from_2d_game_state(
    #     GameState2D(
    #         grid=np.array(
    #             [
    #                 [0, 1, 0],
    #                 [0, 1, 0],
    #                 [0, 1, 0],
    #             ],
    #             dtype=bool,
    #         ),
    #         players=(Player2D(1, 1, True), Player2D(0, 1, True)),
    #     )
    # )

    # print(f"{oracle_table[gs].value}")


if __name__ == "__main__":
    main()
