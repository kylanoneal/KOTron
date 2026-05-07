import sys

import numpy as np

from tron.game import (
    GameState,
    Player,
    GameStatus,
    StatusInfo,
    next,
    get_status,
    Direction,
    PovGameState,
    from_2d_game_state,
)
from tron.game_2d import GameState2D, Player2D
from tron.ai.tron_model import RandomTronModel
from tron.ai.minimax import basic_minimax, MinimaxContext
from tron.ai.algos import choose_direction_random

TIES_5X5 = [
    PovGameState(
        game_state=from_2d_game_state(
            GameState2D(
                grid=np.array(
                    [
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                    ],
                    dtype=bool,
                ),
                players=(Player2D(0, 0, True), Player2D(4, 2, True)),
            )
        ),
        hero_index=0,
        opponent_index=1,
    ),
    PovGameState(
        game_state=from_2d_game_state(
            GameState2D(
                grid=np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1],
                    ],
                    dtype=bool,
                ),
                players=(Player2D(3, 0, True), Player2D(3, 4, True)),
            )
        ),
        hero_index=0,
        opponent_index=1,
    ),
]


def test_minimax():

    g0 = TIES_5X5[0]

    model = RandomTronModel()

    mmc0 = MinimaxContext(
        model, g0.hero_index, g0.opponent_index, win_magnitude=1000.0, debug_stack=[]
    )

    mmr0 = basic_minimax(g0.game_state, 4, is_hero=True, context=mmc0)

    assert mmr0.principal_variation == Direction.RIGHT
    assert mmr0.evaluation == 0.0

    t2 = from_2d_game_state(
        GameState2D(
            grid=np.array(
                [
                    [
                        1,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                    ],
                    [0, 0, 1],
                ],
                dtype=bool,
            ),
            players=(Player2D(0, 0, True), Player2D(2, 2, True)),
        )
    )

    t2mmc = MinimaxContext(
        RandomTronModel(), hero_index=0, opponent_index=1, debug_stack=[]
    )

    t2mmr = basic_minimax(t2, 4, True, context=t2mmc)

    assert (t2mmr.principal_variation == Direction.RIGHT) or (
        t2mmr.principal_variation == Direction.DOWN
    )
    assert t2mmr.evaluation == 0.0


    
    t3 = from_2d_game_state(
        GameState2D(
            grid=np.array(
                [
                    [
                        1,
                        0,
                        0,
                    ],
                    [
                        1,
                        1,
                        1,
                    ],
                    [0, 1, 0],
                ],
                dtype=bool,
            ),
            players=(Player2D(0, 0, True), Player2D(2, 1, True)),
        )
    )

    t3mmc = MinimaxContext(
        RandomTronModel(), hero_index=0, opponent_index=1, debug_stack=[]
    )

    t3mmr = basic_minimax(t3, 2, True, context=t3mmc)

    assert t3mmr.principal_variation == Direction.RIGHT
    assert t3mmr.evaluation == t3mmc.win_magnitude


# def test_minimax():


#     tron_model = RandomTronModel()
#     context = MinimaxContext(tron_model, maximizing_player=0, minimizing_player=1)


#     for i in range(100):

#         for depth in range(2):


#             game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True)
#             game_status = get_status(game).status

#             while game_status == GameStatus.IN_PROGRESS:

#                 basic_mm_result = basic_minimax(game, depth=depth, is_maximizing_player=True, context=context)
#                 ab_mm_result = minimax_alpha_beta_eval_all(game, depth=depth, is_maximizing_player=True, context=context)


#                 # print(f"{basic_mm_result.principal_variation=}, {ab_mm_result.principal_variation=}")

#                 assert basic_mm_result.evaluation == ab_mm_result.evaluation
#                 if basic_mm_result.principal_variation != ab_mm_result.principal_variation:
#                     assert (basic_mm_result.evaluation / 1000.0).is_integer()
#                     print(f"{basic_mm_result.evaluation=}, {ab_mm_result.evaluation=}")


#                 p1_dir = Direction.UP if basic_mm_result.principal_variation is None else basic_mm_result.principal_variation
#                 p2_dir = choose_direction_random(game, player_index=1)

#                 game = next(
#                     game, directions=(p1_dir, p2_dir)
#                 )

#                 game_status = get_status(game).status
