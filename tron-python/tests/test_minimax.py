import sys

import numpy as np

from tron.game import GameState, Player, GameStatus, StatusInfo, next, get_status, Direction
from tron.ai.tron_model import RandomTronModel
from tron.ai.minimax import minimax_alpha_beta_eval_all, basic_minimax, MinimaxContext
from tron.ai.algos import choose_direction_random


def test_minimax():



    tron_model = RandomTronModel()
    context = MinimaxContext(tron_model, maximizing_player=0, minimizing_player=1)


    for i in range(100):

        for depth in range(2):


            game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True)
            game_status = get_status(game).status

            while game_status == GameStatus.IN_PROGRESS:

                basic_mm_result = basic_minimax(game, depth=depth, is_maximizing_player=True, context=context)
                ab_mm_result = minimax_alpha_beta_eval_all(game, depth=depth, is_maximizing_player=True, context=context)


                # print(f"{basic_mm_result.principal_variation=}, {ab_mm_result.principal_variation=}")

                assert basic_mm_result.evaluation == ab_mm_result.evaluation
                if basic_mm_result.principal_variation != ab_mm_result.principal_variation:
                    assert (basic_mm_result.evaluation / 1000.0).is_integer()
                    print(f"{basic_mm_result.evaluation=}, {ab_mm_result.evaluation=}")



            
                p1_dir = Direction.UP if basic_mm_result.principal_variation is None else basic_mm_result.principal_variation
                p2_dir = choose_direction_random(game, player_index=1)

                game = next(
                    game, directions=(p1_dir, p2_dir)
                )

                game_status = get_status(game).status



