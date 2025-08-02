import sys
import numpy as np
from copy import deepcopy

from tron.game import GameState, Player, GameStatus, StatusInfo, next, get_status, Direction
from tron.ai.tron_model import RandomTronModel, HeroGameState
from tron.ai.nnue import NnueTronModel
from tron.ai.minimax import minimax_alpha_beta_eval_all, basic_minimax, MinimaxContext
from tron.ai.algos import choose_direction_random



def test_nnue():


    nnue_model = NnueTronModel(10, 10)
    nnue_model.eval()

    game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True)

    initial_eval = nnue_model.run_inference([HeroGameState(game, 0)])[0]

    # for _ in range(10000):


    #     assert np.isclose(nnue_model.run_inference([HeroGameState(game, 0)])[0], initial_eval, rtol=1e-6)

    for _ in range(1000):
        _game = GameState.new_game(2, 10, 10, random_starts=True)

        if _ in [777, 333, 555, 888]:
            nnue_model.reset_acc()
        nnue_model.run_inference([HeroGameState(_game, 0)])

    #nnue_model.reset_acc()
    assert np.isclose(nnue_model.run_inference([HeroGameState(game, 0)])[0], initial_eval, rtol=1e-7)

    # for _ in range(10000):
    #     _game = GameState.new_game(2, 10, 10, random_starts=True)

    #     nnue_model.run_inference([HeroGameState(_game, 0)])
    #     nnue_model.reset_acc()

    # assert np.isclose(nnue_model.run_inference([HeroGameState(game, 0)])[0], initial_eval, rtol=1e-6)



# def test_nnue_minimax():



#     nnue_model = NnueTronModel(10, 10)
#     initial_state_dict = nnue_model.state_dict()

#     nnue_context = MinimaxContext(nnue_model, maximizing_player=0, minimizing_player=1)





#     for i in range(100):

#         for depth in range(2):


#             game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True)
#             game_status = get_status(game).status

#             static_nnue_model = NnueTronModel(10, 10)
#             static_nnue_model.load_state_dict(initial_state_dict)
#             static_nnue_model.reset_acc()
#             static_nnue_context = MinimaxContext(static_nnue_model, maximizing_player=0, minimizing_player=1)

#             print(f"{static_nnue_model.run_inference([HeroGameState(game, 0)])}")
#             print(f"{nnue_model.run_inference([HeroGameState(game, 0)])}")

#             while game_status == GameStatus.IN_PROGRESS:


#                 nnue_mm_result = minimax_alpha_beta_eval_all(game, depth=depth, is_maximizing_player=True, context=nnue_context)


#                 static_mm_result = minimax_alpha_beta_eval_all(game, depth=depth, is_maximizing_player=True, context=static_nnue_context)

#                 # print(f"{nnue_mm_result.principal_variation=}, {static_mm_result.principal_variation=}")

#                 #assert nnue_mm_result.evaluation == static_mm_result.evaluation

#                 assert np.isclose(nnue_mm_result.evaluation, static_mm_result.evaluation, rtol=1e-4, atol=0)
#                 if nnue_mm_result.principal_variation != static_mm_result.principal_variation:
#                     assert (nnue_mm_result.evaluation / 1000.0).is_integer()
#                     print(f"{nnue_mm_result.evaluation=}, {static_mm_result.evaluation=}")

            
#                 p1_dir = Direction.UP if nnue_mm_result.principal_variation is None else nnue_mm_result.principal_variation
#                 p2_dir = choose_direction_random(game, player_index=1)

#                 game = next(
#                     game, directions=(p1_dir, p2_dir)
#                 )

#                 game_status = get_status(game).status



