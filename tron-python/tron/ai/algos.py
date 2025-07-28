import random
import torch
import numpy as np
from abc import ABC, abstractmethod

import tron
from tron import Direction, GameState

from tron.ai.tron_model import TronModel
#from tron.ai.minimax import minimax_alpha_beta_eval_all, basic_minimax, minimax_dumb


def choose_direction_model_naive(
    model: TronModel, game: GameState, player_index: int
) -> Direction:

    assert game.players[player_index].can_move

    possible_directions = tron.get_possible_directions(game, player_index)

    if len(possible_directions) == 0:
        # Maybe return None instead?
        return Direction.UP
    else:

        game_states_to_eval = []

        for direction in possible_directions:
            directions = [Direction.UP] * len(game.players)
            directions[player_index] = direction
            game_states_to_eval.append(
                tron.next(game, directions))

        

        evaluations = model.run_inference(game_states_to_eval, player_index)

        return possible_directions[np.argmax(evaluations)]

def choose_direction_random(game: GameState, player_index: int) -> Direction:

    assert game.players[player_index].can_move

    possible_directions = tron.get_possible_directions(game, player_index)

    if len(possible_directions) == 0:
        # Maybe return None instead?
        return Direction.UP
    else:
        return random.choice(possible_directions)
        


# def choose_direction_minimax(
#     model: tronModelAbstract,
#     game: tron,
#     player_index: int,
#     opponent_index: int,
#     depth: int,
#     do_alpha_beta: bool = True,
#     do_lru_eval: bool = True,
# ) -> DirectionUpdate:

#     assert game.players[player_index].can_move

#     hero_possible_directions = tron.get_possible_directions(game, player_index)
#     opponent_possible_directions = tron.get_possible_directions(game, opponent_index)

#     if len(hero_possible_directions) == 0:
#         # Hero is effed, just die
#         return DirectionUpdate(
#             direction=game.players[player_index].direction,
#             player_index=player_index,
#         )

#     if len(opponent_possible_directions) == 0:
#         # Hero wins, choose any possible move
#         return DirectionUpdate(
#             direction=hero_possible_directions[0],
#             player_index=player_index,
#         )

#     opponent_best_evals = []

#     for hero_direction in hero_possible_directions:

#         opponent_best_eval = float("inf")

#         for opponent_direction in opponent_possible_directions:

#             dir_updates = (
#                 DirectionUpdate(hero_direction, player_index),
#                 DirectionUpdate(opponent_direction, opponent_index),
#             )

#             new_state = tron.next(game, dir_updates)

#             if do_alpha_beta:

#                 if do_lru_eval:
#                     move_value = minimax_alpha_beta_eval_all(
#                         model,
#                         new_state,
#                         depth - 1,
#                         alpha=float("-inf"),
#                         beta=float("inf"),
#                         is_maximizing_player=True,
#                         maximizing_player_index=player_index,
#                         minimizing_player_index=opponent_index,
#                     )
#                 else:
#                     move_value = minimax_alpha_beta(
#                         model,
#                         new_state,
#                         depth - 1,
#                         alpha=float("-inf"),
#                         beta=float("inf"),
#                         is_maximizing_player=True,
#                         maximizing_player_index=player_index,
#                         minimizing_player_index=opponent_index,
#                     )
#             else:

#                 move_value = basic_minimax(
#                     model,
#                     new_state,
#                     depth - 1,
#                     is_maximizing_player=True,
#                     maximizing_player_index=player_index,
#                     minimizing_player_index=opponent_index,
#                 )

#             if move_value < opponent_best_eval:
#                 opponent_best_eval = move_value

#         opponent_best_evals.append(opponent_best_eval)

#     return DirectionUpdate(
#         hero_possible_directions[np.argmax(opponent_best_evals)], player_index
#     )


# def choose_direction_minimax_alpha_beta(
#     model: tronModelAbstract,
#     game: tron,
#     player_index: int,
#     opponent_index: int,
#     depth: int,
# ) -> DirectionUpdate:

#     assert game.players[player_index].can_move

#     hero_possible_directions = tron.get_possible_directions(game, player_index)
#     opponent_possible_directions = tron.get_possible_directions(game, opponent_index)

#     if len(hero_possible_directions) == 0:
#         # Hero is effed, just die
#         return DirectionUpdate(
#             direction=game.players[player_index].direction,
#             player_index=player_index,
#         )

#     if len(opponent_possible_directions) == 0:
#         # Hero wins, choose any possible move
#         return DirectionUpdate(
#             direction=hero_possible_directions[0],
#             player_index=player_index,
#         )

#     opponent_best_evals = []

#     for hero_direction in hero_possible_directions:

#         opponent_best_eval = float("inf")

#         for opponent_direction in opponent_possible_directions:

#             dir_updates = (
#                 DirectionUpdate(hero_direction, player_index),
#                 DirectionUpdate(opponent_direction, opponent_index),
#             )

#             new_state = tron.lru_cache_next(game, dir_updates)

#             move_value = minimax_alpha_beta_eval_all(
#                 model,
#                 new_state,
#                 depth - 1,
#                 alpha=float("-inf"),
#                 beta=float("inf"),
#                 is_maximizing_player=True,
#                 maximizing_player_index=player_index,
#                 minimizing_player_index=opponent_index,
#             )

#             if move_value < opponent_best_eval:
#                 opponent_best_eval = move_value

#         opponent_best_evals.append(opponent_best_eval)

#     return DirectionUpdate(
#         hero_possible_directions[np.argmax(opponent_best_evals)], player_index
#     )


# def choose_direction_minimax_dumb(
#     game: tron,
#     player_index: int,
#     opponent_index: int,
#     depth: int,
# ) -> DirectionUpdate:

#     assert game.players[player_index].can_move

#     hero_possible_directions = tron.get_possible_directions(game, player_index)
#     opponent_possible_directions = tron.get_possible_directions(game, opponent_index)

#     if len(hero_possible_directions) == 0:
#         # Hero is effed, just die
#         return DirectionUpdate(
#             direction=game.players[player_index].direction,
#             player_index=player_index,
#         )

#     if len(opponent_possible_directions) == 0:
#         # Hero wins, choose any possible move
#         return DirectionUpdate(
#             direction=hero_possible_directions[0],
#             player_index=player_index,
#         )

#     opponent_best_evals = []

#     for hero_direction in hero_possible_directions:

#         opponent_best_eval = float("inf")

#         for opponent_direction in opponent_possible_directions:

#             dir_updates = (
#                 DirectionUpdate(hero_direction, player_index),
#                 DirectionUpdate(opponent_direction, opponent_index),
#             )

#             new_state = GameState.lru_cache_next(game, dir_updates)

#             move_value = minimax_dumb(
#                 new_state,
#                 depth - 1,
#                 is_maximizing_player=True,
#                 maximizing_player_index=player_index,
#                 minimizing_player_index=opponent_index,
#             )

#             if move_value < opponent_best_eval:
#                 opponent_best_eval = move_value

#         opponent_best_evals.append(opponent_best_eval)

#     return DirectionUpdate(
#         hero_possible_directions[np.argmax(opponent_best_evals)], player_index
#     )
