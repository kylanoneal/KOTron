import numpy as np
import sys
from cachetools import LRUCache, cached

from game.tron import Tron, GameStatus, DirectionUpdate
from ai.tron_model import TronModelAbstract


# TODO: This should probably be in different spot
# Initialize an LRU cache with a maximum size
cache = LRUCache(maxsize=sys.maxsize)


@cached(cache)
def lru_eval(model: TronModelAbstract, game, player_index):
    return model.run_inference([game], player_index)[0]


def minimax_alpha_beta_eval_all(
    model: TronModelAbstract,
    game_state: Tron,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing_player: bool,
    maximizing_player_index,
    minimizing_player_index,
    maximizing_player_move=None,
) -> float:

    if game_state.status != GameStatus.IN_PROGRESS:

        if game_state.status == GameStatus.TIE:
            return 0.0
        else:
            raise RuntimeError("Winning terminal state should never be reached here.")
            if GameStatus.index_of_winner(game_state.status) == maximizing_player_index:
                return float("inf")
            else:
                return float("-inf")

    if depth == 0:
        return model.run_inference([game_state], maximizing_player_index)[0]

    if is_maximizing_player:
        max_eval = float("-inf")

        possible_directions = Tron.get_possible_directions(
            game_state, maximizing_player_index
        )

        # Handle no possible directions - maybe other player doesn't either
        if len(possible_directions) == 0:
            opponent_possible_directions = Tron.get_possible_directions(
                game_state, minimizing_player_index
            )

            # This is a tie
            if len(opponent_possible_directions) == 0:
                return 0.0
            # Loss
            else:
                return float("-inf")

        for direction in possible_directions:

            # TODO: Heuristic move ordering here? Doesn't really make sense
            # to eval a position without both players' move
            eval = minimax_alpha_beta_eval_all(
                model,
                game_state,
                depth,
                alpha,
                beta,
                is_maximizing_player=False,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
                maximizing_player_move=direction,
            )
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval
    else:
        min_eval = float("inf")

        possible_directions = Tron.get_possible_directions(
            game_state, minimizing_player_index
        )

        if len(possible_directions) == 0:
            return float("inf")

        child_states = []
        cached_model_evals = []

        child_states = [
            Tron.lru_cache_next(
                game_state,
                direction_updates=(
                    DirectionUpdate(maximizing_player_move, maximizing_player_index),
                    DirectionUpdate(direction, minimizing_player_index),
                ),
            )
            for direction in possible_directions
        ]

        # Sort child states by cached eval
        # Ascending oder (for minimizing player, lowest evaluations are most promising)
        sorted_child_states = sorted(child_states, key=lambda child_state: lru_eval(model, child_state, maximizing_player_index))

        for child_state in sorted_child_states:
            eval = minimax_alpha_beta_eval_all(
                model,
                child_state,
                depth - 1,
                alpha,
                beta,
                is_maximizing_player=True,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
            )
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_eval


# TODO: Fix not handling ties correctly
# def minimax_alpha_beta(
#     model: TronModelAbstract,
#     game_state: Tron,
#     depth: int,
#     alpha: float,
#     beta: float,
#     is_maximizing_player: bool,
#     maximizing_player_index,
#     minimizing_player_index,
#     maximizing_player_move=None,
# ) -> float:

#     if game_state.status != GameStatus.IN_PROGRESS:

#         if game_state.status == GameStatus.TIE:
#             return 0.0
#         else:
#             if GameStatus.index_of_winner(game_state.status) == maximizing_player_index:
#                 return float("inf")
#             else:
#                 return float("-inf")

#     if depth == 0:
#         return model.run_inference([game_state], maximizing_player_index)[0]

#     if is_maximizing_player:
#         max_eval = -float("inf")
#         for direction in Tron.get_possible_directions(
#             game_state, maximizing_player_index
#         ):

#             eval = minimax_alpha_beta(
#                 model,
#                 game_state,
#                 depth,
#                 alpha,
#                 beta,
#                 is_maximizing_player=False,
#                 maximizing_player_index=maximizing_player_index,
#                 minimizing_player_index=minimizing_player_index,
#                 maximizing_player_move=direction,
#             )
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break

#         return max_eval
#     else:
#         min_eval = float("inf")
#         for direction in Tron.get_possible_directions(
#             game_state, minimizing_player_index
#         ):
#             child_state = Tron.lru_cache_next(
#                 game_state,
#                 direction_updates=(
#                     DirectionUpdate(maximizing_player_move, maximizing_player_index),
#                     DirectionUpdate(direction, minimizing_player_index),
#                 ),
#             )
#             eval = minimax_alpha_beta(
#                 model,
#                 child_state,
#                 depth - 1,
#                 alpha,
#                 beta,
#                 is_maximizing_player=True,
#                 maximizing_player_index=maximizing_player_index,
#                 minimizing_player_index=minimizing_player_index,
#             )
#             min_eval = min(min_eval, eval)
#             beta = min(beta, eval)

#             if beta <= alpha:
#                 break

#         return min_eval

# TODO: Fix not handling ties correctly
def basic_minimax(
    model: TronModelAbstract,
    game_state: Tron,
    depth,
    is_maximizing_player: bool,
    maximizing_player_index,
    minimizing_player_index,
    maximizing_player_move=None,
) -> float:

    if game_state.status != GameStatus.IN_PROGRESS:

        if game_state.status == GameStatus.TIE:
            return 0.0
        else:
            if GameStatus.index_of_winner(game_state.status) == maximizing_player_index:
                return float("inf")
            else:
                return float("-inf")

    if depth == 0:
        return model.run_inference([game_state], maximizing_player_index)[0]

    if is_maximizing_player:
        max_eval = -float("inf")
        for direction in Tron.get_possible_directions(
            game_state, maximizing_player_index
        ):
            eval = basic_minimax(
                model,
                game_state,
                depth,
                is_maximizing_player=False,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
                maximizing_player_move=direction,
            )
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for direction in Tron.get_possible_directions(
            game_state, minimizing_player_index
        ):
            child_state = Tron.next(
                game_state,
                direction_updates=(
                    DirectionUpdate(maximizing_player_move, maximizing_player_index),
                    DirectionUpdate(direction, minimizing_player_index),
                ),
            )
            eval = basic_minimax(
                model,
                child_state,
                depth - 1,
                is_maximizing_player=True,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
            )
            min_eval = min(min_eval, eval)
        return min_eval


def minimax_dumb(
    game_state: Tron,
    depth,
    is_maximizing_player: bool,
    maximizing_player_index,
    minimizing_player_index,
    maximizing_player_move=None,
) -> float:

    if game_state.status != GameStatus.IN_PROGRESS:

        if game_state.status == GameStatus.TIE:
            return 0.0
        else:
            if GameStatus.index_of_winner(game_state.status) == maximizing_player_index:
                return float("inf")
            else:
                return float("-inf")

    if depth == 0:
        return 0

    if is_maximizing_player:
        max_eval = -float("inf")
        for direction in Tron.get_possible_directions(
            game_state, maximizing_player_index
        ):
            eval = minimax_dumb(
                game_state,
                depth,
                is_maximizing_player=False,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
                maximizing_player_move=direction,
            )
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for direction in Tron.get_possible_directions(
            game_state, minimizing_player_index
        ):
            child_state = Tron.next(
                game_state,
                direction_updates=(
                    DirectionUpdate(maximizing_player_move, maximizing_player_index),
                    DirectionUpdate(direction, minimizing_player_index),
                ),
            )
            eval = minimax_dumb(
                child_state,
                depth - 1,
                is_maximizing_player=True,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
            )
            min_eval = min(min_eval, eval)
        return min_eval
