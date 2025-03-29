import numpy as np
import sys
from typing import Optional
from cachetools import LRUCache, cached
from dataclasses import dataclass

from game import tron
from game.tron import GameState, GameStatus, Direction

from ai.tron_model import TronModelAbstract


# TODO: Change args to player pos, opponent pos, direction?
def heuristic_towards_opponent(
    game: GameState, hero_index: int, opponent_index: int, direction: Direction
):
    direction.value[0]

    opponent_right = game.players[opponent_index].col > game.players[hero_index].col
    opponent_down = game.players[opponent_index].row > game.players[hero_index].row

    if opponent_right and direction == Direction.RIGHT:
        return 1.0
    elif opponent_down and direction == Direction.DOWN:
        return 1.0

    opponent_left = game.players[opponent_index].col < game.players[hero_index].col
    opponent_up = game.players[opponent_index].row < game.players[hero_index].row

    if opponent_left and direction == Direction.LEFT:
        return 1.0
    if opponent_up and direction == Direction.UP:
        return 1.0

    return -1.0


# TODO: This should probably be in different spot
# Initialize an LRU cache with 20 mil max size
cache = LRUCache(maxsize=20000000)



@cached(cache)
def lru_eval(model: TronModelAbstract, game, player_index):
    return model.run_inference([game], player_index)[0]


@dataclass
class MinimaxDebugState:
    game_state: GameState
    depth: int
    is_maximizing_player: bool
    alpha: int
    beta: int
    maximizing_player_move: Direction


# For debugging minimax
# minimax_stack = []


@dataclass
class MinimaxResult:
    evaluation: float
    principal_variation: Optional[Direction] = None


@dataclass
class MinimaxContext:
    model: TronModelAbstract
    maximizing_player: int
    minimizing_player: int
    debug_mode: bool = False


def minimax_alpha_beta_eval_all(
    game_state: GameState,
    depth: int,
    is_maximizing_player: bool,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    maximizing_player_move: Direction = None,
    context: MinimaxContext = None,
) -> MinimaxResult:
    assert context is not None, "Context must be passed"
    maximizing_player = context.maximizing_player
    minimizing_player = context.minimizing_player

    # if debug_mode:
    #     raise NotImplementedError()
    #     minimax_stack.append(
    #         MinimaxDebugState(
    #             game_state,
    #             depth,
    #             is_maximizing_player,
    #             alpha,
    #             beta,
    #             maximizing_player_move,
    #         )
    #     )

    game_status = tron.get_status(game_state)

    if game_status.status == GameStatus.TIE:
        return MinimaxResult(0.0, None)
    elif game_status.status == GameStatus.WINNER:
        raise RuntimeError("Winning terminal state should never be reached here.")


    if depth == 0:
        return MinimaxResult(lru_eval(context.model, game_state, maximizing_player), None)

    if is_maximizing_player:

        possible_directions = tron.get_possible_directions(
            game_state, maximizing_player
        )

        # Handle no possible directions - maybe other player doesn't either
        if len(possible_directions) == 0:

            opponent_possible_directions = tron.get_possible_directions(
                game_state, minimizing_player
            )

            # This is a tie
            if len(opponent_possible_directions) == 0:
                return MinimaxResult(0.0, None)
            # Guaranteed loss for maximizing player, eval is relative to depth.
            # Losses at deeper depths will be preffered to those at shallower depths.
            # Encourages bot to stay alive because a human could easily not play optimally.
            else:
                return MinimaxResult(-100.0 * depth, None)

        # Heuristic sorting (descending order for maximizing player)
        sorted_directions = sorted(
            possible_directions,
            key=lambda dir: heuristic_towards_opponent(
                game_state, maximizing_player, minimizing_player, dir
            ),
            reverse=True,
        )

        max_eval = float("-inf")
        for direction in sorted_directions:
            mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
                game_state,
                depth,
                is_maximizing_player=False,
                alpha=alpha,
                beta=beta,
                maximizing_player_move=direction,
                context=context,
            )

            if mm_result.evaluation > max_eval:
                max_eval = mm_result.evaluation
                best_dir = direction

            alpha = max(alpha, mm_result.evaluation)
            if beta <= alpha:
                break

        return MinimaxResult(max_eval, best_dir)
    else:
        possible_directions = tron.get_possible_directions(
            game_state, minimizing_player
        )

        # Guaranteed win for maximizing player, eval is relative to depth
        if len(possible_directions) == 0:
            return MinimaxResult(100.0 * depth, None)



        child_states = []

        for direction in possible_directions:

            directions = [None, None]
            directions[maximizing_player] = maximizing_player_move
            directions[minimizing_player] = direction

            child_states.append(tron.next(
                game_state,
                directions,
            ))


        # # Sort child states by eval if cached, else use heuristic
        def sort_possibilities(dir_state_tup: tuple[Direction, GameState]):

            direction, child_state = dir_state_tup

            arg_tup = (context.model, child_state, maximizing_player)

            # Prioritize previously evaluated positions over the heuristic
            if arg_tup in cache:
                # By subtracting 100 we guarantee previously evaluated positions are considered first
                return lru_eval(*arg_tup) - 100.0
            else:
                # From perspective of minimizing player
                # Multiply by -1 because lower = better
                return -1 * heuristic_towards_opponent(
                    child_state,
                    hero_index=minimizing_player,
                    opponent_index=maximizing_player,
                    direction=direction,
                )

        # Ascending order (for minimizing player, lowest evaluations are most promising)
        sorted_possible_directions, sorted_child_states = zip(
            *sorted(zip(possible_directions, child_states), key=sort_possibilities)
        )

        # sorted_child_states = sorted(
        #     child_states,
        #     key=lambda child_state: lru_eval(
        #         model, child_state, maximizing_player_index
        #     ),
        # )

        min_eval = float("inf")

        for direction, child_state in zip(sorted_possible_directions, sorted_child_states):
            mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
                child_state,
                depth - 1,
                is_maximizing_player=True,
                alpha=alpha,
                beta=beta,
                context=context,
            )

            if mm_result.evaluation < min_eval:
                min_eval = mm_result.evaluation
                best_dir = direction

            beta = min(beta, mm_result.evaluation)

            if beta <= alpha:
                break

        return MinimaxResult(min_eval, best_dir)


# # TODO: Fix not handling ties correctly
# def basic_minimax(
#     model: TronModelAbstract,
#     game_state: GameState,
#     depth,
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
#         for direction in GameState.get_possible_directions(
#             game_state, maximizing_player_index
#         ):
#             eval = basic_minimax(
#                 model,
#                 game_state,
#                 depth,
#                 is_maximizing_player=False,
#                 maximizing_player_index=maximizing_player_index,
#                 minimizing_player_index=minimizing_player_index,
#                 maximizing_player_move=direction,
#             )
#             max_eval = max(max_eval, eval)
#         return max_eval
#     else:
#         min_eval = float("inf")
#         for direction in tron.get_possible_directions(
#             game_state, minimizing_player_index
#         ):
#             child_state = tron.next(
#                 game_state,
#                 direction_updates=(
#                     DirectionUpdate(maximizing_player_move, maximizing_player_index),
#                     DirectionUpdate(direction, minimizing_player_index),
#                 ),
#             )
#             eval = basic_minimax(
#                 model,
#                 child_state,
#                 depth - 1,
#                 is_maximizing_player=True,
#                 maximizing_player_index=maximizing_player_index,
#                 minimizing_player_index=minimizing_player_index,
#             )
#             min_eval = min(min_eval, eval)
#         return min_eval


# def minimax_dumb(
#     game_state: GameState,
#     depth,
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
#         return 0

#     if is_maximizing_player:
#         max_eval = -float("inf")
#         for direction in GameState.get_possible_directions(
#             game_state, maximizing_player_index
#         ):
#             eval = minimax_dumb(
#                 game_state,
#                 depth,
#                 is_maximizing_player=False,
#                 maximizing_player_index=maximizing_player_index,
#                 minimizing_player_index=minimizing_player_index,
#                 maximizing_player_move=direction,
#             )
#             max_eval = max(max_eval, eval)
#         return max_eval
#     else:
#         min_eval = float("inf")
#         for direction in tron.get_possible_directions(
#             game_state, minimizing_player_index
#         ):
#             child_state = tron.next(
#                 game_state,
#                 direction_updates=(
#                     DirectionUpdate(maximizing_player_move, maximizing_player_index),
#                     DirectionUpdate(direction, minimizing_player_index),
#                 ),
#             )
#             eval = minimax_dumb(
#                 child_state,
#                 depth - 1,
#                 is_maximizing_player=True,
#                 maximizing_player_index=maximizing_player_index,
#                 minimizing_player_index=minimizing_player_index,
#             )
#             min_eval = min(min_eval, eval)
#         return min_eval
