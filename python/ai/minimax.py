import numpy as np
import sys
from typing import Union
from cachetools import LRUCache, cached

from game.tron import Tron, GameStatus, DirectionUpdate, Direction
from AI.tron_model import TronModelAbstract

# TODO: Change args to player pos, opponent pos, direction?
def heuristic_towards_opponent(game: Tron, hero_index: int, opponent_index: int, direction: Direction):
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
# Initialize an LRU cache with a maximum size
cache = LRUCache(maxsize=sys.maxsize)


@cached(cache)
def lru_eval(model: TronModelAbstract, game, player_index):
    return model.run_inference([game], player_index)[0]


def minimax_alpha_beta_eval_all(
    model: TronModelAbstract,
    game_state: Tron,
    depth: int,
    maximizing_player_index,
    minimizing_player_index,
    is_maximizing_player: bool,
    is_root: bool = False,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    maximizing_player_move=None,
) -> Union[float, DirectionUpdate]:

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
        return lru_eval(model, game_state, maximizing_player_index)

    if is_maximizing_player:
        max_eval = float("-inf")

        possible_directions = Tron.get_possible_directions(
            game_state, maximizing_player_index
        )

        # Handle no possible directions - maybe other player doesn't either
        if len(possible_directions) == 0:

            # Nowhere to go, just return up
            if is_root:
                return DirectionUpdate(Direction.UP, maximizing_player_index)
            opponent_possible_directions = Tron.get_possible_directions(
                game_state, minimizing_player_index
            )

            # This is a tie
            if len(opponent_possible_directions) == 0:
                return 0.0
            # Guaranteed loss for maximizing player, eval is relative to depth.
            # Losses at deeper depths will be preffered to those at shallower depths.
            # Encourages bot to stay alive because a human could easily not play optimally.
            else:
                return -100.0 * depth
            

        # Heuristic sorting (descending order for maximizing player)
        sorted_directions = sorted(
            possible_directions,
            key=lambda dir: heuristic_towards_opponent(
                game_state, maximizing_player_index, minimizing_player_index, dir
            ),
            reverse=True
        )

        for direction in sorted_directions:

            eval = minimax_alpha_beta_eval_all(
                model,
                game_state,
                depth,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
                is_maximizing_player=False,
                alpha=alpha,
                beta=beta,
                maximizing_player_move=direction,
            )

            if eval > max_eval:
                max_eval = eval
                best_dir = direction

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return (
            max_eval if not is_root else DirectionUpdate(best_dir, maximizing_player_index)
        )
    else:
        min_eval = float("inf")

        possible_directions = Tron.get_possible_directions(
            game_state, minimizing_player_index
        )

        # Guaranteed win for maximizing player, eval is relative to depth
        if len(possible_directions) == 0:
            return 100.0 * depth


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
        sorted_child_states = sorted(
            child_states,
            key=lambda child_state: lru_eval(
                model, child_state, maximizing_player_index
            ),
        )

        sorted_child_states = child_states

        for child_state in sorted_child_states:
            eval = minimax_alpha_beta_eval_all(
                model,
                child_state,
                depth - 1,
                maximizing_player_index=maximizing_player_index,
                minimizing_player_index=minimizing_player_index,
                is_maximizing_player=True,
                alpha=alpha,
                beta=beta,
            )
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_eval


# TODO: Fix not handling ties correctly
def minimax_alpha_beta(
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

            eval = minimax_alpha_beta(
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
        for direction in Tron.get_possible_directions(
            game_state, minimizing_player_index
        ):
            child_state = Tron.lru_cache_next(
                game_state,
                direction_updates=(
                    DirectionUpdate(maximizing_player_move, maximizing_player_index),
                    DirectionUpdate(direction, minimizing_player_index),
                ),
            )
            eval = minimax_alpha_beta(
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
