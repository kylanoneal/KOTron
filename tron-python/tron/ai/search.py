import numpy as np
import sys
from typing import Optional
from cachetools import LRUCache, cached
from dataclasses import dataclass

import tron
from tron.game import  GameState, StatusInfo, GameStatus, Direction

from tron.ai.tron_model import TronModel, HeroGameState

@dataclass
class MovePair:
    maximizing_player_move: Direction
    minimizing_player_move: Direction

class PvTable:

    def __init__(self, max_depth: int):

        self.max_depth = max_depth
        self._pv_table = [[None for _ in range(max_depth)] for _ in range(max_depth)]

    def get_move_pair(self, depth: int, ply: int) -> MovePair:

        assert depth > 0 and ply >= 0

        return self._pv_table[depth - 1][ply]
    
    def set_move_pair(self, depth:int, ply:int, move_pair: MovePair):
        assert depth > 0 and ply >= 0
        self._pv_table[depth - 1][ply] = move_pair

        

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
def lru_eval(model: TronModel, game, player_index):
    return model.run_inference([HeroGameState(game, player_index)])[0]


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
    model: TronModel
    maximizing_player: int
    minimizing_player: int
    debug_mode: bool = False


def alphabeta(
    game_state: GameState,
    depth: int,
    ply: int,
    pv_table: PvTable,
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

    status_info: StatusInfo = tron.get_status(game_state)

    if status_info.status == GameStatus.TIE:
        return MinimaxResult(0.0, None)
    elif status_info.status == GameStatus.WINNER:
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
                return MinimaxResult(-1000.0 * depth, None)

        # Heuristic sorting (descending order for maximizing player)
        sorted_directions = sorted(
            possible_directions,
            key=lambda dir: heuristic_towards_opponent(
                game_state, maximizing_player, minimizing_player, dir
            ),
            reverse=True,
        )

        # Explore PV first if we have it
        pv = pv_table.get_move_pair(depth, ply)

        if pv is not None:
            if pv.maximizing_player_move in sorted_directions:
                sorted_directions.remove(pv.maximizing_player_move)
                sorted_directions.insert(0, pv.maximizing_player_move)
            


        max_eval = float("-inf")
        for direction in sorted_directions:
            mm_result: MinimaxResult = alphabeta(
                game_state,
                depth,
                ply,
                pv_table,
                is_maximizing_player=False,
                alpha=alpha,
                beta=beta,
                maximizing_player_move=direction,
                context=context,
            )

            if mm_result.evaluation > max_eval:
                max_eval = mm_result.evaluation
                best_dir = direction

                # PV bookkeeping
                pv_table.set_move_pair(depth, ply, MovePair(maximizing_player_move=best_dir, minimizing_player_move=mm_result.principal_variation))

                # copy the rest of the line from the child
                # TODO: Maybe make this a PvTable method
                if depth > 1:
                    for next_ply in range(ply+1, pv_table.max_depth):

                        child_move_pair = pv_table.get_move_pair(depth - 1, next_ply)

                        if child_move_pair is None:
                            break
                        else:
                            pv_table.set_move_pair(depth, next_ply, child_move_pair)

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
            return MinimaxResult(1000.0 * depth, None)



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
                return lru_eval(*arg_tup) - 1000.0
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
        sorted_directions, sorted_child_states = zip(
            *sorted(zip(possible_directions, child_states), key=sort_possibilities)
        )

        sorted_directions = list(sorted_directions)
        # Explore PV first if we have it
        pv = pv_table.get_move_pair(depth, ply)

        if pv is not None:
            if pv.minimizing_player_move in sorted_directions:
                sorted_directions.remove(pv.minimizing_player_move)
                sorted_directions.insert(0, pv.minimizing_player_move)
            

        # sorted_child_states = sorted(
        #     child_states,
        #     key=lambda child_state: lru_eval(
        #         model, child_state, maximizing_player_index
        #     ),
        # )

        min_eval = float("inf")

        for direction, child_state in zip(sorted_directions, sorted_child_states):
            mm_result: MinimaxResult = alphabeta(
                child_state,
                depth - 1,
                ply + 1,
                pv_table,
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




def iter_deepening_ab(root_game_state, max_depth: int, mm_context: MinimaxContext):

    pv_table = PvTable(max_depth)

    # Utilize PV moves before the rest
    for depth in range(1, max_depth+1):

        mm_result = alphabeta(root_game_state, depth, ply=0, pv_table=pv_table, is_maximizing_player=True, context=mm_context)

    if mm_result.principal_variation is not None:
        assert mm_result.principal_variation == pv_table.get_move_pair(max_depth, 0).maximizing_player_move

    return mm_result

