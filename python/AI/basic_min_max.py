from game.ko_tron import KOTron, GameStatus, DirectionUpdate
from AI.tron_model import TronModelAbstract
import numpy as np

def minimax(
    model: TronModelAbstract,
    game_state: KOTron,
    depth,
    is_maximizing_player: bool,
    maximizing_player_move=None,
    maximizing_player_index=0,
    minimizing_player_index=1,
) -> float:

    if game_state.status != GameStatus.IN_PROGRESS:

        if game_state.status == GameStatus.TIE:
            return 0.0
        else:
            if GameStatus.index_of_winner(game_state.status) == maximizing_player_index:
                return 1.0
            else:
                return -1.0
            
    if depth == 0:
        return model.run_inference([game_state], maximizing_player_index)[0]

    if is_maximizing_player:
        max_eval = -float("inf")
        for direction in KOTron.get_possible_directions(
            game_state, maximizing_player_index
        ):
            eval = minimax(
                model,
                game_state,
                depth,
                is_maximizing_player=False,
                maximizing_player_move=direction,
            )
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for direction in KOTron.get_possible_directions(
            game_state, minimizing_player_index
        ):
            child_state = KOTron.next(
                game_state,
                direction_updates=[
                    DirectionUpdate(maximizing_player_move, maximizing_player_index),
                    DirectionUpdate(direction, minimizing_player_index),
                ],
            )
            eval = minimax(model, child_state, depth - 1, is_maximizing_player=True)
            min_eval = min(min_eval, eval)
        return min_eval




