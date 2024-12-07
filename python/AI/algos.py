import random
import torch
import numpy as np
from abc import ABC, abstractmethod


from game.ko_tron import KOTron, DirectionUpdate
from AI.tron_model import TronModelAbstract
from AI.basic_min_max import minimax


def choose_direction_model_naive(
    model: TronModelAbstract, game: KOTron, player_index: int
) -> DirectionUpdate:

    assert game.players[player_index].can_move

    possible_directions = KOTron.get_possible_directions(game, player_index)

    if len(possible_directions) == 0:
        # Maybe return None instead?
        return DirectionUpdate(
            direction=game.players[player_index].direction,
            player_index=player_index,
        )
    else:
        game_states_to_eval = [
            KOTron.next(game, [DirectionUpdate(direction, player_index)])
            for direction in possible_directions
        ]

        evaluations = model.run_inference(game_states_to_eval, player_index)

        return DirectionUpdate(
            direction=possible_directions[np.argmax(evaluations)],
            player_index=player_index,
        )


def choose_direction_random(game: KOTron, player_index: int) -> DirectionUpdate:

    assert game.players[player_index].can_move

    possible_directions = KOTron.get_possible_directions(game, player_index)

    if len(possible_directions) == 0:
        # Maybe return None instead?
        return DirectionUpdate(
            direction=game.players[player_index].direction,
            player_index=player_index,
        )
    else:
        return DirectionUpdate(
            direction=random.choice(possible_directions), player_index=player_index
        )


def choose_direction_minimax(
    model: TronModelAbstract,
    game: KOTron,
    player_index: int = 0,
    opponent_index: int = 1,
    depth: int = 5,
) -> DirectionUpdate:
    
    assert game.players[player_index].can_move

    hero_possible_directions = KOTron.get_possible_directions(game, player_index)
    opponent_possible_directions = KOTron.get_possible_directions(game, player_index)

    if len(hero_possible_directions) == 0:
        # Hero is effed, just die
        return DirectionUpdate(
            direction=game.players[player_index].direction,
            player_index=player_index,
        )
    
    if len(opponent_possible_directions) == 0:
        # Hero wins, choose any possible move
        return DirectionUpdate(
            direction=hero_possible_directions[0],
            player_index=player_index,
        )

    opponent_best_evals = []

    for hero_direction in hero_possible_directions:

        opponent_best_eval = float("inf")

        for opponent_direction in opponent_possible_directions:

            dir_updates = [
                DirectionUpdate(hero_direction, player_index),
                DirectionUpdate(opponent_direction, opponent_index),
            ]

            new_state = KOTron.next(game, dir_updates)
            move_value = minimax(model, new_state, depth - 1, is_maximizing_player=True)

            if move_value < opponent_best_eval:
                opponent_best_eval = move_value

        opponent_best_evals.append(opponent_best_eval)

    return DirectionUpdate(
        hero_possible_directions[np.argmax(opponent_best_evals)], player_index
    )
