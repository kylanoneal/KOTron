import random
import torch
import numpy as np
from abc import ABC, abstractmethod


from game.tron import Tron, DirectionUpdate
from ai.tron_model import TronModelAbstract
from ai.minimax import minimax_alpha_beta_eval_all, basic_minimax, minimax_dumb


def choose_direction_model_naive(
    model: TronModelAbstract, game: Tron, player_index: int
) -> DirectionUpdate:

    assert game.players[player_index].can_move

    possible_directions = Tron.get_possible_directions(game, player_index)

    if len(possible_directions) == 0:
        # Maybe return None instead?
        return DirectionUpdate(
            direction=game.players[player_index].direction,
            player_index=player_index,
        )
    else:
        game_states_to_eval = [
            Tron.next(game, (DirectionUpdate(direction, player_index),))
            for direction in possible_directions
        ]

        evaluations = model.run_inference(game_states_to_eval, player_index)

        return DirectionUpdate(
            direction=possible_directions[np.argmax(evaluations)],
            player_index=player_index,
        )


def choose_direction_random(game: Tron, player_index: int) -> DirectionUpdate:

    assert game.players[player_index].can_move

    possible_directions = Tron.get_possible_directions(game, player_index)

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
    game: Tron,
    player_index: int,
    opponent_index: int,
    depth: int,
    do_alpha_beta: bool = True,
) -> DirectionUpdate:

    assert game.players[player_index].can_move

    hero_possible_directions = Tron.get_possible_directions(game, player_index)
    opponent_possible_directions = Tron.get_possible_directions(game, opponent_index)

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

            dir_updates = (
                DirectionUpdate(hero_direction, player_index),
                DirectionUpdate(opponent_direction, opponent_index),
            )

            new_state = Tron.next(game, dir_updates)

            if do_alpha_beta:
                move_value = minimax_alpha_beta_eval_all(
                    model,
                    new_state,
                    depth - 1,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    is_maximizing_player=True,
                    maximizing_player_index=player_index,
                    minimizing_player_index=opponent_index,
                )
            else:

                move_value = basic_minimax(
                    model,
                    new_state,
                    depth - 1,
                    is_maximizing_player=True,
                    maximizing_player_index=player_index,
                    minimizing_player_index=opponent_index,
                )

            if move_value < opponent_best_eval:
                opponent_best_eval = move_value

        opponent_best_evals.append(opponent_best_eval)

    return DirectionUpdate(
        hero_possible_directions[np.argmax(opponent_best_evals)], player_index
    )


def choose_direction_minimax_alpha_beta(
    model: TronModelAbstract,
    game: Tron,
    player_index: int,
    opponent_index: int,
    depth: int,
) -> DirectionUpdate:

    assert game.players[player_index].can_move

    hero_possible_directions = Tron.get_possible_directions(game, player_index)
    opponent_possible_directions = Tron.get_possible_directions(game, opponent_index)

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

            dir_updates = (
                DirectionUpdate(hero_direction, player_index),
                DirectionUpdate(opponent_direction, opponent_index),
            )

            new_state = Tron.lru_cache_next(game, dir_updates)

            move_value = minimax_alpha_beta_eval_all(
                model,
                new_state,
                depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                is_maximizing_player=True,
                maximizing_player_index=player_index,
                minimizing_player_index=opponent_index,
            )

            if move_value < opponent_best_eval:
                opponent_best_eval = move_value

        opponent_best_evals.append(opponent_best_eval)

    return DirectionUpdate(
        hero_possible_directions[np.argmax(opponent_best_evals)], player_index
    )


def choose_direction_minimax_dumb(
    game: Tron,
    player_index: int,
    opponent_index: int,
    depth: int,
) -> DirectionUpdate:

    assert game.players[player_index].can_move

    hero_possible_directions = Tron.get_possible_directions(game, player_index)
    opponent_possible_directions = Tron.get_possible_directions(game, opponent_index)

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

            dir_updates = (
                DirectionUpdate(hero_direction, player_index),
                DirectionUpdate(opponent_direction, opponent_index),
            )

            new_state = Tron.lru_cache_next(game, dir_updates)

            move_value = minimax_dumb(
                new_state,
                depth - 1,
                is_maximizing_player=True,
                maximizing_player_index=player_index,
                minimizing_player_index=opponent_index,
            )

            if move_value < opponent_best_eval:
                opponent_best_eval = move_value

        opponent_best_evals.append(opponent_best_eval)

    return DirectionUpdate(
        hero_possible_directions[np.argmax(opponent_best_evals)], player_index
    )
