import numpy as np
import torch
import json

from enum import Enum
from copy import deepcopy
from typing import Callable, Optional

# from AI.MCTS import search

# from AI import model_architectures

from game.ko_tron import KOTron, Directions

device = "cuda" if torch.cuda.is_available() else "cpu"

class GameResult(Enum):
    LOSS = -1
    WIN = 1
    TIE = 0


def get_position_evaluation(
    game_progress: float,
    game_result: GameResult,
    decay_fn: Optional[Callable] = lambda x: x,
    is_tie_neutral: bool = False,
):
    """
    Get tensor evaluation based on game result and game progress.

    Args:
        game_progress (float): Percent representing how far game has progressed (turn_num/total_turns)
        game_result (GameResult): Win, loss, or tie. Losses will have negative evals and vice versa.
        decay_fn (Optional[Callable]): Optionally specify a decay function to give.
            lower magnitude evals to turns earlier in the game. Assumes the end of the
            game is more important than the beginning.
        is_tie_neutral (bool): Whether or not a tie should be considered neutral and all evals set 
            to zero. If not, ties are treated as losses.

    Returns:
        torch.Tensor: Float32 evaluation from [-1, 1]
    """

    assert type(game_result) == GameResult

    if not (0.0 <= game_progress <= 1.0):
        raise ValueError("Game progress must be [0, 1]")

    if game_result == GameResult.TIE and not is_tie_neutral:
        evaluation = -1 * decay_fn(game_progress)
    else:
        evaluation = game_result.value * decay_fn(game_progress)

    assert -1.0 <= evaluation <= 1.0, "Evaluation must be [-1, 1]"

    return torch.tensor(evaluation, dtype=torch.float32).to(device)


def get_relevant_info_from_game_state(game_state):
    return deepcopy(game_state.grid), game_state.get_heads()


def get_model_input_from_raw_info(
    grid,
    heads,
    player_num,
    model_type: Optional[type] = None,
    head_val: Optional[int]=None,
    is_part_of_batch=False,
):
    np_grid = np.array(grid)

    # TODO: JaNK
    if model_type is None or "EvaluationNetConv3" in model_type.__name__:
        # if model_type == EvaluationNetConv3:
        processed_grid = np.zeros((3, np_grid.shape[0], np_grid.shape[1]))
        processed_grid[0, :, :] = np.where(np_grid == 0, 0, 1)
        for p_num, (x, y) in enumerate(heads):
            if p_num == player_num:
                processed_grid[1][x][y] = 1
            else:
                # Enemy player represented as -1 in the third channel
                processed_grid[2][x][y] = -1
    else:
        raise NotImplementedError()
        # processed_grid = np.where(np_grid == 0, 0, 1)
        # for p_num, (x, y) in enumerate(heads):
        #     processed_grid[x][y] = head_val if player_num == p_num else -head_val

    tensor_output = torch.tensor(processed_grid, dtype=torch.float32).to(device)

    if not is_part_of_batch:
        tensor_output = tensor_output.unsqueeze(0)

    return tensor_output


def get_model(model_path):
    return torch.load(model_path).to(device)


def get_next_action(
    model, game, player_num, temperature, head_val: Optional[int] = None
):
    """Try all available actions and choose action based on softmax of evaluations"""
    model_type = type(model)
    evaluations = []
    available_actions = game.get_possible_directions(player_num)

    assert not model.training, "Model in training mode"

    for action in available_actions:
        next_game_state = deepcopy(game)
        next_game_state.update_direction(player_num, action)
        next_game_state.move_racers()

        grid, heads = get_relevant_info_from_game_state(next_game_state)

        model_input = get_model_input_from_raw_info(
            grid, heads, player_num, model_type=model_type, head_val=head_val
        )

        curr_eval = model(model_input)
        evaluations.append(curr_eval.item())

    return choose_action(available_actions, evaluations, temperature)


def get_next_action_mcts(
    model, game, player_num, head_val, n_iterations, exploration_factor, temperature
):
    raise NotImplementedError()
    # available_actions, visits = search(
    #     model,
    #     game,
    #     player_num,
    #     head_val,
    #     n_iterations,
    #     exploration_factor=exploration_factor,
    # )

    # return choose_action(available_actions, visits, temperature)


def get_random_next_action(game, player_num):
    """Try all available actions and choose action based on softmax of evaluations"""
    available_actions = game.get_possible_directions(player_num)

    # if len(available_actions) > 0:
    return (
        available_actions[np.random.randint(0, len(available_actions))]
        if len(available_actions) > 0
        else Directions.up
    )


def choose_action(available_actions, evals, temperature):
    # No available actions, just choose up
    if len(available_actions) == 0:
        return Directions.up
    # Choose the only available action
    elif len(available_actions) == 1:
        return available_actions[0]
    # Choose action with best evaluation
    elif temperature == 0.0:
        return available_actions[np.argmax(evals)]
    # Choose action based on softmax of evaluations
    else:
        exp_values = np.exp(np.array(evals) / temperature)
        action_probs = exp_values / np.sum(exp_values)
        return np.random.choice(available_actions, size=1, p=action_probs).item()


def process_game_data(game_data, winner_player_num, game_length_threshold=0):
    processed_game_data = []
    if len(game_data) > game_length_threshold:
        for turn_num, (game_grid, heads) in enumerate(game_data):
            game_progress = turn_num / (len(game_data) - 1)

            for player_num, head in enumerate(heads):
                won_or_lost_or_tied = (
                    (1 if player_num == winner_player_num else -1)
                    if winner_player_num != -1
                    else 0
                )

                processed_game_data.append(
                    (game_grid, heads, player_num, game_progress, won_or_lost_or_tied)
                )

    return processed_game_data


def process_json_data(filepath, model_type, decay_fn, tie_is_neutral, head_val=None):
    with open(filepath, "r") as file:
        # Load the JSON data into a Python object
        game_sims = json.load(file)

    processed_data = []
    for binary_grid, heads, p_num, game_progress, game_result in game_sims:
        target = get_position_evaluation(
            game_progress, GameResult(game_result), decay_fn, tie_is_neutral
        )

        model_input = get_model_input_from_raw_info(
            binary_grid,
            heads,
            p_num,
            model_type=model_type,
            head_val=head_val,
            is_part_of_batch=True,
        )

        processed_data.append((model_input, target))

    return processed_data

def get_dataloader(game_collection: GameCollection):

    dataset = []

    for game_container in game_collection.game_containers:
        for position in game_container.positions:

            model_input = get_model_input_from_raw_info(
                position, player_num, 
            )

            label = get_position_evaluation()