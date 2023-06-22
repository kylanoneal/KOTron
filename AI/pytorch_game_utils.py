import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import json
import MCTS
from copy import deepcopy
from typing import Callable, Optional

from AI.model_architectures import *
from game.KyTron import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_position_evaluation(decay_fn: Callable[[float], float], game_progress: float, won_lost_tied: int,
                            is_tie_neutral=False):
    if won_lost_tied == 0 and not is_tie_neutral:
        evaluation = -1 * decay_fn(game_progress)
    else:
        evaluation = won_lost_tied * decay_fn(game_progress)

    return torch.tensor(evaluation, dtype=torch.float32).to(device)


def get_relevant_info_from_game_state(game_state):
    return deepcopy(game_state.collision_table), game_state.get_heads()


def get_model_input_from_raw_info(grid, heads, player_num, head_val, model_type: Optional[type] = None,
                                  is_part_of_batch=False):
    processed_grid = np.where(np.array(grid) == 0, 0, 1)

    for p_num, (x, y) in enumerate(heads):
        processed_grid[x][y] = head_val if player_num == p_num else -head_val

    tensor_output = torch.tensor(processed_grid, dtype=torch.float32).unsqueeze(0).to(device)
    if not is_part_of_batch:
        tensor_output = tensor_output.unsqueeze(0)

    if model_type is EvaluationAttentionConvNet:
        tensor_heads = torch.tensor(heads[player_num]).to(device)
        if not is_part_of_batch:
            tensor_heads = tensor_heads.unsqueeze(0)
        return tensor_output, tensor_heads
    else:
        return tensor_output


def get_model(model_path):
    return torch.load(model_path).to(device)


def get_next_action(model, game, player_num, head_val, temperature):
    """Try all available actions and choose action based on softmax of evaluations"""
    model_type = type(model)
    evaluations = []
    available_actions = game.get_possible_directions(player_num)

    for action in available_actions:
        next_game_state = deepcopy(game)
        next_game_state.update_direction(player_num, action)
        next_game_state.move_racers()

        grid, heads = get_relevant_info_from_game_state(next_game_state)
        curr_eval = model(get_model_input_from_raw_info(grid, heads, player_num, head_val, model_type=model_type))
        evaluations.append(curr_eval.item())

    return choose_action(available_actions, evaluations, temperature)


def get_next_action_mcts(model, game, player_num, head_val, n_iterations, exploration_factor, temperature):
    available_actions, visits = MCTS.search(model, game, player_num, head_val,
                                            n_iterations, exploration_factor=exploration_factor)

    return choose_action(available_actions, visits, temperature)

def get_random_next_action(game, player_num):
    """Try all available actions and choose action based on softmax of evaluations"""
    available_actions = game.get_possible_directions(player_num)

    # if len(available_actions) > 0:
    return available_actions[np.random.randint(0, len(available_actions))] if len(
        available_actions) > 0 else Directions.up



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
                won_or_lost_or_tied = (1 if player_num == winner_player_num else -1) \
                    if winner_player_num != -1 else 0

                processed_game_data.append((game_grid, heads, player_num, game_progress, won_or_lost_or_tied))

    return processed_game_data


def process_json_data(filepath, model_type, decay_fn, tie_is_neutral, head_val):
    with open(filepath, 'r') as file:
        # Load the JSON data into a Python object
        game_sims = json.load(file)

    processed_data = []
    for binary_grid, heads, p_num, game_progress, won_lost_or_tied in game_sims:
        target = get_position_evaluation(decay_fn, game_progress, won_lost_or_tied, tie_is_neutral)

        model_input = get_model_input_from_raw_info(binary_grid, heads, p_num, head_val,
                                                    model_type=model_type,
                                                    is_part_of_batch=True)

        processed_data.append((model_input, target))

    return processed_data
