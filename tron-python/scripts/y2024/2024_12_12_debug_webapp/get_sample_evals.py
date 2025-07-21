import json
import random
import torch
import shutil
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import numpy as np


from torch.utils.tensorboard import SummaryWriter

from game.tron import Tron, GameStatus, Player, Direction, DirectionUpdate
from game.utility_gui import init_utility_gui, show_game_state
from ai.algos import (
    choose_direction_model_naive,
    choose_direction_random,
    choose_direction_minimax,
    choose_direction_minimax_dumb,
)
from ai.model_architectures import FastNet, EvaluationNetConv3OneStride
from ai.tron_model import StandardTronModel

from ai.training import train_loop, make_dataloader, get_weights_sum_of_squares


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    state_dict = torch.load(
        "C:/Users/kylan/Documents/code/repos/Tron/python/tasks/2024_12_09_eval/runs/20241211-171205_oldnet_self_train_continuation_v5/checkpoints/oldnet_self_train_continuation_v5_7.pth"
    )
    torch_model = EvaluationNetConv3OneStride(grid_dim=10)
    torch_model.load_state_dict(state_dict)
    torch_model = torch_model.to(device)

    tron_model_v2 = StandardTronModel(torch_model, device)

    game_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    ]

    np_grid = np.array(game_grid).astype(bool)

    players = (
        Player(row=4, col=0, direction=Direction.UP, can_move=True),
        Player(row=7, col=9, direction=Direction.UP, can_move=True),
    )

    game = Tron(players=players)

    game.grid = np_grid

    up_state = Tron.next(game, (DirectionUpdate(Direction.UP, 1),))
    down_state = Tron.next(game, (DirectionUpdate(Direction.DOWN, 1),))

    evals = tron_model_v2.run_inference([up_state, down_state], 1)

    pass
