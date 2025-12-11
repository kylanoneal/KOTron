import zmq
import uuid
import json
import torch
import shutil
import random
import datetime
import argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from cachetools import LRUCache, cached

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction

# from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random

from tron.ai.minimax import basic_minimax, MinimaxContext

from tron.ai.tron_model import RandomTronModel, CnnTronModel, PovGameState, TronModel
from tron.ai import MCTS

from tron.ai.training import (
    train_loop,
    make_dataloader,
    get_sos_info,
)
from tron.ai.benchmarks import (
    BENCHMARKS_5X5,
    TIE_BENCHMARKS_5X5,
    WIN_LOSS_BENCHMARKS_5X5,
    run_benchmark,
    run_model_benchmark,
    match,
)

from tron.io.to_proto import to_proto, from_proto


@dataclass
class BenchmarkContext:
    dir_fn: callable
    description: str


@dataclass
class MatchContext:
    p1_bc: BenchmarkContext
    p2_bc: BenchmarkContext
    starting_positions: list[GameState]


def get_start_position(
    n_rows: int,
    n_cols: int,
    p_neutral: float,
    p_obstacles: float,
    obstacle_density_range: tuple,
) -> GameState:

    is_neutral_start = p_neutral > random.random()
    are_obstacles = p_obstacles > random.random()

    min_d, max_d = obstacle_density_range
    obstacle_density = random.uniform(min_d, max_d) if are_obstacles else 0.0

    return GameState.new_game(
        num_players=2,
        num_rows=n_rows,
        num_cols=n_cols,
        random_starts=True,
        neutral_starts=is_neutral_start,
        obstacle_density=obstacle_density,
    )


NUM_ROWS = NUM_COLS = 5

# SIM_GAME_DEPTH = 2
WIN_REWARD = 1.5

GAMES_PER_ITER = 256
CHECKPOINT_EVERY_N = 10

P_NEUTRAL_START = 0.75
P_OBSTACLES = 0.4
OBSTACLE_DENSITY_RANGE = (0.0, 0.3)


games = []
for i in range(100):

    curr_game = [
        get_start_position(
            NUM_ROWS, NUM_COLS, P_NEUTRAL_START, P_OBSTACLES, OBSTACLE_DENSITY_RANGE
        )
    ]

    games.append(curr_game)

# Serialize game data
serialized_data = to_proto(games)

# Save the serialized data to a file.
with open(
    r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\datasets\20250810_5x5_100_starts.bin", "wb"
) as f:
    f.write(serialized_data)