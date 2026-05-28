import copy
import torch
import pickle
import shutil
import random
import datetime

import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Callable
from functools import partial
from dataclasses import dataclass
from itertools import product

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset


from tron.game import (
    GameState,
    PovGameState,
    from_2d_game_state,
)

from tron.game_2d import GameState2D, Player2D


from tron.ai.tron_model import TronModel, RandomTronModel
from tron.ai.nnue import NnueTronModel, QuantizedNnueTronModel

from tron.ai import MCTS
from tron.ai.MCTS import MctsContext

from tron.ai.training import (
    LabeledExample,
    TrainValSplit,
    train,
    validate,
    get_label_magnitude,
    make_dataset,
    make_k_folds,
    make_batches,
)

from tron.ai.algos import choose_direction_basic_minimax
from tron.ai.MINIMAX_THAT_BUILDS_ORACLE_INFO import OracleInfo, GameResult, SpecialCase
from tron.utils.data_utils import make_tactics_from_oracle

from tron.gui.utility_gui import show_game_state


from tron.utils.sim_utils import get_start_position
from tron.utils.tensorboard_utils import (
    TacticalBenchmarkContext,
    ValueBenchmarkContext,
    MatchContext,
    TacticGroup,
    model_tensorboard_update,
    benchmark,
)


def main():

    NUM_ROWS = NUM_COLS = 4
    ############################################
    # INIT MODELS AND CREATE TRAIN/VAL SPLIT
    ############################################

    model = NnueTronModel(
        NUM_ROWS,
        NUM_COLS,
        acc_dim=256,
        fc_layer_neuron_counts=[32, 16],
        clamp_val=16,
    )

    state_dict = torch.load(
        r"C:\Users\kylan\code\KOTron\tron-python\models\lazy_valloss0.011_L0.001_B16_ACCDIM256_CLAMP16.0_FCLAYERS(32, 16)_best_epoch28.pth",
        map_location="cpu",
    )

    model.load_state_dict(state_dict, strict=True)

    ############################################
    # SIM/TRAIN/BENCHMARK LOOP
    ############################################

    example = PovGameState(
        game_state=from_2d_game_state(
            GameState2D(
                grid=np.array(
                    [
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 0],
                    ],
                    dtype=bool,
                ),
                players=(Player2D(2, 0, True), Player2D(2, 2, True)),
            )
        ),
        hero_index=0,
        opponent_index=1,
    )

    for pov_game_state in [example, PovGameState(example.game_state, 1, 0)]:
        print(model.run_inference(pov_game_state))

        quant_nnue = QuantizedNnueTronModel(model, scale=256)

        print(quant_nnue.run_inference(pov_game_state))


if __name__ == "__main__":
    main()
