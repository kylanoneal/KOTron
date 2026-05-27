import copy
import torch
import pickle
import shutil
import random
import datetime

from tqdm import tqdm
from pathlib import Path
from typing import Callable
from functools import partial
from dataclasses import dataclass
from itertools import product

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

import tron
from tron.game import (
    GameState,
    PovGameState,
    GameStatus,
    StatusInfo,
    Direction,
    next,
    get_possible_directions,
)

from tron.enums import PovGameResult

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


from tron.utils.sim_utils import get_start_position
from tron.utils.tensorboard_utils import (
    TacticalBenchmarkContext,
    ValueBenchmarkContext,
    MatchContext,
    TacticGroup,
    model_tensorboard_update,
    benchmark,
)


from tron.utils.export_utils import export_quantized_nnue


def main() -> None:

    out_path = r"C:\Users\kylan\code\KOTron\tron-python\models\3x3.npz"

    model = NnueTronModel(
        3,
        3,
        acc_dim=256,
        fc_layer_neuron_counts=[32, 8],
        clamp_val=16,
    )

    state_dict = torch.load(
        r"C:\Users\kylan\code\KOTron\tron-python\scripts\y2026\m05\3x3\runs\20260527-142854_nnue_3x3_grid_search\valloss0.04_L0.001_B16_ACCDIM256_CLAMP16.0_FCLAYERS(32, 8)_best_epoch181.pth"
    )

    model.load_state_dict(state_dict, strict=True)

    quantized_model = QuantizedNnueTronModel(model, scale=256)

    export_quantized_nnue(quantized_model, out_path)


if __name__ == "__main__":
    main()
