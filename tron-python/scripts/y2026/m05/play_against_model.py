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


@dataclass(frozen=True)
class CrossValContext:
    model: torch.nn.Module
    optim: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_val_split: TrainValSplit


@dataclass(frozen=True)
class OracleExample:
    pov_game_state: PovGameState
    oracle_info: OracleInfo


def from_oracle_to_tensor_train_val_split(
    model: TronModel,
    oracle_table: dict[GameState, OracleInfo],
    val_pct: float = 0.2,
    train_batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    seed=0,
):

    h0_examples: list[LabeledExample] = []

    for gs, oi in oracle_table.items():

        if oi.result == GameResult.TIE:
            label_mag = 0.0
        else:
            label_mag = get_label_magnitude(oi.steps_to_result)

        assert (
            oi.hero_player == gs.players[0]
        ), "This can change but asserting this for now"
        assert oi.oppo_player == gs.players[1], "For good measure"

        h0_label = label_mag if oi.result == GameResult.HERO_WIN else -label_mag

        if oi.special_case is not None:

            print(f"Special case skipping! {oi.special_case}")

            if oi.special_case == SpecialCase.ONE_TIE_ONE_WIN:
                h0_label = h0_label / 2
            elif oi.special_case == SpecialCase.DIFF_STEPS_TO_SAME_RESULT:
                pass
            elif oi.special_case == SpecialCase.OPPOSITE_RESULT:
                h0_label = 0.0
        else:
            h0_examples.append(LabeledExample(PovGameState(gs, 0, 1), h0_label))

    print("H0 made")
    rng = random.Random(seed)

    rng.shuffle(h0_examples)

    # Build mirrored examples:

    h1_examples = []
    for h0_ex in h0_examples:

        h1_examples.append(
            LabeledExample(
                PovGameState(h0_ex.pov_game_state.game_state, 1, 0), -h0_ex.label
            )
        )

    print("H1 made")
    val_size = int(len(h0_examples) * val_pct)

    train_h0 = h0_examples[val_size:]
    train_h1 = h1_examples[val_size:]

    val_h0 = h0_examples[:val_size]
    val_h1 = h1_examples[:val_size]

    train_ds = train_h0 + train_h1
    val_ds = val_h0 + val_h1

    print("Splits made")
    train_inputs = model.get_model_input([ex.pov_game_state for ex in train_ds]).to(
        device
    )

    print("train inputs made")
    train_labels = torch.tensor([ex.label for ex in train_ds]).to(device)
    print("train labels made")
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_inputs = model.get_model_input([ex.pov_game_state for ex in val_ds]).to(device)
    print("val inputs made")
    val_labels = torch.tensor([ex.label for ex in val_ds]).to(device)
    print("val labels made")

    val_loader = DataLoader(
        TensorDataset(val_inputs, val_labels),
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader


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
        r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2026\m05\4x4\runs\20260527-123227_nnue_4x4_grid_search\lazy_valloss0.011_L0.001_B16_ACCDIM256_CLAMP16.0_FCLAYERS(32, 16)_best_epoch28.pth"
    )

    model.load_state_dict(state_dict, strict=True)

    ############################################
    # SIM/TRAIN/BENCHMARK LOOP
    ############################################


    games: list[list[GameState]] = []

    p1_wins = p2_wins = ties = 0

    for _ in tqdm(range(100)):

        game_state: GameState = get_start_position(
            NUM_ROWS, NUM_COLS, 0.5, 0.5, (0.0, 0.5)
        )

        game_status: StatusInfo = tron.get_status(game_state)

        current_game: list[GameState] = [game_state]

        while game_status.status == GameStatus.IN_PROGRESS:

            p1_dir = choose_direction_basic_minimax(
                PovGameState(game_state, 0, 1), model, 1
            )

            p2_dir = show_game_state(game_state, step_through=True)

            game_state = next(game_state, directions=(p1_dir, p2_dir))

            current_game.append(game_state)

            game_status = tron.get_status(game_state)

        games.append(current_game)

        if game_status.status == GameStatus.TIE:
            print(f"Tie")
            ties += 1
        elif game_status.winner_index == 0:
            p1_wins += 1
            print("P1 Win")
        elif game_status.winner_index == 1:
            p2_wins += 1
            print("P2 win")


if __name__ == "__main__":
    main()
