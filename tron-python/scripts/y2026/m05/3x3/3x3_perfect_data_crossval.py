import torch
import pickle
import shutil
import random
import datetime
import warnings
import argparse

from tqdm import tqdm
from pathlib import Path
from typing import Callable
from functools import partial
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

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
from tron.ai.MINIMAX_THAT_BUILDS_ORACLE_INFO import OracleInfo, GameResult


from tron.utils.sim_utils import get_start_position
from tron.utils.tensorboard_utils import (
    TacticalBenchmarkContext,
    ValueBenchmarkContext,
    MatchContext,
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


def from_oracle_to_labeled_examples(
    oracle_table: dict[GameState, OracleInfo], k_folds: int, seed=0
):

    h0_examples: list[LabeledExample] = []

    for gs, oi in oracle_table.items():

        if oi.result == GameResult.TIE:
            label_mag = 0.0
        else:
            label_mag = get_label_magnitude(oi.steps_to_result)

        h0_label = label_mag if oi.result == GameResult.P1_WIN else -label_mag

        h0_examples.append(LabeledExample(PovGameState(gs, 0, 1), h0_label))

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


    h0_kfolds = make_k_folds(h0_examples, k_folds, shuffle=False)
    h1_kfolds = make_k_folds(h1_examples, k_folds, shuffle=False)

    # Build combined kfolds:

    combined_kfolds = []

    for h0_kfold, h1_kfold in zip(h0_kfolds, h1_kfolds):

        combined_kfolds.append(
            TrainValSplit(
                train_examples=h0_kfold.train_examples + h1_kfold.train_examples,
                val_examples=h0_kfold.val_examples + h1_kfold.val_examples,
            )
        )

    return combined_kfolds


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--acc_dim", type=int)

    args = parser.parse_args()

    LR = args.lr
    BATCH_SIZE = args.batch_size
    ACC_DIM = args.acc_dim

    RUN_DESCRIPTION = "nnue_3x3_perfect_data_crossval"
    NUM_ROWS = NUM_COLS = 3

    tron_dir = Path(tron.__file__).resolve().parent.parent
    DATA_PATH = tron_dir / r"scripts\y2026\m05\perfect_data_oracle\3x3.pkl"

    K_FOLDS = 4
    PRE_TRAIN_EPOCHS = 1_000_000  # 50

    MAKE_VISUALIZATIONS_EVERY_N = 200

    RUN_UID = f"L{LR}_B{BATCH_SIZE}_ACCDIM{ACC_DIM}"

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    current_script_path = Path(__file__).resolve()

    outer_run_dir = current_script_path.parent / "runs"
    outer_run_dir.mkdir(exist_ok=True)

    run_dir = (
        outer_run_dir
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{RUN_DESCRIPTION}"
    )
    run_dir.mkdir(exist_ok=True)

    uid_run_dir = run_dir / RUN_UID
    uid_run_dir.mkdir(exist_ok=False)

    data_out_dir = uid_run_dir / "game_data"
    data_out_dir.mkdir()

    checkpoints_dir = uid_run_dir / "checkpoints"
    checkpoints_dir.mkdir()

    backup_path = uid_run_dir / f"{current_script_path.name.split('.')[0]}.bak.py"
    shutil.copy2(current_script_path, backup_path)

    tb_writer = SummaryWriter(log_dir=uid_run_dir)

    ############################################
    # BENCHMARKING SETUP
    ############################################

    # TODO: No need for matches / benchmark when you have the game solved

    # Best would be to have the optimal move(s) for every left-out game state,
    # test the model's ability to find that move at depth 1,2,3,...

    ############################################
    # CREATE CROSSVAL SPLITS AND INIT MODELS
    ############################################

    with DATA_PATH.open("rb") as f:
        oracle_table = pickle.load(f)

    cross_val_splits = from_oracle_to_labeled_examples(oracle_table, K_FOLDS, seed=0)

    cross_val_contexts: list[CrossValContext] = []

    for i, cv_split in enumerate(cross_val_splits):

        _model = NnueTronModel(NUM_ROWS, NUM_COLS, acc_dim=ACC_DIM)

        _criterion = torch.nn.MSELoss()
        _optimizer = torch.optim.Adam(_model.parameters(), amsgrad=True, lr=LR)

        cross_val_contexts.append(
            CrossValContext(
                model=_model,
                optim=_optimizer,
                criterion=_criterion,
                train_val_split=cv_split,
            )
        )

    ############################################
    # TRAIN
    ############################################

    for i in range(PRE_TRAIN_EPOCHS):

        cum_epoch_train_loss = cum_epoch_val_loss = 0.0

        for j, cv_context in enumerate(cross_val_contexts):

            # # Save the serialized data to a file.
            # with open(data_file, "rb") as f:
            #     bin_data = f.read()

            # game_data = from_proto(bin_data)

            train_ds = make_batches(
                cv_context.train_val_split.train_examples,
                batch_size=BATCH_SIZE,
                shuffle=True,
                seed=i,
            )

            training_result = train(
                cv_context.model,
                train_ds,
                cv_context.optim,
                cv_context.criterion,
                epochs=1,
            )

            avg_validation_loss = validate(
                cv_context.model,
                cv_context.train_val_split.val_examples,
                cv_context.criterion,
            )

            model_tensorboard_update(
                i,
                tb_writer,
                cv_context.model,
                model_desc=f"cvsplit{j}",
                training_result=training_result,
                validation_loss=avg_validation_loss,
                make_visualizations=i % MAKE_VISUALIZATIONS_EVERY_N == 0,
            )

            torch.save(
                cv_context.model.state_dict(),
                checkpoints_dir / f"{RUN_UID}_epoch{i}_cvsplit{j}.pth",
            )

            cum_epoch_train_loss += training_result.avg_loss
            cum_epoch_val_loss += avg_validation_loss

        tb_writer.add_scalar(f"aggregate_train_loss", cum_epoch_train_loss / K_FOLDS, i)
        tb_writer.add_scalar(f"aggregate_val_loss", cum_epoch_val_loss / K_FOLDS, i)


if __name__ == "__main__":
    main()
