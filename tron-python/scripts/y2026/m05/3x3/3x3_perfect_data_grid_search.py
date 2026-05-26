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

            print(f"Special case! {oi.special_case}")

            if oi.special_case == SpecialCase.ONE_TIE_ONE_WIN:
                h0_label = h0_label / 2
            elif oi.special_case == SpecialCase.DIFF_STEPS_TO_SAME_RESULT:
                pass
            elif oi.special_case == SpecialCase.OPPOSITE_RESULT:
                h0_label = 0.0

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

    val_size = int(len(h0_examples) * val_pct)

    train_h0 = h0_examples[val_size:]
    train_h1 = h1_examples[val_size:]

    val_h0 = h0_examples[:val_size]
    val_h1 = h1_examples[:val_size]

    train_ds = train_h0 + train_h1
    val_ds = val_h0 + val_h1

    train_inputs = model.get_model_input([ex.pov_game_state for ex in train_ds]).to(
        device
    )
    train_labels = torch.tensor([ex.label for ex in train_ds]).to(device)

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_inputs = model.get_model_input([ex.pov_game_state for ex in val_ds]).to(device)
    val_labels = torch.tensor([ex.label for ex in val_ds]).to(device)

    val_loader = DataLoader(
        TensorDataset(val_inputs, val_labels),
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader


def main():


    RUN_DESCRIPTION = "nnue_3x3_grid_search"

    current_script_path = Path(__file__).resolve()

    outer_run_dir = current_script_path.parent / "runs"
    outer_run_dir.mkdir(exist_ok=True)

    run_dir = (
        outer_run_dir
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{RUN_DESCRIPTION}"
    )
    run_dir.mkdir(exist_ok=True)

    LRS = [0.001]
    B_SIZES = [16, 32]
    ACC_DIMS = [256]
    FC_NEURONS = [
        (8, 16),
        (32, 8),
        (16, 16, 8),
        (32,),
        (8,),
    ]

    CLAMP_VALS = [1.0, 4.0, 16.0]

    param_combos = list(
        product(
            LRS,
            B_SIZES,
            ACC_DIMS,
            FC_NEURONS,
            CLAMP_VALS,
        )
    )

    for lr, batch_size, acc_dim, fc_neurons, clamp_val in tqdm(
        param_combos,
        desc="Training configs",
    ):
        grid_search(run_dir, lr, batch_size, acc_dim, clamp_val, fc_neurons)


def grid_search(run_dir: Path, _lr, _bs, _adim, _cval, _fcns):

    LR = _lr
    BATCH_SIZE = _bs
    ACC_DIM = _adim
    CLAMP_VAL = _cval
    FC_NEURONS = _fcns

    MIN_DELTA = 0.0001
    PATIENCE = 50

    NUM_ROWS = NUM_COLS = 3

    tron_dir = Path(tron.__file__).resolve().parent.parent
    DATA_PATH = tron_dir / r"scripts\y2026\m05\perfect_data_oracle\3x3.pkl"

    K_FOLDS = 4
    PRE_TRAIN_EPOCHS = 1_000_000  # 50

    CHECKPOINT_EVERY_N = 1_000_000
    MAKE_VISUALIZATIONS_EVERY_N = 1_000

    VAL_PCT = 0.20
    DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}")

    RUN_UID = (
        f"L{LR}_B{BATCH_SIZE}_ACCDIM{ACC_DIM}_CLAMP{CLAMP_VAL}_FCLAYERS{FC_NEURONS}"
    )

    print(f"{RUN_UID=}")

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    uid_run_dir = run_dir / RUN_UID
    uid_run_dir.mkdir(exist_ok=False)

    tb_writer = SummaryWriter(log_dir=uid_run_dir)

    ############################################
    # INIT MODELS AND CREATE TRAIN/VAL SPLIT
    ############################################

    model = NnueTronModel(
        NUM_ROWS,
        NUM_COLS,
        acc_dim=ACC_DIM,
        fc_layer_neuron_counts=FC_NEURONS,
        clamp_val=CLAMP_VAL,
    ).to(DEVICE)


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=LR)

    with DATA_PATH.open("rb") as f:
        oracle_table = pickle.load(f)

    train_loader, val_loader = from_oracle_to_tensor_train_val_split(
        model, oracle_table, VAL_PCT, train_batch_size=BATCH_SIZE, device=DEVICE, seed=0
    )

    ############################################
    # BENCHMARKING SETUP
    ############################################

    # TODO: No need for matches / benchmark when you have the game solved

    # Best would be to have the optimal move(s) for every left-out game state,
    # test the model's ability to find that move at depth 1,2,3,...
    # tactics = make_tactics_from_oracle(oracle_table, n_tactics=10)
    # random_tactics = TacticGroup(tactics, "Randomly made.")

    # model_tactical_contexts = [
    #     model_d1_tbc := TacticalBenchmarkContext(
    #         partial(choose_direction_basic_minimax, model=model, depth=1),
    #         description="D1 Minimax",
    #     ),
    #     model_d3_tbc := TacticalBenchmarkContext(
    #         partial(choose_direction_basic_minimax, model=model, depth=3),
    #         description="D3 Minimax",
    #     ),
    # ]

    ############################################
    # TRAIN
    ############################################

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    epochs_without_improvement = 0

    for i in range(PRE_TRAIN_EPOCHS):

        training_result = train(
            model,
            train_loader,
            optimizer,
            criterion,
            epochs=1,
        )

        avg_validation_loss = validate(
            model,
            val_loader,
            criterion,
        )

        model_tensorboard_update(
            i,
            tb_writer,
            model,
            model_desc=f"",
            training_result=training_result,
            validation_loss=avg_validation_loss,
            make_visualizations=False,
        )

        # benchmark(
        #     i,
        #     tb_writer,
        #     make_visualizations=i % MAKE_VISUALIZATIONS_EVERY_N == 0,
        #     value_contexts=None,
        #     tactical_contexts=model_tactical_contexts,
        #     match_contexts=None,
        #     tactic_groups=[random_tactics],
        # )

        improved = avg_validation_loss < best_val_loss - MIN_DELTA

        if improved:
            best_val_loss = avg_validation_loss
            best_epoch = i

            best_state_dict = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

            print(f"  New best val loss: {best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1

            print(
                f"  No improvement for "
                f"{epochs_without_improvement}/{PATIENCE} epochs"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {i + 1}. "
                f"Best epoch was {best_epoch + 1} "
                f"with val_loss={best_val_loss:.6f}."
            )
            break

    # -----------------------
    # Save only the best network
    # -----------------------
    if best_state_dict is None:
        raise RuntimeError("No best model was recorded.")

    torch.save(
        best_state_dict,
        run_dir / f"valloss{round(best_val_loss, 3)}_{RUN_UID}_best_epoch{i}.pth",
    )


if __name__ == "__main__":
    main()
