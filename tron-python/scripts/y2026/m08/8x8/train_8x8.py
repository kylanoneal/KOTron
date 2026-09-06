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
    percent_board_reachable,
)

from tron.enums import PovGameResult

from tron.ai.tron_model import TronModel, RandomTronModel
from tron.ai.nnue import NnueTronModel, QuantizedNnueTronModel

import tron.ai.MCTS_pessimistic as MCTS


from tron.ai.training import (
    LabeledExample,
    TrainValSplit,
    train,
    validate,
    get_label_magnitude,
)

from tron.ai.algos import choose_direction_basic_minimax
from tron.ai.minimax_oracle_pessimistic import OracleGameState, GameResult

from tron.utils.sim_utils import get_start_position
from tron.utils.tensorboard_utils import (
    TacticalBenchmarkContext,
    ValueBenchmarkContext,
    MatchContext,
    TacticGroup,
    model_tensorboard_update,
    benchmark,
)

from tron.utils.data_utils import make_oracle_data, swap_two_player_game

from tron.io.json import read_tactics_json
from tron.io.proto import labeled_game_states_from_proto


@dataclass(frozen=True)
class CrossValContext:
    model: torch.nn.Module
    optim: torch.optim.Optimizer
    criterion: torch.nn.Module
    train_val_split: TrainValSplit


from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler


def combine_dataloaders(
    loader_a: DataLoader,
    loader_b: DataLoader,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    num_samples: int | None = None,
) -> DataLoader:
    """Combine two DataLoaders with weighted random sampling.

    Args:
        loader_a: First DataLoader.
        loader_b: Second DataLoader.
        weight_a: Probability weight assigned to dataset A.
        weight_b: Probability weight assigned to dataset B.
        num_samples: Number of samples per epoch. Defaults to the combined
            dataset size.

    Returns:
        A DataLoader sampling from both datasets according to the given weights.
    """
    dataset_a = loader_a.dataset
    dataset_b = loader_b.dataset

    n_a = len(dataset_a)
    n_b = len(dataset_b)

    combined_dataset = ConcatDataset([dataset_a, dataset_b])

    sample_weights = torch.cat(
        [
            torch.full((n_a,), weight_a / n_a),
            torch.full((n_b,), weight_b / n_b),
        ]
    )

    if num_samples is None:
        num_samples = n_a + n_b

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )

    return DataLoader(
        combined_dataset,
        batch_size=loader_a.batch_size,
        sampler=sampler,
    )


# NOTE: Choosing not to include examples from both perspectives for now
def dataloader_from_oracle(
    model: TronModel,
    oracle_examples: list[OracleGameState],
    batch_size: int,
    keep_rate: float = 1.0,
    perspective_swaps: bool = True,
    augment: bool = True,
    shuffle: bool = True,
    device: torch.device = torch.device("cpu"),
    seed=0,
):

    # Make shallow copy first
    oracle_examples = copy.copy(oracle_examples)

    rng = random.Random(seed)

    if shuffle:

        rng.shuffle(oracle_examples)

    if keep_rate < 1.0:

        keep_step = len(oracle_examples) // max(
            1, int((len(oracle_examples) * keep_rate))
        )

        oracle_examples = oracle_examples[::keep_step]

    if augment:

        for i in range(len(oracle_examples)):

            augmented = GameState.transform(
                oracle_examples[i].game,
                do_lr_flip=random.random() > 0.5,
                n_rot_90=random.randrange(0, 4),
            )

            oracle_examples[i] = OracleGameState(
                augmented,
                result=oracle_examples[i].result,
                steps_to_result=oracle_examples[i].steps_to_result,
            )

    labeled_examples: list[LabeledExample] = []

    for oi in oracle_examples:

        if oi.result == GameResult.TIE:
            label_mag = 0.0
        else:
            label_mag = get_label_magnitude(oi.steps_to_result)

        h0_label = label_mag if oi.result == GameResult.P1_WIN else -label_mag

        ex = LabeledExample(PovGameState(oi.game, 0, 1), h0_label)

        if perspective_swaps and rng.random() > 0.5:
            ex = LabeledExample(PovGameState(oi.game, 1, 0), -h0_label)

        labeled_examples.append(ex)

    inputs = model.get_model_input([ex.pov_game_state for ex in labeled_examples]).to(
        device
    )
    print("val inputs made")
    labels = torch.tensor([ex.label for ex in labeled_examples]).to(device)
    print("val labels made")

    data_loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    return data_loader


def start_oracle(game_state, trigger_ratio):

    p1, _ = percent_board_reachable(game_state, 0)
    p2, _ = percent_board_reachable(game_state, 1)

    imminent_loss = p1 < 0.15 or p2 < 0.15

    both_under_cutoff = p1 < trigger_ratio and p2 < trigger_ratio

    return imminent_loss or both_under_cutoff


def main():

    RUN_DESCRIPTION = "nnue_8x8_improved_metrics"

    current_script_path = Path(__file__).resolve()

    outer_run_dir = current_script_path.parent / "runs"
    outer_run_dir.mkdir(exist_ok=True)

    run_dir = (
        outer_run_dir
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{RUN_DESCRIPTION}"
    )
    run_dir.mkdir(exist_ok=True)

    LRS = [0.001]
    B_SIZES = [16]
    ACC_DIMS = [512]
    FC_NEURONS = [(32, 16)]

    CLAMP_VALS = [16]

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

    NUM_ROWS = NUM_COLS = 8

    tron_dir = Path(tron.__file__).resolve().parent.parent
    VAL_DATA_PATH = (
        tron_dir
        / r"scripts\y2026\m08\8x8\8x8_validation_data_v1\8x8_validation_1000games.bin"
    )
    VAL_TACTICS_PATH = (
        tron_dir / r"scripts\y2026\m08\8x8\8x8_validation_data_v1\tactics.json"
    )

    K_FOLDS = 4
    PRE_TRAIN_EPOCHS = 0  # 50

    TRAIN_ITERS = 100_000
    GAMES_PER_ITER = 512
    CHECKPOINT_EVERY_N = 5
    PLAY_MATCH_EVERY_N = 20
    MAKE_VISUALIZATIONS_EVERY_N = 1_000

    VAL_PCT = 0.20
    DEVICE = torch.device("cpu")

    MCTS_ITERS = 16
    TEMP = 0.4  # 0.7
    EXPLR_FACTOR = 2.0

    KEEP_RATE = 0.5
    WIN_REWARD = 1.5
    QUANT_SCALE = 32

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.2  # 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    ORACLE_TRIGGER_REACHABLE_RATIO = 0.2

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

    checkpoints_dir = uid_run_dir / "checkpoints"
    checkpoints_dir.mkdir()

    data_out_dir = uid_run_dir / "game_data"
    data_out_dir.mkdir()

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

    with open(VAL_DATA_PATH, "rb") as f:

        bin_validation_oracle_examples = f.read()

    validation_oracle_examples = labeled_game_states_from_proto(
        bin_validation_oracle_examples
    )

    val_loader = dataloader_from_oracle(
        model, validation_oracle_examples, batch_size=64, keep_rate=0.01
    )

    ############################################
    # BENCHMARKING SETUP
    ############################################

    # TODO: No need for matches / benchmark when you have the game solved

    tactics_json = read_tactics_json(VAL_TACTICS_PATH)[::10]
    val_tactics = TacticGroup(tactics_json, "Validation tactics")

    model_tactical_contexts = [
        model_d1_tbc := TacticalBenchmarkContext(
            partial(choose_direction_basic_minimax, model=model, depth=1),
            description="D1 Minimax",
        ),
        model_d3_tbc := TacticalBenchmarkContext(
            partial(choose_direction_basic_minimax, model=model, depth=3),
            description="D3 Minimax",
        ),
    ]

    ############################################
    # PRE-TRAIN
    ############################################

    # best_val_loss = float("inf")
    # best_epoch = -1
    # best_state_dict = None

    # epochs_without_improvement = 0

    # print("Starting training...")
    # for i in range(PRE_TRAIN_EPOCHS):

    #     training_result = train(
    #         model,
    #         train_loader,
    #         optimizer,
    #         criterion,
    #         epochs=1,
    #     )

    #     avg_validation_loss = validate(
    #         model,
    #         val_loader,
    #         criterion,
    #     )

    #     model_tensorboard_update(
    #         i,
    #         tb_writer,
    #         model,
    #         model_desc=f"",
    #         training_result=training_result,
    #         validation_loss=avg_validation_loss,
    #         make_visualizations=False,
    #     )

    #     # benchmark(
    #     #     i,
    #     #     tb_writer,
    #     #     make_visualizations=i % MAKE_VISUALIZATIONS_EVERY_N == 0,
    #     #     value_contexts=None,
    #     #     tactical_contexts=model_tactical_contexts,
    #     #     match_contexts=None,
    #     #     tactic_groups=[random_tactics],
    #     # )

    #     improved = avg_validation_loss < best_val_loss - MIN_DELTA

    #     if improved:
    #         best_val_loss = avg_validation_loss
    #         best_epoch = i

    #         best_state_dict = copy.deepcopy(model.state_dict())

    #         torch.save(
    #             best_state_dict,
    #             run_dir / f"lazy_valloss{round(best_val_loss, 3)}_{RUN_UID}_best_epoch{i}.pth",
    #         )

    #         epochs_without_improvement = 0

    #         print(f"  New best val loss: {best_val_loss:.6f}")
    #     else:
    #         epochs_without_improvement += 1

    #         print(
    #             f"  No improvement for "
    #             f"{epochs_without_improvement}/{PATIENCE} epochs"
    #         )

    #     if epochs_without_improvement >= PATIENCE:
    #         print(
    #             f"Early stopping at epoch {i + 1}. "
    #             f"Best epoch was {best_epoch + 1} "
    #             f"with val_loss={best_val_loss:.6f}."
    #         )
    #         break

    # # -----------------------
    # # Save only the best network
    # # -----------------------
    # if best_state_dict is None:
    #     raise RuntimeError("No best model was recorded.")

    # torch.save(
    #     best_state_dict,
    #     run_dir / f"valloss{round(best_val_loss, 3)}_{RUN_UID}_best_epoch{i}.pth",
    # )

    ############################################
    # SIM/TRAIN/BENCHMARK LOOP
    ############################################

    total_p1_wins = total_p2_wins = total_ties = 0

    for i in range(PRE_TRAIN_EPOCHS, TRAIN_ITERS + PRE_TRAIN_EPOCHS):

        quant_model = QuantizedNnueTronModel(model, scale=QUANT_SCALE)

        p1_mcts_context = MCTS.MctsContext(0, 1, WIN_REWARD, quant_model, use_acc=True)
        p2_mcts_context = MCTS.MctsContext(1, 0, WIN_REWARD, quant_model, use_acc=True)

        games_prior_to_oracle: list[list[OracleGameState]] = []
        oracle_games: list[dict[GameState, OracleGameState]] = []

        p1_wins = p2_wins = ties = 0

        for _ in tqdm(range(GAMES_PER_ITER)):

            game_state: GameState = get_start_position(
                NUM_ROWS,
                NUM_COLS,
                P_NEUTRAL_START,
                P_OBSTACLES,
                OBSTACLE_DENSITY_RANGE,
            )

            game_status: StatusInfo = tron.get_status(game_state)

            current_game: list[GameState] = [game_state]

            p1_initial_acc = quant_model.initialize_acc(PovGameState(game_state, 0, 1))
            p2_initial_acc = quant_model.initialize_acc(PovGameState(game_state, 1, 0))

            next_p1_root = next_p2_root = None

            game_steps = 0

            while (
                not start_oracle(game_state, ORACLE_TRIGGER_REACHABLE_RATIO)
                and game_status.status == GameStatus.IN_PROGRESS
            ):

                p1_dir, p1_root = MCTS.search(
                    p1_mcts_context,
                    game_state,
                    n_iterations=MCTS_ITERS,
                    temp=TEMP,
                    exploration_factor=EXPLR_FACTOR,
                    root=next_p1_root,
                    initial_acc=p1_initial_acc if next_p1_root is None else None,
                )

                p2_dir, p2_root = MCTS.search(
                    p2_mcts_context,
                    game_state,
                    n_iterations=MCTS_ITERS,
                    temp=TEMP,
                    exploration_factor=EXPLR_FACTOR,
                    root=next_p2_root,
                    initial_acc=p2_initial_acc if next_p2_root is None else None,
                )

                next_p1_root = MCTS.get_next_root(p1_root, p1_dir, p2_dir)
                next_p2_root = MCTS.get_next_root(p2_root, p2_dir, p1_dir)

                # show_game_state(game_state, step_through=True)

                # p1_dir = choose_direction_random(game_state, 0)
                # p2_dir = choose_direction_random(game_state, 1)

                game_state = next(game_state, directions=(p1_dir, p2_dir))

                current_game.append(game_state)

                game_status = tron.get_status(game_state)

            oracle_table, _special_cases = make_oracle_data(game_state)

            # Skip special cases
            if not game_state in oracle_table:
                continue

            oracle_info = oracle_table[game_state]
            game_prior_to_oracle = []

            for j, gs in enumerate(current_game[:-1]):

                new_steps_to_result = oracle_info.steps_to_result + (
                    len(current_game) - 1 - j
                )

                game_prior_to_oracle.append(
                    OracleGameState(gs, oracle_info.result, new_steps_to_result)
                )

            assert sum([oi.game in oracle_table for oi in game_prior_to_oracle]) == 0

            if len(game_prior_to_oracle) > 0:
                assert (
                    game_prior_to_oracle[-1].steps_to_result
                    == oracle_info.steps_to_result + 1
                )

            games_prior_to_oracle.append(game_prior_to_oracle)

            print(f"{len(oracle_table)=}")
            oracle_games.append(list(oracle_table.values()))

            print(f"{oracle_info.result=}")
            if oracle_info.result == GameResult.TIE:
                # print(f"Tie")
                ties += 1
            elif oracle_info.result == GameResult.P1_WIN:
                p1_wins += 1
                # print("P1 Win")
            elif oracle_info.result == GameResult.P2_WIN:
                p2_wins += 1
                # print("P2 win")

        # serialized_data = tron.to_proto(games)

        # Save the serialized data to a file.
        # with open(
        #     data_out_dir / f"gamedata_{i}_ngames_{GAMES_PER_ITER}.bin", "wb"
        # ) as f:
        #     f.write(serialized_data)

        print("\n" + "-" * 25 + "\n")
        print(f"{i=}, {p1_wins=}, {p2_wins=}, {ties=}")

        total_ties += ties
        total_p1_wins += p1_wins
        total_p2_wins += p2_wins

        print(f"{total_p1_wins=}, {total_p2_wins=}, {total_ties=}")

        tb_writer.add_scalar("P1 Wins / Total Wins", p1_wins / (p1_wins + p2_wins), i)
        tb_writer.add_scalar("Tie Rate", ties / GAMES_PER_ITER, i)
        # tb_writer.add_scalar(
        #     "Average Game Length",
        #     sum([len(game) for game in games]) / GAMES_PER_ITER,
        #     i,
        # )

        quant_model_tactical_contexts = [
            quant_model_d1_tbc := TacticalBenchmarkContext(
                partial(choose_direction_basic_minimax, model=quant_model, depth=1),
                description="D1 Minimax (Quantized)",
            ),
            quant_model_d3_tbc := TacticalBenchmarkContext(
                partial(choose_direction_basic_minimax, model=quant_model, depth=3),
                description="D3 Minimax (Quantized)",
            ),
        ]

        match_contexts = [
            # MatchContext(model_d1_tbc, random_bot_d1_tbc, match_starting_positions),
            # MatchContext(model_d3_tbc, random_bot_d3_tbc, match_starting_positions),
            # MatchContext(
            #     quant_model_d1_tbc, random_bot_d1_tbc, match_starting_positions
            # ),
            # MatchContext(
            #     quant_model_d3_tbc, random_bot_d3_tbc, match_starting_positions
            # ),
            # MatchContext(model_d1_bc, prev_model_d1_bc, match_starting_positions),
            # MatchContext(model_d3_bc, prev_model_d3_bc, match_starting_positions),
        ]

        benchmark(
            i,
            tb_writer,
            make_visualizations=False,
            value_contexts=[
                ValueBenchmarkContext(model, "non-quant model"),
                ValueBenchmarkContext(quant_model, "quant model"),
            ],
            tactical_contexts=model_tactical_contexts + quant_model_tactical_contexts,
            match_contexts=match_contexts if i % PLAY_MATCH_EVERY_N == 0 else [],
            tactic_groups=[val_tactics],
        )

        print(f"Creating dataloaders...")

        flat_prior_to_oracle = [gs for g in games_prior_to_oracle for gs in g]
        flat_oracle = [gs for g in oracle_games for gs in g]

        print(f"{len(flat_prior_to_oracle)=}")
        print(f"{len(flat_oracle)=}")

        prior_to_oracle_dataloader = dataloader_from_oracle(
            model,
            flat_prior_to_oracle,
            batch_size=BATCH_SIZE,
            keep_rate=1.0,
        )

        oracle_dataloader = dataloader_from_oracle(
            model,
            flat_oracle,
            batch_size=BATCH_SIZE,
            keep_rate=0.1,
        )

        print(f"{len(oracle_dataloader)=}, {len(prior_to_oracle_dataloader)=}")

        combined_loader = combine_dataloaders(
            prior_to_oracle_dataloader, oracle_dataloader, weight_a=0.05, weight_b=0.95
        )

        training_result = train(model, combined_loader, optimizer, criterion, epochs=1)

        print(f"Validating...")
        avg_validation_loss = validate(
            model,
            val_loader,
            criterion,
        )

        print("Done validating\n\n")
        model_tensorboard_update(
            i,
            tb_writer,
            model,
            model_desc="",
            training_result=training_result,
            validation_loss=avg_validation_loss,
            make_visualizations=False,
        )

        if i % CHECKPOINT_EVERY_N == 0:
            torch.save(model.state_dict(), checkpoints_dir / f"{RUN_UID}_{i}.pth")


if __name__ == "__main__":
    main()
