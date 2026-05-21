from typing import Callable
from dataclasses import dataclass

import torch
import shutil
import random
import datetime
import warnings

from tqdm import tqdm
from pathlib import Path
from functools import partial
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
from tron.ai.cnn import CnnTronModel

from tron.ai import MCTS
from tron.ai.MCTS import MctsContext

from tron.ai.training import (
    TrainingResult,
    LabeledExample,
    ModelExample,
    train_loop,
    make_dataset,
    get_sos_info,
    get_label_magnitude,
)

from tron.ai.algos import choose_direction_basic_minimax

from tron.ai.benchmarks import (
    SPATIAL_TACTICS_5X5,
    DECISIVE_TACTICS_5X5,
    TIES_5X5,
    DECISIVE_5X5,
    Tactic,
    TacticResult,
    match,
    run_tactic,
    run_value_benchmark,
)

from tron.utils.sim import get_start_position

from tron.utils.viz import render_tactic_benchmark_image, render_model_example_image


@dataclass
class ValueBenchmarkContext:
    model: TronModel
    description: str


@dataclass
class TacticalBenchmarkContext:
    dir_fn: Callable[[PovGameState], Direction]
    description: str


@dataclass
class MatchContext:
    p1_bc: TacticalBenchmarkContext
    p2_bc: TacticalBenchmarkContext
    starting_positions: list[GameState]

# NOTE: Does not guarantee game didn't end on next state 
# e.g. players chose to crash into each other to secure tie
def is_one_move_from_terminal(game_state: GameState):

    assert len(game_state.players) == 2
    assert tron.get_status(game_state).status == GameStatus.IN_PROGRESS

    p1_dirs = get_possible_directions(game_state, 0)
    p2_dirs = get_possible_directions(game_state, 1)

    if len(p1_dirs) == 0 or len(p2_dirs) == 0:
        return True
    elif len(p1_dirs) == 1 and len(p2_dirs) == 1:
        next_state = next(game_state, (p1_dirs[0], p2_dirs[0]))

        if tron.get_status(next_state).status != GameStatus.IN_PROGRESS:
            return True
    else:
        return False


def model_tensorboard_update(
    i: int,
    tb_writer: SummaryWriter,
    model: TronModel,
    training_result: TrainingResult,
    make_visualizations: bool,
):

    print(
        f"{training_result.avg_loss=:.3f}, {training_result.avg_prediction_magnitude=:.3f}"
    )

    sos_dict, total_sos = get_sos_info(model)
    print("\nSum of squares (weights/biases):")
    for param, sos_val in sos_dict.items():
        print(f"{param:40s} {sos_val}")

    tb_writer.add_scalar("Weights Sum of Squares", total_sos, i)
    tb_writer.add_scalar("Average Loss", training_result.avg_loss, i)
    tb_writer.add_scalar(
        "Average Prediction Magnitude", training_result.avg_prediction_magnitude, i
    )

    if make_visualizations:

        terminal_examples = []
        in_prog_examples = []

        for example in training_result.model_examples:

            game_state = example.labeled_example.pov_game_state.game_state

            if is_one_move_from_terminal(game_state):
                terminal_examples.append(example)
            else:
                in_prog_examples.append(example)

        tb_writer.add_image(
            f"training_examples/terminal",
            render_model_example_image(random.sample(terminal_examples, k=20)),
            global_step=i,
            dataformats="HWC",
        )

        tb_writer.add_image(
            f"training_examples/in_progress",
            render_model_example_image(random.sample(in_prog_examples, k=20)),
            global_step=i,
            dataformats="HWC",
        )


def benchmark(
    i,
    tb_writer,
    make_visualizations: bool,
    value_contexts: list[ValueBenchmarkContext],
    tactical_contexts: list[TacticalBenchmarkContext],
    match_contexts: list[MatchContext],
):
    # TODO: formalize
    value_benchmark_info = [(TIES_5X5, "ties"), (DECISIVE_5X5, "decisive")]

    # Model value benchmarks
    for vc in value_contexts:

        for value_benchmarks, bench_description in value_benchmark_info:

            results = []

            for vb in value_benchmarks:
                results.extend(run_value_benchmark(vb, vc.model))

            if make_visualizations:

                tb_writer.add_image(
                    f"value_benchmarks/{bench_description}/{vc.description}",
                    render_model_example_image(results),
                    global_step=i,
                    dataformats="HWC",
                )

            avg_diff = sum(
                [abs(r.labeled_example.label - r.prediction) for r in results]
            ) / len(results)

            tb_writer.add_scalar(
                f"Avg. Value Diff ({bench_description}) ({vc.description})", avg_diff, i
            )

    # TODO: formalize
    tactical_benchmark_info = [
        (SPATIAL_TACTICS_5X5, "spatial"),
        (DECISIVE_TACTICS_5X5, "decisive"),
    ]

    for tc in tactical_contexts:

        for tactics, tactics_description in tactical_benchmark_info:
            passes = fails = 0

            results = []

            for t in tactics:

                results.extend(run_tactic(t, tc.dir_fn))

            for r in results:

                if r.correct_moves == len(r.tactic.opposing_dirs):
                    passes += 1
                else:
                    fails += 1

            tb_writer.add_scalar(
                f"{tactics_description} tactics pass rate ({tc.description})",
                passes / (passes + fails),
                i,
            )

            if make_visualizations:

                tb_writer.add_image(
                    f"tactics/{tactics_description}/{tc.description}",
                    render_tactic_benchmark_image(results),
                    global_step=i,
                    dataformats="HWC",
                )

    # Match score

    for mc in match_contexts:

        p1_wins, p2_wins, ties = match(
            mc.p1_bc.dir_fn, mc.p2_bc.dir_fn, mc.starting_positions
        )

        p1_match_score = (p1_wins + ties * 0.5) / (p1_wins + p2_wins + ties)

        tb_writer.add_scalar(
            f"{mc.p1_bc.description} match score vs. {mc.p2_bc.description})",
            p1_match_score,
            i,
        )


def main():

    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--mcts_iters", type=int)
    # # parser.add_argument("--temp", type=float)
    # # parser.add_argument("--explr_factor", type=float)

    # parser.add_argument("--lr", type=float)
    # parser.add_argument("--batch_size", type=int)

    # args = parser.parse_args()

    BATCH_SIZE = 4
    PRE_TRAIN_EPOCHS = 1_000_000  # 50
    PRE_TRAIN_KEEP_RATE = 0.5  # 0.1
    KEEP_RATE = 0.5
    LR = 0.001  # 0.01  # 0.001

    STARTING_WEIGHTS = None # r"C:\Users\kylan\code\KOTron\tron-python\scripts\y2026\m05\old_runs\20260505-175141_quant_5x5_debug_v1\L0.001_B4\checkpoints\L0.001_B4_18245.pth"  # None


    # MCTS_ITERS = args.mcts_iters
    # TEMP = args.temp
    # EXPLR_FACTOR = args.explr_factor

    MCTS_ITERS = 16
    TEMP = 0.4  # 0.7
    EXPLR_FACTOR = 2.0

    RUN_DESCRIPTION = "CNN_5x5_perfect_data"

    NUM_ROWS = NUM_COLS = 5

    # SIM_GAME_DEPTH = 2
    WIN_REWARD = 1.5

    GAMES_PER_ITER = 64
    CHECKPOINT_EVERY_N = 5

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.2  # 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    TRAIN_ITERS = 100_000

    PRETRAIN_PLAY_MATCH_EVERY_N = 10
    PLAY_MATCH_EVERY_N = 20
    MAKE_VISUALIZATIONS_EVERY_N = 1
    N_MATCH_START_POSITIONS = 100

    ptkr_str = f"{PRE_TRAIN_KEEP_RATE:.4f}".replace(".", "p")
    RUN_UID = f"L{LR}_B{BATCH_SIZE}_MCITERS{MCTS_ITERS}_PTONLY"
    #    RUN_UID = f"L{LR}_B{BATCH_SIZE}_MCITERS{MCTS_ITERS}_PTKR{ptkr_str}_ACCDIM{ACC_DIM}"

    ############################################
    # INITIALIZE MODELS
    ############################################

    # TODO: Train a model only on deep games as well?
    model = CnnTronModel(num_rows=NUM_ROWS, num_cols=NUM_COLS)

    if STARTING_WEIGHTS is not None:
        state_dict = torch.load(STARTING_WEIGHTS)
        model.load_state_dict(state_dict, strict=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=LR)

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    tron_dir = Path(tron.__file__).resolve().parent.parent

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

    with open(
        tron_dir / "datasets" / "20260505_5x5_100starts.bin",
        "rb",
    ) as f:
        bin_match_starts = f.read()

    # TODO: Just flattening this for now
    match_starts_proto = tron.from_proto(bin_match_starts)

    match_starting_positions = []

    for g in match_starts_proto:
        assert len(g) == 1
        assert g[0].num_cols == NUM_COLS and g[0].num_rows == NUM_ROWS

        match_starting_positions.append(g[0])

    # TODO: Figure out a way to seed the random tron model so output is consistent run to run
    random_model = RandomTronModel()

    # random_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    # fresh_state_dict = torch.load(tron_dir / "models" / "20250810_5x5_random_init.pth")
    # random_model.load_state_dict(fresh_state_dict, strict=True)

    random_bot_tactical_contexts = [
        random_bot_d1_tbc := TacticalBenchmarkContext(
            partial(choose_direction_basic_minimax, model=random_model, depth=1),
            description="Random Model D1 Minimax",
        ),
        random_bot_d3_tbc := TacticalBenchmarkContext(
            partial(choose_direction_basic_minimax, model=random_model, depth=3),
            description="Random Model D3 Minimax",
        ),
    ]

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

    pretrain_match_contexts = [
        MatchContext(model_d1_tbc, random_bot_d1_tbc, match_starting_positions),
        # MatchContext(model_d1_bc, prev_model_d1_bc, match_starting_positions),
        MatchContext(model_d3_tbc, random_bot_d3_tbc, match_starting_positions),
        # MatchContext(model_d3_bc, prev_model_d3_bc, match_starting_positions),
    ]

    # prev_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    # prev_state_dict = torch.load(
    #     r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\scripts\y2025\m08\cnn\runs\20250808-164733_better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate\checkpoints\better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate_500.pth"
    # )
    # prev_model.load_state_dict(prev_state_dict, True)

    # prev_model_benchmark_contexts = [
    #     prev_model_d1_bc := BenchmarkContext(
    #         partial(_minimax_dir_fn, model=prev_model, depth=1),
    #         description="Prev. Model D1 Minimax",
    #     ),
    #     prev_model_d3_bc := BenchmarkContext(
    #         partial(_minimax_dir_fn, model=prev_model, depth=3),
    #         description="Prev. Model D3 Minimax",
    #     ),
    # ]

    ############################################
    # PRE-TRAIN
    ############################################

    if PRE_TRAIN_EPOCHS > 0:

        datasets_dir = Path(tron.__file__).resolve().parent.parent / "datasets"

        data_dir = datasets_dir / "20260511_5x5_perfect_play_all_starts_no_obstacles"

        games = []

        for i, data_file in tqdm(enumerate(data_dir.iterdir())):

            # Save the serialized data to a file.
            with open(data_file, "rb") as f:
                bin_data = f.read()

            games.extend(tron.from_proto(bin_data))

        for i in range(0, len(games), len(games) // 100):
            game = games[i]
            assert game[0].num_rows == NUM_ROWS and game[0].num_cols == NUM_COLS

        for i in range(PRE_TRAIN_EPOCHS):

            benchmark(
                i,
                tb_writer,
                i % MAKE_VISUALIZATIONS_EVERY_N == 0,
                value_contexts=[
                    ValueBenchmarkContext(model, "non-quant model"),
                ],
                tactical_contexts=model_tactical_contexts,
                match_contexts=(
                    pretrain_match_contexts
                    if i % PRETRAIN_PLAY_MATCH_EVERY_N == 0
                    else []
                ),
            )

            # # Save the serialized data to a file.
            # with open(data_file, "rb") as f:
            #     bin_data = f.read()

            # game_data = from_proto(bin_data)

            dataloader = make_dataset(
                games, batch_size=BATCH_SIZE, keep_rate=PRE_TRAIN_KEEP_RATE
            )

            training_result = train_loop(
                model, dataloader, optimizer, criterion, epochs=1
            )

            model_tensorboard_update(
                i,
                tb_writer,
                model,
                training_result,
                make_visualizations=i % MAKE_VISUALIZATIONS_EVERY_N == 0,
            )

            torch.save(
                model.state_dict(),
                checkpoints_dir / f"pretrain_{RUN_UID}_{i}.pth",
            )


if __name__ == "__main__":
    main()
