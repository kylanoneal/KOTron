from typing import Callable
from dataclasses import dataclass


import torch
import imageio
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
)

from tron.ai.tron_model import TronModel, RandomTronModel
from tron.ai.nnue import NnueTronModel, QuantizedNnueTronModel

from tron.ai import MCTS
from tron.ai.MCTS import MctsContext

from tron.ai.training import (
    train_loop,
    make_dataloader,
    get_sos_info,
)

from tron.ai.algos import choose_direction_basic_minimax

from tron.ai.benchmarks import (
    SPATIAL_TACTICS_5X5,
    DECISIVE_TACTICS_5X5,
    TIES_5X5,
    WINS_LOSSES_5X5,
    Tactic,
    TacticResult,
    match,
    run_tactic,
    run_value_benchmark,
)

from tron.utils.sim import get_start_position

from tron.utils.viz import render_tactic_benchmark_image


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


script_dir = Path(__file__).resolve().parent
VIZ_OUT_DIR = script_dir / "viz"
VIZ_OUT_DIR.mkdir(exist_ok=True)


def benchmark(
    i,
    tb_writer,
    value_contexts: list[ValueBenchmarkContext],
    tactical_contexts: list[TacticalBenchmarkContext],
    match_contexts: list[MatchContext],
):

    for vc in value_contexts:
        # Model value benchmarks
        tie_benchmark_avg_score = sum(
            [run_value_benchmark(b, vc.model) for b in TIES_5X5]
        ) / len(TIES_5X5)
        wl_benchmark_avg_score = sum(
            [run_value_benchmark(b, vc.model) for b in WINS_LOSSES_5X5]
        ) / len(WINS_LOSSES_5X5)

        tb_writer.add_scalar(
            f"Avg. Value Diff (Ties) ({vc.description})", tie_benchmark_avg_score, i
        )
        tb_writer.add_scalar(
            f"Avg. Value Diff (W/Ls) ({vc.description})", wl_benchmark_avg_score, i
        )

    for tc in tactical_contexts:
        # Tactical benchmarks

        spatial_passes = spatial_fails = 0

        for j, t in enumerate(SPATIAL_TACTICS_5X5):

            results = run_tactic(t, tc.dir_fn)

            for r in results:

                if r.correct_moves == len(r.tactic.opposing_dirs):
                    spatial_passes += 1
                else:
                    spatial_fails += 1

            viz = render_tactic_benchmark_image(results)

            tb_writer.add_image(
                f"tactics/spatial/{j}",
                viz,
                global_step=i,
                dataformats="HWC",
            )




        tb_writer.add_scalar(
            f"Spatial Tactics Pass Rate ({tc.description})",
            spatial_passes / (spatial_passes + spatial_fails),
            i,
        )


        decisive_passes = decisive_fails = 0

        for j, t in enumerate(DECISIVE_TACTICS_5X5):

            results = run_tactic(t, tc.dir_fn)

            for r in results:

                if r.correct_moves == len(r.tactic.opposing_dirs):
                    decisive_passes += 1
                else:
                    decisive_fails += 1

            viz = render_tactic_benchmark_image(results)

            tb_writer.add_image(
                f"tactics/decisive/{j}",
                viz,
                global_step=i,
                dataformats="HWC",
            )

        tb_writer.add_scalar(
            f"Decisive Tactics Pass Rate ({tc.description})",
            decisive_passes / (decisive_passes + decisive_fails),
            i,
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
    PRE_TRAIN_EPOCHS = 0  # 50
    PRE_TRAIN_KEEP_RATE = 0.0025  # 0.1
    KEEP_RATE = 0.5
    LR = 0.002  # 0.01  # 0.001

    ACC_DIM = 128

    # MCTS_ITERS = args.mcts_iters
    # TEMP = args.temp
    # EXPLR_FACTOR = args.explr_factor

    MCTS_ITERS = 128
    TEMP = 0.4  # 0.7
    EXPLR_FACTOR = 2.0

    QUANT_SCALE = 256

    RUN_DESCRIPTION = "refine_viz"

    NUM_ROWS = NUM_COLS = 5

    # SIM_GAME_DEPTH = 2
    WIN_REWARD = 1.5

    GAMES_PER_ITER = 16
    CHECKPOINT_EVERY_N = 5

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.2  # 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    TRAIN_ITERS = 100_000

    PRETRAIN_PLAY_MATCH_EVERY_N = 2
    PLAY_MATCH_EVERY_N = 20
    N_MATCH_START_POSITIONS = 100

    ptkr_str = f"{PRE_TRAIN_KEEP_RATE:.4f}".replace(".", "p")
    RUN_UID = f"L{LR}_B{BATCH_SIZE}_MCITERS{MCTS_ITERS}_NOPRETRAIN_ACCDIM{ACC_DIM}"
    #    RUN_UID = f"L{LR}_B{BATCH_SIZE}_MCITERS{MCTS_ITERS}_PTKR{ptkr_str}_ACCDIM{ACC_DIM}"

    ############################################
    # INITIALIZE MODELS
    ############################################

    # TODO: Train a model only on deep games as well?
    model = NnueTronModel(NUM_ROWS, NUM_COLS, acc_dim=ACC_DIM)

    # state_dict = torch.load(
    #     r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2025\m08\cnn\runs\20250810-224509_mcts_cnn_5x5\LR0.001_B4\checkpoints\LR0.001_B4_30.pth"
    # )
    # model.load_state_dict(state_dict, strict=True)

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

    for i in range(1):
        benchmark(
            i,
            tb_writer,
            value_contexts=[
                ValueBenchmarkContext(model, "non-quant model"),
            ],
            tactical_contexts=model_tactical_contexts,
            match_contexts=[],
        )


if __name__ == "__main__":
    main()
