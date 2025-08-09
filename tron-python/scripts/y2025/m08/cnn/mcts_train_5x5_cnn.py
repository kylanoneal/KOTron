import zmq
import uuid
import json
import torch
import shutil
import random
import datetime
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


def benchmark(i, tb_writer, model, bench_contexts, match_contexts):

    # Model value benchmarks
    tie_benchmark_avg_score = sum(
        [run_model_benchmark(b, model) for b in TIE_BENCHMARKS_5X5]
    ) / len(TIE_BENCHMARKS_5X5)
    wl_benchmark_avg_score = sum(
        [run_model_benchmark(b, model) for b in WIN_LOSS_BENCHMARKS_5X5]
    ) / len(WIN_LOSS_BENCHMARKS_5X5)

    tb_writer.add_scalar("Avg. Model Diff (Ties)", tie_benchmark_avg_score, i)
    tb_writer.add_scalar("Avg. Model Diff (W/Ls)", wl_benchmark_avg_score, i)

    for bc in bench_contexts:
        # Tactical benchmarks
        avg_benchmark_score = sum(
            [run_benchmark(b, bc.dir_fn) for b in BENCHMARKS_5X5]
        ) / len(BENCHMARKS_5X5)

        tb_writer.add_scalar(
            f"Avg. Benchmark Score ({bc.description})", avg_benchmark_score, i
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

    RUN_DESCRIPTION = "better_5x5_cnn_amsgrad_mcts500_temp0p7_8batchsize_0p5keeprate"

    NUM_ROWS = NUM_COLS = 5

    SIM_GAME_DEPTH = 2
    MCTS_ITERS = 500
    TEMP = 0.7
    GAMES_PER_ITER = 256
    CHECKPOINT_EVERY_N = 10

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    PLAY_MATCH_EVERY_N = 10
    N_MATCH_START_POSITIONS = 100

    BATCH_SIZE = 8
    KEEP_RATE = 0.5
    LR = 0.001

    ############################################
    # INITIALIZE MODELS
    ############################################

    # TODO: Train a model only on deep games as well?
    model = CnnTronModel(NUM_ROWS, NUM_COLS)

    # state_dict = torch.load(
    #     r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\models\pretrain_mcts_5x5_v2_190.pth"
    # )
    # model.load_state_dict(state_dict, strict=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    current_script_path = Path(__file__).resolve()

    outer_run_folder = current_script_path.parent / "runs"
    outer_run_folder.mkdir(exist_ok=True)

    run_folder = (
        outer_run_folder
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{RUN_DESCRIPTION}"
    )
    run_folder.mkdir()

    data_out_folder = run_folder / "game_data"
    data_out_folder.mkdir()

    checkpoints_folder = run_folder / "checkpoints"
    checkpoints_folder.mkdir()

    backup_path = run_folder / f"{current_script_path.name.split('.')[0]}.bak.py"
    shutil.copy2(current_script_path, backup_path)

    tb_writer = SummaryWriter(log_dir=run_folder)

    ############################################
    # BENCHMARKING SETUP
    ############################################

    def _minimax_dir_fn(
        pov_game_state: PovGameState, model: TronModel, depth: int
    ) -> Direction:


        opponent_index = 0 if pov_game_state.hero_index == 1 else 1
        mm_context = MinimaxContext(model.run_inference, pov_game_state.hero_index, opponent_index, win_magnitude=100_000.0)

        mm_result = basic_minimax(
            pov_game_state.game_state,
            depth=depth,
            is_maximizing_player=True,
            context=mm_context,
        )
        return (
            mm_result.principal_variation
            if mm_result.principal_variation is not None
            else Direction.UP
        )

    model_benchmark_contexts = [
        model_d1_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=model, depth=1), description="D1 Minimax"
        ),
        model_d3_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=model, depth=3), description="D3 Minimax"
        ),
    ]

    fresh_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    fresh_model_benchmark_contexts = [
        fresh_model_d1_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=fresh_model, depth=1),
            description="Fresh Model D1 Minimax",
        ),
        fresh_model_d3_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=fresh_model, depth=3),
            description="Fresh Model D3 Minimax",
        ),
    ]

    prev_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    prev_state_dict = torch.load(r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\scripts\y2025\m08\cnn\runs\20250808-164733_better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate\checkpoints\better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate_500.pth")
    prev_model.load_state_dict(prev_state_dict, True)


    prev_model_benchmark_contexts = [
        prev_model_d1_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=prev_model, depth=1),
            description="Prev. Model D1 Minimax",
        ),
        prev_model_d3_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=prev_model, depth=3),
            description="Prev. Model D3 Minimax",
        ),
    ]

    match_starting_positions = [
        get_start_position(
            NUM_ROWS, NUM_COLS, P_NEUTRAL_START, P_OBSTACLES, OBSTACLE_DENSITY_RANGE
        )
        for _ in range(N_MATCH_START_POSITIONS)
    ]

    match_contexts = [
        MatchContext(
            model_d1_bc, fresh_model_d1_bc, match_starting_positions
        ),
        MatchContext(
            model_d1_bc, prev_model_d1_bc, match_starting_positions
        ),
        MatchContext(
            model_d3_bc, fresh_model_d3_bc, match_starting_positions
        ),
        MatchContext(
            model_d3_bc, prev_model_d3_bc, match_starting_positions
        ),
    ]

    ############################################
    # PRE-TRAIN
    ############################################

    datasets_dir = Path(tron.__file__).resolve().parent.parent / "datasets"

    data_dir = datasets_dir / "20250804_5x5_random_d2_200k"

    pre_train_iters = 0

    games = []

    for i, data_file in tqdm(enumerate(data_dir.iterdir())):

        # Save the serialized data to a file.
        with open(data_file, "rb") as f:
            bin_data = f.read()

        games.extend(from_proto(bin_data))


    for i in range(5):

        pre_train_iters = i

        benchmark(i, tb_writer, model, model_benchmark_contexts, match_contexts if i % PLAY_MATCH_EVERY_N == 0 else [])

        # # Save the serialized data to a file.
        # with open(data_file, "rb") as f:
        #     bin_data = f.read()

        # game_data = from_proto(bin_data)

        dataloader = make_dataloader(games, batch_size=BATCH_SIZE, keep_rate=0.0025)

        avg_loss, avg_pred_magnitude = train_loop(
            model, dataloader, optimizer, criterion, epochs=1
        )


        sos_dict, total_sos = get_sos_info(model)

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {total_sos=}")

        # print("\nSum of squares (weights/biases):")
        # for param, sos_val in sos_dict.items():
        #     print(f"{param:40s} {sos_val}")

        tb_writer.add_scalar("Weights Sum of Squares", total_sos, i)
        tb_writer.add_scalar("Average Loss", avg_loss, i)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, i)

        if i % 10 == 0:
            torch.save(
                model.state_dict(),
                checkpoints_folder / f"pretrain_{RUN_DESCRIPTION}_{i}.pth",
            )

    ############################################
    # CLIENT LOOP
    ############################################

    cache = LRUCache(maxsize=20000000)
    @cached(cache)
    def lru_eval(pov_game_state: PovGameState):
        return model.run_inference(pov_game_state)


    total_p1_wins = total_p2_wins = total_ties = 0

    for i in range(pre_train_iters + 1, 1_000_000):

        games: list[list[GameState]] = []

        p1_wins = p2_wins = ties = 0

        # Clear cache once model updated
        cache.clear()

        for _ in tqdm(range(GAMES_PER_ITER)):

            game_state = get_start_position(
                NUM_ROWS, NUM_COLS, P_NEUTRAL_START, P_OBSTACLES, OBSTACLE_DENSITY_RANGE
            )

            game_status: StatusInfo = tron.get_status(game_state)

            current_game: list[GameState] = [game_state]

            next_root = None
            while game_status.status == GameStatus.IN_PROGRESS:

                # p1_mm_result = basic_minimax(
                #     game_state,
                #     SIM_GAME_DEPTH,
                #     is_maximizing_player=True,
                #     context=MinimaxContext(model.run_inference, 0, 1),
                # )

                # p2_mm_result = basic_minimax(
                #     game_state,
                #     SIM_GAME_DEPTH,
                #     is_maximizing_player=True,
                #     context=MinimaxContext(model.run_inference, 1, 0),
                # )

                # p1_dir = (
                #     p1_mm_result.principal_variation
                #     if p1_mm_result.principal_variation is not None
                #     else Direction.UP
                # )

                # p2_dir = (
                #     p2_mm_result.principal_variation
                #     if p2_mm_result.principal_variation is not None
                #     else Direction.UP
                # )


                root: MCTS.Node = MCTS.search(
                    lru_eval, game_state, 0, n_iterations=MCTS_ITERS, root=next_root
                )

                p1_dir, p2_dir, next_root = MCTS.get_move_pair(root, 0, temp=TEMP)

                # show_game_state(game_state, step_through=True)

                # p1_dir = choose_direction_random(game_state, 0)
                # p2_dir = choose_direction_random(game_state, 1)

                game_state = tron.next(game_state, directions=(p1_dir, p2_dir))

                if next_root is not None:
                    assert game_state == next_root.game_state

                current_game.append(game_state)

                game_status = tron.get_status(game_state)

            games.append(current_game)

            if game_status.status == GameStatus.TIE:
                # print(f"Tie")
                ties += 1
            elif game_status.winner_index == 0:
                p1_wins += 1
                # print("P1 Win")
            elif game_status.winner_index == 1:
                p2_wins += 1
                # print("P2 win")

        # Serialize game data
        serialized_data = to_proto(games)

        # Save the serialized data to a file.
        with open(
            data_out_folder / f"gamedata_{i}_ngames_{GAMES_PER_ITER}.bin", "wb"
        ) as f:
            f.write(serialized_data)

        print("\n" + "-" * 25 + "\n")
        print(f"{i=}, {p1_wins=}, {p2_wins=}, {ties=}")

        total_ties += ties
        total_p1_wins += p1_wins
        total_p2_wins += p2_wins

        print(f"{total_p1_wins}, {total_p2_wins=}, {total_ties=}")

        tb_writer.add_scalar("P1 Wins / Total Wins", p1_wins / (p1_wins + p2_wins), i)
        tb_writer.add_scalar("Tie Rate", ties / GAMES_PER_ITER, i)
        tb_writer.add_scalar(
            "Average Game Length",
            sum([len(game) for game in games]) / GAMES_PER_ITER,
            i,
        )

        dataloader = make_dataloader(games, batch_size=BATCH_SIZE, keep_rate=KEEP_RATE)

        avg_loss, avg_pred_magnitude = train_loop(
            model, dataloader, optimizer, criterion, epochs=1
        )

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}")

        sos_dict, total_sos = get_sos_info(model)
        print("\nSum of squares (weights/biases):")
        for param, sos_val in sos_dict.items():
            print(f"{param:40s} {sos_val}")

        tb_writer.add_scalar("Weights Sum of Squares", total_sos, i)
        tb_writer.add_scalar("Average Loss", avg_loss, i)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, i)

        benchmark(i, tb_writer, model, model_benchmark_contexts, match_contexts if i % PLAY_MATCH_EVERY_N == 0 else [])

        if i % CHECKPOINT_EVERY_N == 0:
            torch.save(
                model.state_dict(), checkpoints_folder / f"{RUN_DESCRIPTION}_{i}.pth"
            )


if __name__ == "__main__":
    main()
