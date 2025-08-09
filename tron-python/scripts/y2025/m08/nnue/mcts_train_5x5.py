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

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction

# from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random

from tron.ai.minimax import basic_minimax, MinimaxContext

from tron.ai import MCTS
from tron.ai.MCTS import cache
from tron.ai.nnue import NnueTronModel
from tron.ai.tron_model import RandomTronModel

from tron.ai.training import (
    train_loop,
    make_dataloader,
    get_weights_sum_of_squares,
    print_state_and_sos,
)
from tron.ai.benchmarks import (
    BENCHMARKS_5X5,
    TIE_BENCHMARKS_5X5,
    WIN_LOSS_BENCHMARKS_5X5,
    run_benchmark,
    run_model_benchmark,
    match
)

from tron.io.to_proto import to_proto, from_proto


def main():

    NUM_ROWS = NUM_COLS = 5

    GAMES_PER_ITER = 64
    CHECKPOINT_EVERY = 10
    PLAY_MATCH_EVERY = 10

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    ############################################
    # ZMQ SETUP
    ############################################

    # ctx = zmq.Context()
    # sock = ctx.socket(zmq.DEALER)
    # # give yourself a unique ID so server can reply
    # my_id = uuid.uuid4().hex.encode()
    # sock.setsockopt(zmq.IDENTITY, my_id)
    # sock.connect(f"tcp://192.168.1.65:{5555}")
    # print(f"[CLIENT {my_id!r}] connected")

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    model = NnueTronModel(NUM_ROWS, NUM_COLS)

    state_dict = torch.load(
        r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\models\pretrain_mcts_5x5_v2_190.pth"
    )
    model.load_state_dict(state_dict, strict=True)
    model.reset_acc()

    model.to(device)

    # random_model = RandomTronModel()

    ############################################
    # TRAINING SETUP / HYPERPARAMETERS
    ############################################

    batch_size = 32

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = f"mcts_5x5_v3"

    current_script_path = Path(__file__).resolve()

    outer_run_folder = current_script_path.parent / "runs"
    outer_run_folder.mkdir(exist_ok=True)

    run_folder = (
        outer_run_folder
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{run_uid}"
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
    # PRE-TRAIN
    ############################################

    # datasets_dir = Path(tron.__file__).resolve().parent.parent / "datasets"

    # data_dir = datasets_dir / "20250804_5x5_random_d2_200k"

    # for i, data_file in tqdm(enumerate(data_dir.iterdir())):

    #     # Save the serialized data to a file.
    #     with open(data_file, "rb") as f:
    #         bin_data = f.read()

    #     game_data = from_proto(bin_data)

    #     dataloader = make_dataloader(
    #         game_data, batch_size=batch_size
    #     )

    #     avg_loss, avg_pred_magnitude = train_loop(model, dataloader, optimizer, criterion, device, epochs=1)
    #     weights_sos = get_weights_sum_of_squares(model)

    #     print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {weights_sos=:.3f}")
    #     print_state_and_sos(model, decimals=3)

    #     if i % 10 == 0:
    #         torch.save(model.state_dict(), checkpoints_folder / f"pretrain_{run_uid}_{i}.pth")

    ############################################
    # BENCHMARKING SETUP
    ############################################

    def _mcts_dir_fn(game, model, hero_index, n_iterations):

        model.reset_acc()
        root: MCTS.Node = MCTS.search(
            model, game, hero_index, n_iterations=n_iterations
        )

        p1_dir, _, _ = MCTS.get_move_pair(root, hero_index, temp=0.0)

        return p1_dir

    def _d1_minimax_p1_dir_fn(game, model):

        mm_result = basic_minimax(
            game,
            depth=1,
            is_maximizing_player=True,
            context=MinimaxContext(model, 0, 1),
        )
        return (
            mm_result.principal_variation
            if mm_result.principal_variation is not None
            else Direction.UP
        )

    d1_minimax_p1_dir_fn = partial(_d1_minimax_p1_dir_fn, model=model)
    mcts_p1_dir_fn = partial(_mcts_dir_fn, model=model, hero_index=0, n_iterations=1000)

    baseline_model = NnueTronModel(NUM_ROWS, NUM_COLS)

    state_dict = torch.load(
        r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\models\pretrain_mcts_5x5_v2_190.pth"
    )
    baseline_model.load_state_dict(state_dict, strict=True)
    baseline_model.reset_acc()
    baseline_model.to(device)

    baseline_mcts_p2_dir_fn = partial(
        _mcts_dir_fn, model=baseline_model, hero_index=1, n_iterations=1000
    )

    match_starting_positions = []

    for i in range(100):

        is_neutral_start = P_NEUTRAL_START > random.random()
        are_obstacles = P_OBSTACLES > random.random()

        obstacle_density = (
            random.uniform(OBSTACLE_DENSITY_RANGE[0], OBSTACLE_DENSITY_RANGE[1])
            if are_obstacles
            else 0.0
        )

        match_starting_positions.append(
            GameState.new_game(
                num_players=2,
                num_rows=NUM_ROWS,
                num_cols=NUM_COLS,
                random_starts=True,
                neutral_starts=is_neutral_start,
                obstacle_density=obstacle_density,
            )
        )

    ############################################
    # CLIENT LOOP
    ############################################

    total_games_tied = total_p1_wins = total_p2_wins = 0

    for i in range(1_000_000):
        # TODO: Clear da cache once model has been updated, this shouldn't be here probably
        cache.clear()

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0

        mcts_iters = 500

        for _ in tqdm(range(GAMES_PER_ITER)):

            is_neutral_start = P_NEUTRAL_START > random.random()
            are_obstacles = P_OBSTACLES > random.random()

            obstacle_density = (
                random.uniform(OBSTACLE_DENSITY_RANGE[0], OBSTACLE_DENSITY_RANGE[1])
                if are_obstacles
                else 0.0
            )

            game = GameState.new_game(
                num_players=2,
                num_rows=NUM_ROWS,
                num_cols=NUM_COLS,
                random_starts=True,
                neutral_starts=is_neutral_start,
                obstacle_density=obstacle_density,
            )

            game_status: StatusInfo = tron.get_status(game)

            curr_game_states = [deepcopy(game)]

            next_root = None
            while game_status.status == GameStatus.IN_PROGRESS:

                # NOTE: Need to reset accumulator once weights have been updated
                model.reset_acc()

                root: MCTS.Node = MCTS.search(
                    model, game, 0, n_iterations=mcts_iters, root=next_root
                )

                p1_dir, p2_dir, next_root = MCTS.get_move_pair(root, 0, temp=TEMP)

                # child_visits = [c.n_visits for c in root.children]
                # print(f"{child_visits=}")

                # show_game_state(game, step_through=True)

                # p1_dir = choose_direction_random(game, 0)
                # p2_dir = choose_direction_random(game, 1)

                game = tron.next(game, directions=(p1_dir, p2_dir))

                if next_root is not None:
                    assert game == next_root.game_state

                curr_game_states.append(game)

                game_status = tron.get_status(game)

            all_game_states.append(curr_game_states)

            if game_status.status == GameStatus.TIE:
                # print(f"Tie")
                games_tied += 1
            elif game_status.winner_index == 0:
                p1_wins += 1
                # print("P1 Win")
            elif game_status.winner_index == 1:
                p2_wins += 1
                # print("P2 win")

        # Serialize game data
        serialized_data = to_proto(all_game_states)

        # Save the serialized data to a file.
        with open(
            data_out_folder / f"gamedata_{i}_ngames_{GAMES_PER_ITER}.bin", "wb"
        ) as f:
            f.write(serialized_data)

        print(f"This iter P1 wins: {p1_wins}, p2 wins: {p2_wins}, ties: {games_tied}")

        total_games_tied += games_tied
        total_p1_wins += p1_wins
        total_p2_wins += p2_wins

        print(
            f"Running total P1 wins: {total_p1_wins}, p2 wins: {total_p2_wins}, ties: {total_games_tied}"
        )

        tb_writer.add_scalar("Player 1 Winrate", p1_wins / GAMES_PER_ITER, i)
        tb_writer.add_scalar("Tie Rate", games_tied / GAMES_PER_ITER, i)
        tb_writer.add_scalar(
            "Average Game Length",
            sum([len(game_states) for game_states in all_game_states]) / GAMES_PER_ITER,
            i,
        )

        print(f"Training time! Iter: {i}")

        dataloader = make_dataloader(all_game_states, batch_size=batch_size)

        avg_loss, avg_pred_magnitude = train_loop(
            model, dataloader, optimizer, criterion, device, epochs=1
        )
        weights_sos = get_weights_sum_of_squares(model)

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {weights_sos=:.3f}")
        print_state_and_sos(model, decimals=3)

        tb_writer.add_scalar("Sum of Squares of Weights", weights_sos, i)
        tb_writer.add_scalar("Average Loss", avg_loss, i)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, i)

        # Model value benchmarks
        tie_benchmark_avg_score = sum(
            [run_model_benchmark(b, model) for b in TIE_BENCHMARKS_5X5]
        ) / len(TIE_BENCHMARKS_5X5)
        wl_benchmark_avg_score = sum(
            [run_model_benchmark(b, model) for b in WIN_LOSS_BENCHMARKS_5X5]
        ) / len(WIN_LOSS_BENCHMARKS_5X5)

        tb_writer.add_scalar("Avg. Model Diff (Ties)", tie_benchmark_avg_score, i)
        tb_writer.add_scalar("Avg. Model Diff (W/Ls)", wl_benchmark_avg_score, i)

        # Tactical benchmarks
        mcts_benchmark_avg_score = sum(
            [run_benchmark(b, mcts_p1_dir_fn) for b in BENCHMARKS_5X5]
        ) / len(BENCHMARKS_5X5)
        d1_mm_benchmark_avg_score = sum(
            [run_benchmark(b, d1_minimax_p1_dir_fn) for b in BENCHMARKS_5X5]
        ) / len(BENCHMARKS_5X5)
        tb_writer.add_scalar("Avg. Benchmark Score (MCTS)", mcts_benchmark_avg_score, i)
        tb_writer.add_scalar(
            "Avg. Benchmark Score (D1 Minimax)", d1_mm_benchmark_avg_score, i
        )

        # Match score

        if i % PLAY_MATCH_EVERY == 0:
            curr_model_wins, baseline_wins, ties = match(mcts_p1_dir_fn, baseline_mcts_p2_dir_fn, match_starting_positions)

            match_score = (curr_model_wins + ties * 0.5) / (curr_model_wins + baseline_wins + ties)
            tb_writer.add_scalar("Match score", match_score, i)



        if i % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), checkpoints_folder / f"{run_uid}_{i}.pth")


if __name__ == "__main__":
    main()
