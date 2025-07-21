import json
import torch
import shutil
import random
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from game import tron
from game.tron import GameState, GameStatus, StatusInfo, Direction
from game.utility_gui import show_game_state

# from ai.algos import (
#     #choose_direction_model_naive,
#     # choose_direction_random,
#     #choose_direction_minimax,
# )

from ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult
from ai.model_architectures import FastNet, EvaluationNetConv3OneStride, LeakyReLU, Transformer, EpicTransformer
from ai.tron_model import StandardTronModel

from ai.training import train_loop, make_dataloader, get_weights_sum_of_squares

from tron_io.to_proto import to_proto, from_proto


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    state_dict = torch.load(
        "C:/Users/kylan/Documents/code/repos/KOTron/python/scripts/y2025/m05/experiments/runs/20250524-004611_obstacles_v2/checkpoints/obstacles_v2_155.pth"
    )
    torch_model = LeakyReLU(grid_dim=10)
    torch_model.load_state_dict(state_dict)

    #torch_model = EpicTransformer(embed_dim=3, num_heads=3)
    torch_model = torch_model.to(device)

    model = StandardTronModel(torch_model, device)

    ############################################
    # TRAINING SETUP / HYPERPARAMETERS
    ############################################

    batch_size = 32
    shuffle = True

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.parameters())

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = "obstacles_v2"

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
    # SIMULATION-TRAIN LOOP
    ############################################

    n_train_sim_cycles = 1_000_000
    n_games_per_loop = 384
    checkpoint_every_n_cyles = 5

    p_neutral_start = 1.0
    p_obstacles = 0.5
    obstacle_density_range = (0.0, 0.3)

    total_games_tied = total_p1_wins = total_p2_wins = 0
 
    for train_iter in range(n_train_sim_cycles):

        # TODO: Clear da cache once model has been updated, this shouldn't be here probably
        cache.clear()

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0

        for j in tqdm(range(n_games_per_loop)):

            is_neutral_start = p_neutral_start > random.random() 
            are_obstacles = p_obstacles > random.random()

            obstacle_density = random.uniform(obstacle_density_range[0], obstacle_density_range[1]) if are_obstacles else 0.0

            game = tron.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True, neutral_starts=is_neutral_start, obstacle_density=obstacle_density)

            game_status: StatusInfo = tron.get_status(game)

            curr_game_states = [deepcopy(game)]

            while game_status.status == GameStatus.IN_PROGRESS:


                p1_mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
                    game,
                    depth=5,
                    is_maximizing_player=True,
                    context=MinimaxContext(model, maximizing_player=0, minimizing_player=1)
                )

                # p2_mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
                #     game,
                #     depth=3,
                #     is_maximizing_player=True,
                #     context=MinimaxContext(model, maximizing_player=1, minimizing_player=0)
                # )

                p1_direction = Direction.UP if p1_mm_result.principal_variation is None else p1_mm_result.principal_variation
                # p2_direction = Direction.UP if p2_mm_result.principal_variation is None else p2_mm_result.principal_variation
                p2_direction = show_game_state(game, step_through=True)


                game = tron.next(
                    game, directions=(p1_direction, p2_direction)
                )

                curr_game_states.append(game)

                game_status = tron.get_status(game)

            all_game_states.append(curr_game_states)

            if game_status.status == GameStatus.TIE:
                games_tied += 1
            elif game_status.winner_index == 0:
                p1_wins += 1
            elif game_status.winner_index == 1:
                p2_wins += 1

        # Serialize game data
        serialized_data = to_proto(all_game_states)

        # Save the serialized data to a file.
        with open(data_out_folder / f"game_data_iter_{train_iter}_ngames_{n_games_per_loop}.bin", "wb") as f:
            f.write(serialized_data)


        print(f"This iter P1 wins: {p1_wins}, p2 wins: {p2_wins}, ties: {games_tied}")

        total_games_tied += games_tied
        total_p1_wins += p1_wins
        total_p2_wins += p2_wins

    
        print(f"Running total P1 wins: {total_p1_wins}, p2 wins: {total_p2_wins}, ties: {total_games_tied}")

        tb_writer.add_scalar("Player 1 Winrate", p1_wins / n_games_per_loop, train_iter)
        tb_writer.add_scalar("Tie Rate", games_tied / n_games_per_loop, train_iter)
        tb_writer.add_scalar(
            "Average Game Length",
            sum([len(game_states) for game_states in all_game_states])
            / n_games_per_loop,
            train_iter,
        )

        print(f"Training time! Iter: {train_iter}")

        dataloader = make_dataloader(
            all_game_states, model, batch_size=batch_size, shuffle=shuffle
        )

        avg_loss, avg_pred_magnitude = train_loop(model.model, dataloader, optimizer, criterion, device, epochs=1)

        tb_writer.add_scalar("Sum of Squares of Weights", get_weights_sum_of_squares(model.model), train_iter)
        tb_writer.add_scalar("Average Loss", avg_loss, train_iter)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, train_iter)

        if train_iter % checkpoint_every_n_cyles == 0:
            torch.save(model.model.state_dict(), checkpoints_folder / f"{run_uid}_{train_iter}.pth")
