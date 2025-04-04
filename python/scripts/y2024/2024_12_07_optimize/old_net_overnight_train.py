import json
import torch
import shutil
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from game.tron import Tron, GameStatus
from ai.algos import (
    choose_direction_model_naive,
    choose_direction_random,
    choose_direction_minimax,
)
from ai.model_architectures import FastNet, EvaluationNetConv3OneStride
from ai.tron_model import StandardTronModel

from ai.training import train_loop, make_dataloader, get_weights_sum_of_squares


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    # state_dict = torch.load(
    #     "C://Users//kylan//Documents//code//repos//Tron//python//tasks//2024_12_refactor//model_state.pth"
    # )
    torch_model = EvaluationNetConv3OneStride(grid_dim=10)
    # torch_model.load_state_dict(state_dict)
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

    run_uid = "oldnet_self_train_v1"

    current_script_path = Path(__file__).resolve()

    outer_run_folder = current_script_path.parent / "runs"
    outer_run_folder.mkdir(exist_ok=True)

    run_folder = (
        outer_run_folder
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{run_uid}"
    )
    run_folder.mkdir()

    checkpoints_folder = run_folder / "checkpoints"
    checkpoints_folder.mkdir()

    backup_path = run_folder / f"{current_script_path.name.split('.')[0]}.bak.py"
    shutil.copy2(current_script_path, backup_path)

    tb_writer = SummaryWriter(log_dir=run_folder)

    ############################################
    # SIMULATION-TRAIN LOOP
    ############################################

    n_train_sim_cycles = 1000 * 1000
    n_games_per_loop = 400
    checkpoint_every_n_cyles = 10

    for train_iter in range(n_train_sim_cycles):

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0

        for j in tqdm(range(n_games_per_loop)):

            game = Tron(num_players=2, num_rows=10, num_cols=10, random_starts=True)

            curr_game_states = [deepcopy(game)]

            while game.status == GameStatus.IN_PROGRESS:

                p1_direction_update = choose_direction_minimax(
                    model, game, player_index=0, opponent_index=1, depth=2
                )

                p2_direction_update = choose_direction_minimax(
                    model, game, player_index=1, opponent_index=0, depth=2
                )

                game = Tron.next(
                    game, direction_updates=[p1_direction_update, p2_direction_update]
                )

                curr_game_states.append(game)

            all_game_states.append(curr_game_states)

            if game.status == GameStatus.TIE:
                games_tied += 1
            elif game.status == GameStatus.P1_WIN:
                p1_wins += 1
            elif game.status == GameStatus.P2_WIN:
                p2_wins += 1

        print(f"P1 wins: {p1_wins}, p2 wins: {p2_wins}, ties: {games_tied}")

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
