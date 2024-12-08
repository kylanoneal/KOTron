import json
import torch
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from game.ko_tron import KOTron, GameStatus
from game.utility_gui import init_utility_gui, show_game_state
from AI.algos import (
    choose_direction_model_naive,
    choose_direction_random,
    choose_direction_minimax,
)
from AI.model_architectures import FastNet, EvaluationNetConv3OneStride
from AI.tron_model import StandardTronModel

from AI.training import train_loop, make_dataloader, get_weights_sum_of_squares


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    state_dict = torch.load(
        "C://Users//kylan//Documents//code//repos//KOTron//python//tasks//2024_12_4_refactor//model_state.pth"
    )
    torch_model = EvaluationNetConv3OneStride(grid_dim=10)
    torch_model.load_state_dict(state_dict)
    torch_model = torch_model.to(device)

    old_model = StandardTronModel(torch_model, device)

    state_dict = torch.load(
        "C://Users//kylan//Documents//code//repos//KOTron//python//tasks//2024_12_7_optimize//runs//20241207-193229_oldnet_self_train_v1//checkpoints//oldnet_self_train_v1_30.pth"
    )
    torch_model = EvaluationNetConv3OneStride(grid_dim=10)
    torch_model.load_state_dict(state_dict)
    torch_model = torch_model.to(device)

    old_model_self_train = StandardTronModel(torch_model, device)

    ############################################
    # TRAINING SETUP / HYPERPARAMETERS
    ############################################

    # batch_size = 32
    # shuffle = True

    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(fast_model.model.parameters())

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = "fastnet_vs_old_v3"

    outer_run_folder = Path(__file__).resolve().parent / "runs"
    outer_run_folder.mkdir(exist_ok=True)

    run_folder = (
        outer_run_folder
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{run_uid}"
    )
    run_folder.mkdir()

    checkpoints_folder = run_folder / "checkpoints"
    checkpoints_folder.mkdir()

    tb_writer = SummaryWriter(log_dir=run_folder)

    ############################################
    # SIMULATION-TRAIN LOOP
    ############################################

    n_train_sim_cycles = 100
    n_games_per_loop = 100
    checkpoint_every_n_cyles = 20

    for train_iter in range(n_train_sim_cycles):

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0

        for j in tqdm(range(n_games_per_loop)):

            game = KOTron(num_players=2, num_rows=10, num_cols=10, random_starts=True)

            curr_game_states = [deepcopy(game)]

            while game.status == GameStatus.IN_PROGRESS:

                p1_direction_update = choose_direction_minimax(
                    old_model_self_train, game, player_index=0, opponent_index=1, depth=2
                )

                p2_direction_update = choose_direction_minimax(
                    old_model, game, player_index=1, opponent_index=0, depth=2
                )

                game = KOTron.next(
                    game, direction_updates=[p1_direction_update, p2_direction_update]
                )

                curr_game_states.append(game)

                #show_game_state(game)

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

        # print(f"Training time! Iter: {train_iter}")

        # dataloader = make_dataloader(
        #     all_game_states, fast_model, batch_size=batch_size, shuffle=shuffle
        # )

        # avg_loss, avg_pred_magnitude = train_loop(fast_model.model, dataloader, optimizer, criterion, device, epochs=1)

        # tb_writer.add_scalar("Sum of Squares of Weights", get_weights_sum_of_squares(fast_model.model), train_iter)
        # tb_writer.add_scalar("Average Loss", avg_loss, train_iter)
        # tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, train_iter)

        # if train_iter % checkpoint_every_n_cyles == 0:
        #     torch.save(fast_model.model.state_dict(), checkpoints_folder / f"{run_uid}_{train_iter}.pth")
