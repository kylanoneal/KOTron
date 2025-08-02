import json
import torch
import shutil
import random
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction
from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random


#from tron.ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult

from tron.ai import MCTS
from tron.ai.MCTS import cache
from tron.ai.nnue import NnueTronModel
from tron.ai.tron_model import RandomTronModel

from tron.ai.training import train_loop, make_dataloader, get_weights_sum_of_squares, print_state_and_sos

from tron.io.to_proto import to_proto, from_proto


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    model = NnueTronModel(10, 10)

    state_dict = torch.load(
        r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\scripts\y2025\m07\runs\20250731-223656_nnue_v7_continuation\checkpoints\nnue_v7_continuation_1415.pth"
    )
    model.load_state_dict(state_dict)
    model.reset_acc()

    model.to(device)

    # random_model = RandomTronModel()

    ############################################
    # TRAINING SETUP / HYPERPARAMETERS
    ############################################

    batch_size = 32
    shuffle = True

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = "nnue_v7_continuation"

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
    
    # data_dir = Path(r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2025\m06\make_random_data\runs\20250604-220539_obstacles_v2\game_data")

    # for i, data_file in tqdm(enumerate(list(data_dir.iterdir())[:200])):
        
    #     # Save the serialized data to a file.
    #     with open(data_file, "rb") as f:
    #         bin_data = f.read()

    #     game_data = from_proto(bin_data)

    #     dataloader = make_dataloader(
    #         game_data, batch_size=batch_size, shuffle=shuffle
    #     )

    #     avg_loss, avg_pred_magnitude = train_loop(model, dataloader, optimizer, criterion, device, epochs=1)
    #     weights_sos = get_weights_sum_of_squares(model)

    #     print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {weights_sos=:.3f}")
    #     print_state_and_sos(model, decimals=3)

    #     if i % 10 == 0:
    #         torch.save(model.state_dict(), checkpoints_folder / f"pretrain_{run_uid}_{i}.pth")


    ############################################
    # SIMULATION-TRAIN LOOP
    ############################################

    n_train_sim_cycles = 100_000
    n_games_per_loop = 64
    checkpoint_every_n_cyles = 5

    p_neutral_start = 0.75
    p_obstacles = 0.4
    obstacle_density_range = (0.0, 0.3)

    total_games_tied = total_p1_wins = total_p2_wins = 0
 
    for train_iter in range(n_train_sim_cycles):

        # TODO: Clear da cache once model has been updated, this shouldn't be here probably
        cache.clear()

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0

        if train_iter < 500:
            mcts_iters = 10
        elif train_iter < 1000:
            mcts_iters = 40
        elif train_iter < 2000:
            mcts_iters = 100
        else:
            mcts_iters = 200

        mcts_iters = 400

        for j in tqdm(range(n_games_per_loop)):

            is_neutral_start = p_neutral_start > random.random() 
            are_obstacles = p_obstacles > random.random()

            obstacle_density = random.uniform(obstacle_density_range[0], obstacle_density_range[1]) if are_obstacles else 0.0

            game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True, neutral_starts=is_neutral_start, obstacle_density=obstacle_density)

            game_status: StatusInfo = tron.get_status(game)

            curr_game_states = [deepcopy(game)]

            while game_status.status == GameStatus.IN_PROGRESS:

                # NOTE: Need to reset accumulator once weights have been updated
                model.reset_acc()

                root: MCTS.Node = MCTS.search(model, game, 0, n_iterations=mcts_iters)

                p1_dir, p2_dir = MCTS.get_move_pair(root, 0, temp=0.01)

                # child_visits = [c.n_visits for c in root.children]
                # print(f"{child_visits=}")

                show_game_state(game, step_through=True)

                # p1_dir = choose_direction_random(game, 0)
                # p2_dir = choose_direction_random(game, 1)


                game = tron.next(
                    game, directions=(p1_dir, p2_dir)
                )

                curr_game_states.append(game)

                game_status = tron.get_status(game)

            all_game_states.append(curr_game_states)

            if game_status.status == GameStatus.TIE:
                #print(f"Tie")
                games_tied += 1
            elif game_status.winner_index == 0:
                p1_wins += 1
                #print("P1 Win")
            elif game_status.winner_index == 1:
                p2_wins += 1
                #print("P2 win")

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
            all_game_states, batch_size=batch_size, shuffle=shuffle
        )

        avg_loss, avg_pred_magnitude = train_loop(model, dataloader, optimizer, criterion, device, epochs=1)
        weights_sos = get_weights_sum_of_squares(model)

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {weights_sos=:.3f}")
        print_state_and_sos(model, decimals=3)

        tb_writer.add_scalar("Sum of Squares of Weights", weights_sos, train_iter)
        tb_writer.add_scalar("Average Loss", avg_loss, train_iter)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, train_iter)

        if train_iter % checkpoint_every_n_cyles == 0:
            torch.save(model.state_dict(), checkpoints_folder / f"{run_uid}_{train_iter}.pth")


