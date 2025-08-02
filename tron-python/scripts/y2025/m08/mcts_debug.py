import json
import torch
import shutil
import random
import datetime
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction, Player
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
    model = RandomTronModel()
    # random_model = RandomTronModel()

    ############################################
    # TRAINING SETUP / HYPERPARAMETERS
    ############################################

    # batch_size = 32
    # shuffle = True

    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters())

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



    # TODO: Clear da cache once model has been updated, this shouldn't be here probably
    cache.clear()

    all_game_states = []

    games_tied = p1_wins = p2_wins = 0

    mcts_iters = 2500

    players = (Player(0,0,True), Player(1,2,True))
    grid = np.zeros((10,10), dtype=bool)

    indices = [(0,0), (1,3), (1,2), (1,1), (1,0)]

    for row, col in indices:
        grid[row][col] = True
    game = GameState(grid, players)

    game_status: StatusInfo = tron.get_status(game)

    curr_game_states = [deepcopy(game)]

    while game_status.status == GameStatus.IN_PROGRESS:

        # NOTE: Need to reset accumulator once weights have been updated
        #model.reset_acc()

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
        print(f"Tie")
        games_tied += 1
    elif game_status.winner_index == 0:
        p1_wins += 1
        print("P1 Win")
    elif game_status.winner_index == 1:
        p2_wins += 1
        print("P2 win")

