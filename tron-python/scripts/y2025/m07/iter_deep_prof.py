import json
import torch
import shutil
import numpy as np
import random
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction, get_possible_directions, Player
#from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random


from tron.ai.search import iter_deepening_ab, cache, MinimaxContext, MinimaxResult
#from tron.ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult

from tron.ai.nnue import NnueTronModel
from tron.ai.tron_model import RandomTronModel

from tron.ai.training import train_loop, make_dataloader, get_weights_sum_of_squares, print_state_and_sos

from tron.io.to_proto import to_proto, from_proto

def main():

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    model = NnueTronModel(10, 10)

    state_dict = torch.load(
        r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\scripts\y2025\m07\runs\20250728-185050_nnue_v7_continuation\checkpoints\nnue_v7_continuation_2311.pth"    
    )
    model.load_state_dict(state_dict)

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

    run_uid = "id_prof_v1"

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

    n_train_sim_cycles = 1
    n_games_per_loop = 100
    checkpoint_every_n_cyles = 1

    p_neutral_start = 0.5
    p_obstacles = 0.5
    obstacle_density_range = (0.0, 0.3)

    total_games_tied = total_p1_wins = total_p2_wins = 0
 

    players = (Player(4, 2, True), Player(7, 2, True))

    game = GameState.from_players(players, num_rows=10, num_cols=10)

    game_status: StatusInfo = tron.get_status(game)


    while game_status.status == GameStatus.IN_PROGRESS:

        # NOTE: Need to reset accumulator once weights have been updated
        model.reset_acc()

        #print(f"{get_possible_directions(game, 0)=}")
        p1_mm_result: MinimaxResult = iter_deepening_ab(
            game,
            max_depth=6,
            mm_context=MinimaxContext(model, maximizing_player=0, minimizing_player=1)
        )

        p2_mm_result: MinimaxResult = iter_deepening_ab(
            game,
            max_depth=6,
            mm_context=MinimaxContext(model, maximizing_player=1, minimizing_player=0)
        )

        # p1_mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
        #     game,
        #     depth=6,
        #     is_maximizing_player=True,
        #     context=MinimaxContext(model, maximizing_player=0, minimizing_player=1)
        # )

        # p2_mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
        #     game,
        #     depth=6,
        #     is_maximizing_player=True,
        #     context=MinimaxContext(model, maximizing_player=1, minimizing_player=0)
        # )

        p1_direction = Direction.UP if p1_mm_result.principal_variation is None else p1_mm_result.principal_variation
        p2_direction = Direction.UP if p2_mm_result.principal_variation is None else p2_mm_result.principal_variation

        # p2_direction = show_game_state(game, step_through=True)

        #p1_direction = choose_direction_random(game, 0)
        #p2_direction = choose_direction_random(game, 1)


        game = tron.next(
            game, directions=(p1_direction, p2_direction)
        )
        game_status = tron.get_status(game)


if __name__ == "__main__":
    main()


