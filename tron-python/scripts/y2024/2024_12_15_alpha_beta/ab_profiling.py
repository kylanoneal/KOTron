import json
import torch
import shutil
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from game.tron import Tron, GameStatus, DirectionUpdate, Direction
# from game.utility_gui import show_game_state

from ai.algos import (
    #choose_direction_model_naive,
    choose_direction_random,
    #choose_direction_minimax,
)

from ai.minimax import minimax_alpha_beta_eval_all, minimax_stack

from ai.model_architectures import FastNet, EvaluationNetConv3OneStride
from ai.tron_model import StandardTronModel

from ai.training import train_loop, make_dataloader, get_weights_sum_of_squares


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    state_dict = torch.load(
        "C:/Users/kylan/Documents/code/repos/KOTron/python/tasks/2024_12_15_alpha_beta/oldnet_self_train_continuation_v5_8.pth"
    )

    torch_model = EvaluationNetConv3OneStride(grid_dim=10)
    torch_model.load_state_dict(state_dict)
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

    run_uid = "ab_vs_basic_profiling_v1"

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

    all_game_states = []

    games_tied = p1_wins = p2_wins = 0

    for i in range(1):

        game = Tron(num_players=2, num_rows=10, num_cols=10, random_starts=False)

        curr_game_states = [deepcopy(game)]

        while game.status == GameStatus.IN_PROGRESS:

            # ab_index = i
            # basic_index = (i + 1) % 2

            p1_dir_update = minimax_alpha_beta_eval_all(
                model,
                game,
                depth=6,
                maximizing_player_index=0,
                minimizing_player_index=1,
                is_maximizing_player=True,
                is_root=True,
                debug_mode=True
            )

            p2_dir_update = minimax_alpha_beta_eval_all(
                model,
                game,
                depth=6,
                maximizing_player_index=1,
                minimizing_player_index=0,
                is_maximizing_player=True,
                is_root=True,
                debug_mode=True
            )

            # p2_dir_update = choose_direction_random(game, 1)

            # while len(minimax_stack) > 0:
            #     explored_state = minimax_stack.pop(0)
            #     show_game_state(explored_state.game_state, explored_state)

            # p2_dir_update = DirectionUpdate(show_game_state(game), player_index=1)

            print(f"\nP1: {p1_dir_update.direction}")
            print(f"P2: {p2_dir_update.direction}")
            

            game = Tron.next(game, direction_updates=(p1_dir_update, p2_dir_update))

            curr_game_states.append(game)

        all_game_states.append(curr_game_states)

        if game.status == GameStatus.TIE:
            games_tied += 1
        elif game.status == GameStatus.P1_WIN:
            p1_wins += 1
        elif game.status == GameStatus.P2_WIN:
            p2_wins += 1

    print(f"P1 wins: {p1_wins}, p2 wins: {p2_wins}, ties: {games_tied}")
