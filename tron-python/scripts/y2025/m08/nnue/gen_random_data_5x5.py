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
from tron.ai.minimax import basic_minimax, MinimaxContext, MinimaxResult


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

    model = RandomTronModel()

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = "random_data_gen"

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

    n_train_sim_cycles = 200
    n_games_per_loop = 1024

    p_neutral_start = 0.5
    p_obstacles = 0.4
    obstacle_density_range = (0.0, 0.3)

    total_games_tied = total_p1_wins = total_p2_wins = 0
 
    for train_iter in range(n_train_sim_cycles):

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0

        for j in tqdm(range(n_games_per_loop)):

            is_neutral_start = p_neutral_start > random.random() 
            are_obstacles = p_obstacles > random.random()

            obstacle_density = random.uniform(obstacle_density_range[0], obstacle_density_range[1]) if are_obstacles else 0.0

            game = GameState.new_game(num_players=2, num_rows=5, num_cols=5, random_starts=True, neutral_starts=is_neutral_start, obstacle_density=obstacle_density)

            game_status: StatusInfo = tron.get_status(game)

            curr_game_states = [deepcopy(game)]

            while game_status.status == GameStatus.IN_PROGRESS:

  

                p1_mm_result: MinimaxResult = basic_minimax(
                    game,
                    depth=2,
                    is_maximizing_player=True,
                    context=MinimaxContext(model, maximizing_player=0, minimizing_player=1)
                )

                p2_mm_result: MinimaxResult = basic_minimax(
                    game,
                    depth=2,
                    is_maximizing_player=True,
                    context=MinimaxContext(model, maximizing_player=1, minimizing_player=0)
                )

                p1_dir = Direction.UP if p1_mm_result.principal_variation is None else p1_mm_result.principal_variation
                p2_dir = Direction.UP if p2_mm_result.principal_variation is None else p2_mm_result.principal_variation

                
                # child_visits = [c.n_visits for c in root.children]
                # print(f"{child_visits=}")

                #show_game_state(game, step_through=True)

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
