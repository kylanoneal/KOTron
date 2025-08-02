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

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction
# from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random


#from tron.ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult

from tron.ai import MCTS
from tron.ai.MCTS import cache
from tron.ai.nnue import NnueTronModel
from tron.ai.tron_model import RandomTronModel

from tron.ai.training import train_loop, make_dataloader, get_weights_sum_of_squares, print_state_and_sos

from tron.io.to_proto import to_proto, from_proto


def main():

    ############################################
    # ZMQ SETUP
    ############################################

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    # give yourself a unique ID so server can reply
    my_id = uuid.uuid4().hex.encode()
    sock.setsockopt(zmq.IDENTITY, my_id)
    sock.connect(f"tcp://192.168.1.67:{5555}")
    print(f"[CLIENT {my_id!r}] connected")


    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    model = NnueTronModel(10, 10)

    state_dict = torch.load(
        r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\models\mcts_v5_440.pth")
    model.load_state_dict(state_dict)
    model.reset_acc()

    model.to(device)

    # random_model = RandomTronModel()


    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = f"mcts_v6_client_{uuid.uuid4()}"

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
    # CLIENT LOOP
    ############################################

    n_games_per_loop = 64


    p_neutral_start = 0.75
    p_obstacles = 0.4
    obstacle_density_range = (0.0, 0.3)


    for i in range(1_000_000):
        # TODO: Clear da cache once model has been updated, this shouldn't be here probably
        cache.clear()

        all_game_states = []

        games_tied = p1_wins = p2_wins = 0


        if i < 500:
            mcts_iters = 100
        elif i < 1000:
            mcts_iters = 200
        else:
            mcts_iters = 300


        for _ in tqdm(range(n_games_per_loop)):

            is_neutral_start = p_neutral_start > random.random() 
            are_obstacles = p_obstacles > random.random()

            obstacle_density = random.uniform(obstacle_density_range[0], obstacle_density_range[1]) if are_obstacles else 0.0

            game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True, neutral_starts=is_neutral_start, obstacle_density=obstacle_density)

            game_status: StatusInfo = tron.get_status(game)

            curr_game_states = [deepcopy(game)]

            next_root = None
            while game_status.status == GameStatus.IN_PROGRESS:

                # NOTE: Need to reset accumulator once weights have been updated
                model.reset_acc()

                root: MCTS.Node = MCTS.search(model, game, 0, n_iterations=mcts_iters, root=next_root)

                p1_dir, p2_dir, next_root = MCTS.get_move_pair(root, 0, temp=1.0)

                # child_visits = [c.n_visits for c in root.children]
                # print(f"{child_visits=}")

                #show_game_state(game, step_through=True)

                # p1_dir = choose_direction_random(game, 0)
                # p2_dir = choose_direction_random(game, 1)


                game = tron.next(
                    game, directions=(p1_dir, p2_dir)
                )

                if next_root is not None:
                    assert game == next_root.game_state

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


        sock.send(serialized_data)

        state_dict = sock.recv_pyobj()   # will be b"ACK"
        model.load_state_dict(state_dict, strict=True)
        model.reset_acc()
        print_state_and_sos(model, decimals=3)

if __name__ == "__main__":
    main()
