import zmq

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



def main(bind_addr="tcp://*:5555"):


    ############################################
    # ZMQ SETUP
    ############################################

    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(bind_addr)
    print(f"[SERVER] ROUTER bound to {bind_addr}")

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    model = NnueTronModel(10, 10)

    state_dict = torch.load(
        r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\models\20250801_mcts_v6_4590.pth")
    model.load_state_dict(state_dict)
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

    run_uid = "mcts_v7_server"

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
    # SERVER LOOP
    ############################################

    total_games_tied = total_p1_wins = total_p2_wins = 0

    train_iter = 0
    checkpoint_every = 10

    while True:

        model.reset_acc()

        games_tied = p1_wins = p2_wins = 0
        # ROUTER.recv_multipart gives [client_id, empty, data]
        client_id, data = sock.recv_multipart()
        print(f"[SERVER] ← {len(data)} bytes from {client_id!r}")

        # Save the serialized data to a file.
        with open(data_out_folder / f"game_data_iter_{train_iter}.bin", "wb") as f:
            f.write(data)

        game_data = from_proto(data)

        for game in game_data:
            terminal_state = game[-1]

            status_info = tron.get_status(terminal_state)

            if status_info.status == GameStatus.TIE:
                #print(f"Tie")
                games_tied += 1
            elif status_info.winner_index == 0:
                p1_wins += 1
                #print("P1 Win")
            elif status_info.winner_index == 1:
                p2_wins += 1
                #print("P2 win")
            else:
                raise RuntimeError("Non terminal state")
        print(f"This iter P1 wins: {p1_wins}, p2 wins: {p2_wins}, ties: {games_tied}")

        total_games_tied += games_tied
        total_p1_wins += p1_wins
        total_p2_wins += p2_wins


        print(f"Running total P1 wins: {total_p1_wins}, p2 wins: {total_p2_wins}, ties: {total_games_tied}")

        tb_writer.add_scalar("Player 1 Winrate", p1_wins / len(game_data), train_iter)
        tb_writer.add_scalar("Tie Rate", games_tied / len(game_data), train_iter)
        tb_writer.add_scalar(
            "Average Game Length",
            sum([len(game) for game in game_data])
            / len(game_data),
            train_iter,
        )

        print(f"Training time! Iter: {train_iter}")

        dataloader = make_dataloader(
            game_data, batch_size=batch_size
        )

        avg_loss, avg_pred_magnitude = train_loop(model, dataloader, optimizer, criterion, device, epochs=1)
        weights_sos = get_weights_sum_of_squares(model)

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {weights_sos=:.3f}")
        print_state_and_sos(model, decimals=3)

        tb_writer.add_scalar("Sum of Squares of Weights", weights_sos, train_iter)
        tb_writer.add_scalar("Average Loss", avg_loss, train_iter)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, train_iter)

        if train_iter % checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoints_folder / f"{run_uid}_{train_iter}.pth")

        # …process data…
        # send reply back only to that client
        sock.send(client_id, flags=zmq.SNDMORE)  
        sock.send_pyobj(model.state_dict())     

        train_iter += 1




if __name__ == "__main__":
    main()
