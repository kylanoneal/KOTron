import zmq
import json
import torch
import shutil
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


from game.tron import GameState, GameStatus, StatusInfo, Direction

# from tron.ai.algos import (
#     #choose_direction_model_naive,
#     # choose_direction_random,
#     #choose_direction_minimax,
# )

from tron.ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult
from tron.ai.model_architectures import FastNet, EvaluationNetConv3OneStride, LeakyReLU
from tron.ai.tron_model import StandardTronModel

from tron.ai.training import train_loop, make_dataloader, get_weights_sum_of_squares

from tron_io.to_proto import to_proto, from_proto


def main():

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    # state_dict = torch.load(
    #     "C:/Users/kylan/Documents/code/repos/KOTron/python/scripts/y2024/2024_12_15_alpha_beta/leaky_relu_continuation_v4_122.pth"
    # )
    torch_model = LeakyReLU(grid_dim=10)
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
    # Create a ZeroMQ context and a REP socket.
    context = zmq.Context()
    socket = context.socket(zmq.REP)

    ############################################
    # MODEL CHECKPOINT SETUP
    ############################################

    run_uid = "ipc_test_v2"

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


    ############################################
    # SIMULATION-TRAIN LOOP
    ############################################
    
    # Bind the REP socket to a TCP address (this must match what the Rust side connects to).
    socket.bind("tcp://localhost:5555")
    print("Python REP server running on tcp://localhost:5555")

    i = 0
    while True:
        i += 1
        # Wait for the next request (this call blocks).
        print("Awaiting game data")
        message = socket.recv()
        print("Received message of length:", len(message))
        
        # Optionally, decode the protobuf message.
        game_data = from_proto(message)

        
        dataloader = make_dataloader(
            game_data, model, batch_size=batch_size, shuffle=shuffle
        )

        avg_loss, avg_pred_magnitude = train_loop(model.model, dataloader, optimizer, criterion, device, epochs=1)


        torch.save(model.model.state_dict(), checkpoints_folder / f"{run_uid}_{i}.pth")
        # Process the message as needed...


        torch_input = torch.randn(1, 3, 10, 10).to(device)

        onnx_checkpoint_file = checkpoints_folder / f"{run_uid}_{i}.onnx"

        torch.onnx.export(
            model.model,
            torch_input,
            onnx_checkpoint_file,
            opset_version=18,  # or a compatible opset version
            input_names=["input"],
            output_names=["output"]
        )

        with open(onnx_checkpoint_file, "rb") as f:
            onnx_model_bytes = f.read()
                
        socket.send(onnx_model_bytes)

        print("Sent onnx model bytes")

if __name__ == "__main__":
    main()
