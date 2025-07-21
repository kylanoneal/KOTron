import json
import torch
import shutil
import random
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


from game.tron import GameState, GameStatus, StatusInfo, Direction
from game.utility_gui import show_game_state

# from tron.ai.algos import (
#     #choose_direction_model_naive,
#     # choose_direction_random,
#     #choose_direction_minimax,
# )

from tron.ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult
from tron.ai.model_architectures import FastNet, EvaluationNetConv3OneStride, LeakyReLU, TransformerGameEvaluator, EmbeddingTransformerGameEvaluator
from tron.ai.tron_model import CnnTronModel, EmbeddingTransformerTronModel, OneHotTransformerTronModel

from tron.ai.training import train_loop, make_dataloader, get_weights_sum_of_squares
from tron.ai.chat_gpt_transformers import TransformerImageRegressor


from tron_io.to_proto import to_proto, from_proto


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cuda:0")


    torch_model = TransformerImageRegressor(image_height=10, image_width=10)
    torch_model = torch_model.to(device)

    model = EmbeddingTransformerTronModel(torch_model, device)


    #     state_dict = torch.load(
    #     "C:/Users/kylan/Documents/code/repos/KOTron/python/scripts/y2025/m05/experiments/runs/20250524-004611_obstacles_v2/checkpoints/obstacles_v2_155.pth"
    # )
    # torch_model = LeakyReLU(grid_dim=10)
    # torch_model.to(device)
    # #torch_model.load_state_dict(state_dict)
    # model = CnnTronModel(torch_model, device)

    ############################################
    # TRAINING SETUP / HYPERPARAMETERS
    ############################################

    batch_size = 32
    shuffle = True

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.0001)

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    run_uid = "pretrain_transformer"

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

    data_dir = Path(r"C:\Users\kylan\Documents\code\repos\KOTron\python\scripts\y2025\m06\make_random_data\runs\20250604-220539_obstacles_v2\game_data")

    for i, data_file in enumerate(data_dir.iterdir()):
        
        # Save the serialized data to a file.
        with open(data_file, "rb") as f:
            bin_data = f.read()

        game_data = from_proto(bin_data)

        dataloader = make_dataloader(
            game_data, model, batch_size=batch_size, shuffle=shuffle, include_ties=True
        )

        avg_loss, avg_pred_magnitude = train_loop(model.model, dataloader, optimizer, criterion, device, epochs=1)
        

        print(f"\nData file: {data_file.name}")
        print(f"SoS: {get_weights_sum_of_squares(model.model)}")
        print(f"Avg loss: {avg_loss}")
        print(f"Avg pred magnitude: {avg_pred_magnitude}\n")
        print('-' * 12)

        tb_writer.add_scalar("Sum of Squares of Weights", get_weights_sum_of_squares(model.model), i)
        tb_writer.add_scalar("Average Loss", avg_loss, i)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, i)

        #     # To see them one by one:
        # for token_idx in range(4):
        #     vec = model.model.embedding.weight[token_idx]    # shape: [embed_dim]
        #     print(f"Embedding for index {token_idx}:", vec)

        if i % 10 == 0:
            torch.save(model.model.state_dict(), checkpoints_folder / f"{run_uid}_{i}.pth")
