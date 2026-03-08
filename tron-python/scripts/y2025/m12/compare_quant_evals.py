import zmq
import uuid
import json
import torch
import shutil
import random
import datetime
import argparse
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from cachetools import LRUCache, cached

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction

# from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random

from tron.ai.minimax import basic_minimax, MinimaxContext


from tron.ai.quant_nnue import NnueTronModel, TronModel
from tron.ai import MCTS


class QuantizedTronModel(TronModel):

    def __init__(self, model: NnueTronModel, scale = 256):
        self.raw_model = model

        self.scale = scale

        self.embed_weights = torch.round(model.embedding.weight * scale).to(torch.int32)

        self.linear_weights = torch.round(model.fc1.weight * scale).to(torch.int32)
        self.linear_bias = torch.round(model.fc1.bias * scale * scale).to(torch.int32)


    def run_inference(self, pov_game_state: PovGameState) -> float:

        indices = self.raw_model.get_active_indices(pov_game_state)
        # 1. Sum embeddings (int accumulator)
        acc = self.embed_weights[indices].sum(dim=0)  # [acc_dim], int32
        #print(f"After sum: {acc.sum().item() / 1024=}")

        # 2. Clamp to [0, scale]
        acc = torch.clamp(acc, 0, self.scale)
        #print(f"After clamp: {acc.sum().item()/ 1024=}")

        acc = acc.to(dtype=torch.int32)

        #print(f"After int32 cast: {acc.sum().item()/ 1024=}")

        # 3. Linear layer in integer domain
        #    (1 x acc_dim) @ (acc_dim) -> scalar
        y_int = (self.linear_weights @ acc) + self.linear_bias  # still int32

        #print(f"After linear: {y_int.sum().item()/ 1024 / 1024=}")

        # 4. Rescale back to float
        y = y_int.float() / (self.scale * self.scale)

        return y.item()


def quantize_nnue(model: NnueTronModel) -> callable:

    scale = 1024  # example fixed-point scale
    # embedding
    W_emb = model.embedding.weight  # shape: [num_features, acc_dim]

    # linear layers
    W1 = model.fc1.weight  # [hidden_dim, acc_dim]
    b1 = model.fc1.bias  # [hidden_dim]

    W_emb_i = torch.round(W_emb * scale).to(torch.int32)

    W1_i = torch.round(W1 * scale).to(torch.int32)
    b1_i = torch.round(b1 * scale * scale).to(torch.int32)

    print(f"{W1.median().item()=}")

    print(f"{b1}")

    print(f"{W1.sum().item()=}, {b1.sum().item()=}, {W1_i.sum().item() / 1024 =}, {b1_i.sum().item() / 1024 =}")

    def run_inf(indices):


        ######################################

        # # 1. Sum embeddings (int accumulator)
        # acc = W_emb[indices].sum(dim=0)  # [acc_dim], int32
        # print(f"\n\nReg After sum: {acc.sum().item()=}")

        # # 2. Clamp to [0, 1]
        # acc = torch.clamp(acc, 0, 1)
        # print(f"Reg After clamp: {acc.sum().item()=}")

        # y = (W1 @ acc) + b1

        # print(f"Reg After linear: {y.sum().item()=}")

        ######################################

        # 1. Sum embeddings (int accumulator)
        acc = W_emb_i[indices].sum(dim=0)  # [acc_dim], int32
        #print(f"After sum: {acc.sum().item() / 1024=}")

        # 2. Clamp to [0, scale]
        acc = torch.clamp(acc, 0, scale)
        #print(f"After clamp: {acc.sum().item()/ 1024=}")

        acc = acc.to(dtype=torch.int32)

        #print(f"After int32 cast: {acc.sum().item()/ 1024=}")

        # 3. Linear layer in integer domain
        #    (1 x acc_dim) @ (acc_dim) -> scalar
        y_int = (W1_i @ acc) + b1_i  # still int32

        #print(f"After linear: {y_int.sum().item()/ 1024 / 1024=}")

        # 4. Rescale back to float
        y = y_int.float() / (scale * scale)

        return y.item()

    return run_inf


def main():
    state_dict = torch.load(
        r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2025\m12\runs\20260205-150303_mcts_nnue_5x5\FINETUNELR0.001_B4\checkpoints\FINETUNELR0.001_B4_50.pth"
    )

    # TODO: Train a model only on deep games as well?
    model = NnueTronModel(5, 5)
    model.load_state_dict(state_dict, strict=True)    

    model2 = NnueTronModel(5, 5)
    model2.load_state_dict(state_dict, strict=True)

    inf_function = quantize_nnue(model)

    sum_diffs1 = sum_diffs2=0

    n_iters = 1000
    for i in range(n_iters):

        if i % 100 == 0:
            model.reset_acc()

        pov_game_state = PovGameState(
            GameState.new_game(2, 5, 5, True, obstacle_density=0.7), 0
        )


        reg_eval = model.run_inference(pov_game_state)

        quant_eval = inf_function(model.get_active_indices(pov_game_state))

        diff1 = abs(reg_eval - quant_eval)

        print(
            f"{'Reg: ' + str(round(reg_eval, 4)):<20}"
            f"{'Quant: ' + str(round(quant_eval, 4)):<20}"
            f"{'Diff1: ' + str(round(diff1, 4)):<20}"

        )

        sum_diffs1 += diff1

    print(f"Avg diff1: {sum_diffs1 / n_iters}")



if __name__ == "__main__":
    main()
