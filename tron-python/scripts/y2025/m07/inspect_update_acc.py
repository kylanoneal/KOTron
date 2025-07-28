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

from tron.ai.algos import choose_direction_random


from tron.ai.minimax import minimax_alpha_beta_eval_all, cache, MinimaxContext, MinimaxResult

from tron.ai.nnue import NnueTronModel
from tron.ai.tron_model import RandomTronModel, HeroGameState

from tron.ai.training import train_loop, make_dataloader, get_weights_sum_of_squares, print_state_and_sos

from tron.io.to_proto import to_proto, from_proto


if __name__ == "__main__":

    ############################################
    # INITIALIZE MODELS
    ############################################

    device = torch.device("cpu")

    model = NnueTronModel(10, 10)

    state_dict = torch.load(
        r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2025\m07\runs\20250722-225251_nnue_v3_continuation\checkpoints\nnue_v3_continuation_382.pth"    
    )
    model.load_state_dict(state_dict)

    model.to(device)

    print("\n\n" * 8)

    # random_model = RandomTronModel()


    game = GameState.new_game(num_players=2, num_rows=10, num_cols=10, random_starts=True, neutral_starts=True)

    acc = model.init_accumulator([0,1,3,4,100,200])

    model.update_acc(acc, [200], [150, 161])

    #model.run_inference([HeroGameState(game, 0)])
