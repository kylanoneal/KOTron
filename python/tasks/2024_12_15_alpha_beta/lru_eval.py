from functools import lru_cache
import json
import torch
import shutil
import datetime
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from game.tron import Tron, GameStatus
from ai.algos import (
    choose_direction_model_naive,
    choose_direction_random,
    choose_direction_minimax,
)
from ai.model_architectures import FastNet, EvaluationNetConv3OneStride
from ai.tron_model import StandardTronModel, TronModelAbstract

from ai.training import train_loop, make_dataloader, get_weights_sum_of_squares



from cachetools import LRUCache, cached

# Initialize an LRU cache with a maximum size
cache = LRUCache(maxsize=100000)



# Define an LRU-cached function
@cached(cache)
def lru_eval(model: TronModelAbstract, game, player_index):
    return model.run_inference([game], player_index)[0]



if __name__=="__main__":
    device = torch.device("cpu")

    state_dict = torch.load(
        "C:/Users/kylan/Documents/code/repos/KOTron/python/tasks/2024_12_15_alpha_beta/oldnet_self_train_continuation_v5_8.pth"
    )


    torch_model = EvaluationNetConv3OneStride(grid_dim=10)
    torch_model.load_state_dict(state_dict)
    torch_model = torch_model.to(device)

    model = StandardTronModel(torch_model, device)

    game = Tron(num_players=2, num_rows=10, num_cols=10, random_starts=False)

    print(lru_eval(model, game, 0))
    print(lru_eval(model, game, 0))


    arg_tup = (model, game, 0)

    print(f"Arg tup in caache? {arg_tup in cache}")

