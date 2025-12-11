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
from functools import partial

import tron

from torch.utils.tensorboard import SummaryWriter

from tron.game import GameState, GameStatus, StatusInfo, Direction, Player

from tron.gui.utility_gui import show_game_state

from tron.ai.algos import choose_direction_random

from tron.ai.minimax import basic_minimax, MinimaxContext

from tron.ai import MCTS
from tron.ai.tron_model import CnnTronModel

from tron.io.to_proto import to_proto, from_proto

# pip install ete3
from ete3 import Tree, TreeStyle, TextFace


def visualize_tree_ete(root, out_file):

    def repr_node(node):

        return f"{("HERO" if node.is_hero else "OPPO")} {node.n_visits:<6}{(node.prev_move.name if node.prev_move is not None else "none"):<6}{round(node.total_reward, 2)}"

    # 1. Build Newick as before
    def to_newick(node):
        if not node.children:
            return repr_node(node)
        return (
            "(" + ",".join(to_newick(c) for c in node.children) + f"){repr_node(node)}"
        )

    newick = to_newick(root) + ";"

    # 2. Load the tree
    t = Tree(newick, format=1)

    # 3. For every node, stick a TextFace on the branch showing node.name (visits)
    for n in t.traverse():
        face = TextFace(str(n.name), fsize=10)  # adjust fsize if you like
        n.add_face(face, column=0, position="branch-right")

    # 4. Turn off the default leaf‐only labels
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False

    # 5. Display
    #t.show(tree_style=ts)
    t.render(str(out_file))


def main():

    
    current_script_path = Path(__file__).resolve()

    outer_run_dir = current_script_path.parent / "tree_viz"
    outer_run_dir.mkdir(exist_ok=True)

    v=1

    while (outer_run_dir / f"v{v}").exists():
        v+=1
    
    run_dir = outer_run_dir / f"v{v}"
    run_dir.mkdir()

    NUM_ROWS = NUM_COLS = 5

    GAMES_PER_ITER = 64
    CHECKPOINT_EVERY = 10
    PLAY_MATCH_EVERY = 10

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    device = torch.device("cpu")

    init_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    opinionated_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    state_dict = torch.load(r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\scripts\y2025\m08\cnn\runs\20250808-164733_better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate\checkpoints\better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate_500.pth")
    opinionated_model.load_state_dict(state_dict, True)


    players = (Player(0,0, True), Player(4,4,True))
    game = GameState.from_players(players, 5, 5)

    game_status: StatusInfo = tron.get_status(game)

    curr_game_states = [deepcopy(game)]

    next_root = None
    while game_status.status == GameStatus.IN_PROGRESS:

        init_root: MCTS.Node = MCTS.search(
            init_model, game, 0, n_iterations=100,
        )

        opin_root = MCTS.search(opinionated_model, game, 0, 1_000)

        p1_dir, p2_dir, _ = MCTS.get_move_pair(opin_root, 0, temp=0.0)

        visualize_tree_ete(opin_root, run_dir / f"opin_{len(curr_game_states)}.png")
        visualize_tree_ete(init_root, run_dir / f"init_{len(curr_game_states)}.png")

        # child_visits = [c.n_visits for c in root.children]
        # print(f"{child_visits=}")

        p2_dir = show_game_state(game, step_through=True)

        # p1_dir = choose_direction_random(game, 0)
        # p2_dir = choose_direction_random(game, 1)

        game = tron.next(game, directions=(p1_dir, p2_dir))

        if next_root is not None:
            assert game == next_root.game_state

        curr_game_states.append(game)

        game_status = tron.get_status(game)
    
    print(f"{game_status=}")


if __name__ == "__main__":
    main()
