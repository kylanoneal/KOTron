import itertools
import subprocess
import sys

from pathlib import Path
import tron

# MCTS_ITERS_LIST = [512, 1024]
# TEMP_LIST = [0.5, 1.0]
# EXPLR_FACTOR_LIST = [2.0, 5.0]

LRS = [0.01]
B_SIZES = [4, 16]
ACC_DIMS = [256]
FC_NEURONS = [
    (8, 16),
    (32, 8),
    (16, 16, 8),
    (32),
    (8),
]

CLAMP_VALS = [1.0, 4.0, 16.0]


tron_dir = Path(tron.__file__).resolve().parent.parent

# for mcts_iters, temp, explr in itertools.product(
#     MCTS_ITERS_LIST, TEMP_LIST, EXPLR_FACTOR_LIST
# ):
for lr, batch_size, acc_dim, neurons, clamp_val in itertools.product(
    LRS, B_SIZES, ACC_DIMS, FC_NEURONS, CLAMP_VALS
):
    cmd = [
        sys.executable,
        str(tron_dir / r"scripts\y2026\m05\3x3\3x3_perfect_data_experiments.py"),
        "--lr",
        str(lr),
        "--batch_size",
        str(batch_size),
        "--acc_dim",
        str(acc_dim),
        "--clamp_val",
        str(clamp_val),
        "--fc_neurons",
        " ".join(neurons),
        # "--mcts_iters", str(mcts_iters),
        # "--temp", str(temp),
        # "--explr_factor", str(explr)
    ]
    # print(f"Starting: MCTS_ITERS={mcts_iters}, TEMP={temp}, EXPLR_FACTOR={explr}")
    subprocess.Popen(cmd)  # run in parallel
