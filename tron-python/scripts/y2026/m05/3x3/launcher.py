import itertools
import subprocess
import sys

# MCTS_ITERS_LIST = [512, 1024]
# TEMP_LIST = [0.5, 1.0]
# EXPLR_FACTOR_LIST = [2.0, 5.0]

LRS = [0.01, 0.001, 0.0001]
B_SIZES = [4, 256]
ACC_DIMS = [256, 512, 1024]

# for mcts_iters, temp, explr in itertools.product(
#     MCTS_ITERS_LIST, TEMP_LIST, EXPLR_FACTOR_LIST
# ):
for lr, batch_size, acc_dim in itertools.product(
    LRS, B_SIZES, ACC_DIMS
):
    cmd = [
        sys.executable, r"C:\Users\kylan\code\KOTron\tron-python\scripts\y2026\m05\3x3\3x3_perfect_data_experiments.py",
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--acc_dim", str
        # "--mcts_iters", str(mcts_iters),
        # "--temp", str(temp),
        # "--explr_factor", str(explr)
    ]
    # print(f"Starting: MCTS_ITERS={mcts_iters}, TEMP={temp}, EXPLR_FACTOR={explr}")
    subprocess.Popen(cmd)  # run in parallel
