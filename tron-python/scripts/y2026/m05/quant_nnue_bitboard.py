import torch
import shutil
import random
import datetime
import warnings

from tqdm import tqdm
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


import tron
from tron.game import (
    GameState,
    PovGameState,
    GameStatus,
    StatusInfo,
    Direction,
    # from_2d_game_state,
    # from_bitboard,
    next,
)

# from tron.game_2d import GameState2D


from tron.ai.minimax import basic_minimax, MinimaxContext

from tron.ai.tron_model import TronModel, RandomTronModel
from tron.ai.nnue import NnueTronModel, QuantizedNnueTronModel

from tron.ai import MCTS
from tron.ai.MCTS import MctsContext

from tron.ai.training import (
    train_loop,
    make_dataloader,
    get_sos_info,
)
from tron.ai.benchmarks import (
    BENCHMARKS_5X5,
    TIE_BENCHMARKS_5X5,
    WIN_LOSS_BENCHMARKS_5X5,
    run_benchmark,
    run_model_benchmark,
    match,
)


@dataclass
class BenchmarkContext:
    dir_fn: callable
    description: str


@dataclass
class MatchContext:
    p1_bc: BenchmarkContext
    p2_bc: BenchmarkContext
    starting_positions: list[GameState]


def get_start_position(
    n_rows: int,
    n_cols: int,
    p_neutral: float,
    p_obstacles: float,
    obstacle_density_range: tuple,
) -> GameState:

    is_neutral_start = p_neutral > random.random()
    are_obstacles = p_obstacles > random.random()

    min_d, max_d = obstacle_density_range
    obstacle_density = random.uniform(min_d, max_d) if are_obstacles else 0.0

    return GameState.new_game(
        num_players=2,
        num_rows=n_rows,
        num_cols=n_cols,
        random_starts=True,
        neutral_starts=is_neutral_start,
        obstacle_density=obstacle_density,
    )


def benchmark(i, tb_writer, model, bench_contexts, match_contexts):

    # Model value benchmarks
    tie_benchmark_avg_score = sum(
        [run_model_benchmark(b, model) for b in TIE_BENCHMARKS_5X5]
    ) / len(TIE_BENCHMARKS_5X5)
    wl_benchmark_avg_score = sum(
        [run_model_benchmark(b, model) for b in WIN_LOSS_BENCHMARKS_5X5]
    ) / len(WIN_LOSS_BENCHMARKS_5X5)

    tb_writer.add_scalar("Avg. Model Diff (Ties)", tie_benchmark_avg_score, i)
    tb_writer.add_scalar("Avg. Model Diff (W/Ls)", wl_benchmark_avg_score, i)

    for bc in bench_contexts:
        # Tactical benchmarks
        avg_benchmark_score = sum(
            [run_benchmark(b, bc.dir_fn) for b in BENCHMARKS_5X5]
        ) / len(BENCHMARKS_5X5)

        tb_writer.add_scalar(
            f"Avg. Benchmark Score ({bc.description})", avg_benchmark_score, i
        )

    # Match score

    for mc in match_contexts:
        p1_wins, p2_wins, ties = match(
            mc.p1_bc.dir_fn, mc.p2_bc.dir_fn, mc.starting_positions
        )

        p1_match_score = (p1_wins + ties * 0.5) / (p1_wins + p2_wins + ties)

        tb_writer.add_scalar(
            f"{mc.p1_bc.description} match score vs. {mc.p2_bc.description})",
            p1_match_score,
            i,
        )


def main():

    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--mcts_iters", type=int)
    # # parser.add_argument("--temp", type=float)
    # # parser.add_argument("--explr_factor", type=float)

    # parser.add_argument("--lr", type=float)
    # parser.add_argument("--batch_size", type=int)

    # args = parser.parse_args()

    BATCH_SIZE = 4
    PRE_TRAIN_KEEP_RATE =  0.1 #0.0025
    KEEP_RATE = 0.5
    LR = 0.01 # 0.001

    ACC_DIM = 64

    # MCTS_ITERS = args.mcts_iters
    # TEMP = args.temp
    # EXPLR_FACTOR = args.explr_factor

    MCTS_ITERS = 512
    TEMP = 0.4  # 0.7
    EXPLR_FACTOR = 2.0

    QUANT_SCALE = 256

    RUN_DESCRIPTION = "quant_nnue_5x5"

    NUM_ROWS = NUM_COLS = 5

    # SIM_GAME_DEPTH = 2
    WIN_REWARD = 1.5

    GAMES_PER_ITER = 256
    CHECKPOINT_EVERY_N = 5

    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.2  # 0.4
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    TRAIN_ITERS = 100_000
    PLAY_MATCH_EVERY_N = 20
    N_MATCH_START_POSITIONS = 100

    ptkr_str = f"{PRE_TRAIN_KEEP_RATE:.4f}".replace(".", "p")
    RUN_UID = f"L{LR}_B{BATCH_SIZE}_MCITERS{MCTS_ITERS}_PTKR{ptkr_str}_ACCDIM{ACC_DIM}"

    ############################################
    # INITIALIZE MODELS
    ############################################

    # TODO: Train a model only on deep games as well?
    model = NnueTronModel(NUM_ROWS, NUM_COLS, acc_dim=ACC_DIM)

    # state_dict = torch.load(
    #     r"C:\Users\kylan\Documents\code\repos\KOTron\tron-python\scripts\y2025\m08\cnn\runs\20250810-224509_mcts_cnn_5x5\LR0.001_B4\checkpoints\LR0.001_B4_30.pth"
    # )
    # model.load_state_dict(state_dict, strict=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=LR)

    ############################################
    # TENSORBOARD AND MODEL CHECKPOINT SETUP
    ############################################

    tron_dir = Path(tron.__file__).resolve().parent.parent

    current_script_path = Path(__file__).resolve()

    outer_run_dir = current_script_path.parent / "runs"
    outer_run_dir.mkdir(exist_ok=True)

    run_dir = (
        outer_run_dir
        / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{RUN_DESCRIPTION}"
    )
    run_dir.mkdir(exist_ok=True)

    uid_run_dir = run_dir / RUN_UID
    uid_run_dir.mkdir(exist_ok=False)

    data_out_dir = uid_run_dir / "game_data"
    data_out_dir.mkdir()

    checkpoints_dir = uid_run_dir / "checkpoints"
    checkpoints_dir.mkdir()

    backup_path = uid_run_dir / f"{current_script_path.name.split('.')[0]}.bak.py"
    shutil.copy2(current_script_path, backup_path)

    tb_writer = SummaryWriter(log_dir=uid_run_dir)

    ############################################
    # BENCHMARKING SETUP
    ############################################

    def _minimax_dir_fn(
        pov_game_state: PovGameState, model: TronModel, depth: int
    ) -> Direction:

        opponent_index = 0 if pov_game_state.hero_index == 1 else 1
        mm_context = MinimaxContext(
            model.run_inference,
            pov_game_state.hero_index,
            opponent_index,
            win_magnitude=100_000.0,
        )

        mm_result = basic_minimax(
            pov_game_state.game_state,
            depth=depth,
            is_maximizing_player=True,
            context=mm_context,
        )
        return (
            mm_result.principal_variation
            if mm_result.principal_variation is not None
            else Direction.UP
        )

    # TODO: Figure out a way to seed the random tron model so output is consistent run to run
    random_model = RandomTronModel()

    # random_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    # fresh_state_dict = torch.load(tron_dir / "models" / "20250810_5x5_random_init.pth")
    # random_model.load_state_dict(fresh_state_dict, strict=True)

    random_model_benchmark_contexts = [
        random_model_d1_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=random_model, depth=1),
            description="Random Model D1 Minimax",
        ),
        random_model_d3_bc := BenchmarkContext(
            partial(_minimax_dir_fn, model=random_model, depth=3),
            description="Random Model D3 Minimax",
        ),
    ]

    # prev_model = CnnTronModel(NUM_ROWS, NUM_COLS)

    # prev_state_dict = torch.load(
    #     r"C:\Users\KylanO'Neal\Non-OneDrive Storage\code\my_repos\KOTron\tron-python\scripts\y2025\m08\cnn\runs\20250808-164733_better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate\checkpoints\better_5x5_cnn_amsgrad_d2sims_8batchsize_0p15keeprate_500.pth"
    # )
    # prev_model.load_state_dict(prev_state_dict, True)

    # prev_model_benchmark_contexts = [
    #     prev_model_d1_bc := BenchmarkContext(
    #         partial(_minimax_dir_fn, model=prev_model, depth=1),
    #         description="Prev. Model D1 Minimax",
    #     ),
    #     prev_model_d3_bc := BenchmarkContext(
    #         partial(_minimax_dir_fn, model=prev_model, depth=3),
    #         description="Prev. Model D3 Minimax",
    #     ),
    # ]

    # match_starting_positions = [
    #     get_start_position(
    #         NUM_ROWS, NUM_COLS, P_NEUTRAL_START, P_OBSTACLES, OBSTACLE_DENSITY_RANGE
    #     )
    #     for _ in range(N_MATCH_START_POSITIONS)
    # ]

    with open(
        tron_dir / "datasets" / "20260505_5x5_100starts.bin",
        "rb",
    ) as f:
        bin_match_starts = f.read()

    # TODO: Just flattening this for now
    match_starts_proto = tron.from_proto(bin_match_starts)

    match_starting_positions = []

    for g in match_starts_proto:
        assert len(g) == 1
        match_starting_positions.append(g[0])

    ############################################
    # PRE-TRAIN
    ############################################

    pre_train_iters = -1

    datasets_dir = Path(tron.__file__).resolve().parent.parent / "datasets"

    data_dir = datasets_dir / "20260505_5x5_random_depth2"

    games = []

    for i, data_file in tqdm(enumerate(data_dir.iterdir())):

        # Save the serialized data to a file.
        with open(data_file, "rb") as f:
            bin_data = f.read()

        games.extend(tron.from_proto(bin_data))

    for i in range(0, len(games), len(games) // 100):
        game = games[i]
        assert game[0].num_rows == NUM_ROWS and game[0].num_cols == NUM_COLS

    for i in range(10):

        pre_train_iters = i

        
        # TODO: Play matches and run tactics during pre-training
        warnings.warn("Should be running benchmarks during pre-training.")
        
        # benchmark(
        #     i,
        #     tb_writer,
        #     model,
        #     model_benchmark_contexts,
        #     match_contexts if i % PLAY_MATCH_EVERY_N == 0 else [],
        # )

        # # Save the serialized data to a file.
        # with open(data_file, "rb") as f:
        #     bin_data = f.read()

        # game_data = from_proto(bin_data)

        dataloader = make_dataloader(games, batch_size=BATCH_SIZE, keep_rate=PRE_TRAIN_KEEP_RATE)

        avg_loss, avg_pred_magnitude = train_loop(
            model, dataloader, optimizer, criterion, epochs=1
        )

        sos_dict, total_sos = get_sos_info(model)

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}, {total_sos=}")

        # print("\nSum of squares (weights/biases):")
        # for param, sos_val in sos_dict.items():
        #     print(f"{param:40s} {sos_val}")

        tb_writer.add_scalar("Weights Sum of Squares", total_sos, i)
        tb_writer.add_scalar("Average Loss", avg_loss, i)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, i)

        torch.save(
            model.state_dict(),
            checkpoints_dir / f"pretrain_{RUN_UID}_{i}.pth",
        )

    ############################################
    # SIM/TRAIN/BENCHMARK LOOP
    ############################################

    total_p1_wins = total_p2_wins = total_ties = 0

    for i in range(pre_train_iters + 1, TRAIN_ITERS):

        quant_model = QuantizedNnueTronModel(model, scale=QUANT_SCALE)

        p1_mcts_context = MctsContext(0, 1, WIN_REWARD, quant_model, use_acc=True)
        p2_mcts_context = MctsContext(1, 0, WIN_REWARD, quant_model, use_acc=True)

        games: list[list[GameState]] = []

        p1_wins = p2_wins = ties = 0

        for _ in tqdm(range(GAMES_PER_ITER)):

            game_state: GameState = get_start_position(
                NUM_ROWS,
                NUM_COLS,
                P_NEUTRAL_START,
                P_OBSTACLES,
                OBSTACLE_DENSITY_RANGE,
            )

            game_status: StatusInfo = tron.get_status(game_state)

            current_game: list[GameState] = [game_state]

            p1_initial_acc = quant_model.initilize_acc(PovGameState(game_state, 0, 1))
            p2_initial_acc = quant_model.initilize_acc(PovGameState(game_state, 1, 0))

            next_p1_root = next_p2_root = None
            while game_status.status == GameStatus.IN_PROGRESS:

                p1_dir, p1_root = MCTS.search(
                    p1_mcts_context,
                    game_state,
                    n_iterations=MCTS_ITERS,
                    temp=TEMP,
                    exploration_factor=EXPLR_FACTOR,
                    root=next_p1_root,
                    initial_acc=p1_initial_acc if next_p1_root is None else None,
                )

                p2_dir, p2_root = MCTS.search(
                    p2_mcts_context,
                    game_state,
                    n_iterations=MCTS_ITERS,
                    temp=TEMP,
                    exploration_factor=EXPLR_FACTOR,
                    root=next_p2_root,
                    initial_acc=p2_initial_acc if next_p2_root is None else None,
                )

                next_p1_root = MCTS.get_next_root(p1_root, p1_dir, p2_dir)
                next_p2_root = MCTS.get_next_root(p2_root, p2_dir, p1_dir)

                # show_game_state(game_state, step_through=True)

                # p1_dir = choose_direction_random(game_state, 0)
                # p2_dir = choose_direction_random(game_state, 1)

                game_state = next(game_state, directions=(p1_dir, p2_dir))

                current_game.append(game_state)

                game_status = tron.get_status(game_state)

            games.append(current_game)

            if game_status.status == GameStatus.TIE:
                # print(f"Tie")
                ties += 1
            elif game_status.winner_index == 0:
                p1_wins += 1
                # print("P1 Win")
            elif game_status.winner_index == 1:
                p2_wins += 1
                # print("P2 win")

        serialized_data = tron.to_proto(games)

        # Save the serialized data to a file.
        with open(
            data_out_dir / f"gamedata_{i}_ngames_{GAMES_PER_ITER}.bin", "wb"
        ) as f:
            f.write(serialized_data)

        print("\n" + "-" * 25 + "\n")
        print(f"{i=}, {p1_wins=}, {p2_wins=}, {ties=}")

        total_ties += ties
        total_p1_wins += p1_wins
        total_p2_wins += p2_wins

        print(f"{total_p1_wins=}, {total_p2_wins=}, {total_ties=}")

        tb_writer.add_scalar("P1 Wins / Total Wins", p1_wins / (p1_wins + p2_wins), i)
        tb_writer.add_scalar("Tie Rate", ties / GAMES_PER_ITER, i)
        tb_writer.add_scalar(
            "Average Game Length",
            sum([len(game) for game in games]) / GAMES_PER_ITER,
            i,
        )


        model_benchmark_contexts = [
            model_d1_bc := BenchmarkContext(
                partial(_minimax_dir_fn, model=quant_model, depth=1),
                description="D1 Minimax",
            ),
            model_d3_bc := BenchmarkContext(
                partial(_minimax_dir_fn, model=quant_model, depth=3),
                description="D3 Minimax",
            ),
        ]


        if i % PLAY_MATCH_EVERY_N == 0:

            match_contexts = [
                MatchContext(model_d1_bc, random_model_d1_bc, match_starting_positions),
                # MatchContext(model_d1_bc, prev_model_d1_bc, match_starting_positions),
                MatchContext(model_d3_bc, random_model_d3_bc, match_starting_positions),
                # MatchContext(model_d3_bc, prev_model_d3_bc, match_starting_positions),
            ]

            benchmark(
                i, tb_writer, quant_model, model_benchmark_contexts, match_contexts
            )

        else:
            benchmark(
                i,
                tb_writer,
                quant_model,
                model_benchmark_contexts,
                [],  # No match
            )

        dataloader = make_dataloader(games, batch_size=BATCH_SIZE, keep_rate=KEEP_RATE)

        avg_loss, avg_pred_magnitude = train_loop(
            model, dataloader, optimizer, criterion, epochs=1
        )

        print(f"{avg_loss=:.3f}, {avg_pred_magnitude=:.3f}")

        sos_dict, total_sos = get_sos_info(model)
        print("\nSum of squares (weights/biases):")
        for param, sos_val in sos_dict.items():
            print(f"{param:40s} {sos_val}")

        tb_writer.add_scalar("Weights Sum of Squares", total_sos, i)
        tb_writer.add_scalar("Average Loss", avg_loss, i)
        tb_writer.add_scalar("Average Prediction Magnitude", avg_pred_magnitude, i)

        if i % CHECKPOINT_EVERY_N == 0:
            torch.save(model.state_dict(), checkpoints_dir / f"{RUN_UID}_{i}.pth")


if __name__ == "__main__":
    main()
