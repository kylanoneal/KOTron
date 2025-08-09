from typing import Union, Optional
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import itertools

from tron.game import (
    GameState,
    Player,
    Direction,
    GameStatus,
    get_status,
    next,
    get_possible_directions,
)

from tron.ai.tron_model import TronModel, PovGameState


@dataclass(frozen=True)
class ModelBenchmark:

    pov_game_state: PovGameState
    expected_value: float


@dataclass(frozen=True)
class Benchmark:

    pov_game_state: PovGameState
    opposing_dirs: list[Direction]
    expected_hero_dirs: Optional[list[Direction]] = None

    @staticmethod
    def transform(bench: "Benchmark", do_lr_flip: bool, n_rot_90: int) -> "Benchmark":

        t_game_state = GameState.transform(
            bench.pov_game_state.game_state, do_lr_flip, n_rot_90
        )
        t_pov_game_state = PovGameState(
            t_game_state, hero_index=bench.pov_game_state.hero_index
        )

        t_opposing_dirs = Direction.transform(bench.opposing_dirs, do_lr_flip, n_rot_90)

        if bench.expected_hero_dirs is not None:
            t_expected_hero_dirs = Direction.transform(
                bench.expected_hero_dirs, do_lr_flip, n_rot_90
            )
        else:
            t_expected_hero_dirs = None

        return Benchmark(t_pov_game_state, t_opposing_dirs, t_expected_hero_dirs)


# TODO: Add "null" p1 moves - test bot's "will to live"
# TODO: Run each benchmark 8 times? with the 8 possible symmetries
BENCHMARKS_5X5 = [
    Benchmark(
        pov_game_state=PovGameState(
            game_state=GameState(
                grid=np.array(
                    [
                        [1, 0, 1, 0, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                    ],
                    dtype=bool,
                ),
                players=(Player(0, 0, True), Player(0, 4, True)),
            ),
            hero_index=0,
        ),
        opposing_dirs=([Direction.DOWN] * 4) + [Direction.LEFT] + ([Direction.UP] * 4),
    ),
    Benchmark(
        pov_game_state=PovGameState(
            game_state=GameState(
                grid=np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                players=(Player(1, 2, True), Player(3, 0, True)),
            ),
            hero_index=0,
        ),
        opposing_dirs=([Direction.RIGHT] * 4)
        + [Direction.DOWN]
        + ([Direction.LEFT] * 4),
    ),
    Benchmark(
        pov_game_state=PovGameState(
            game_state=GameState(
                grid=np.array(
                    [
                        [0, 0, 0, 0, 1],
                        [0, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 0],
                        [1, 0, 1, 0, 0],
                    ],
                    dtype=bool,
                ),
                players=(Player(1, 1, True), Player(2, 4, True)),
            ),
            hero_index=0,
        ),
        opposing_dirs=[Direction.DOWN],
        expected_hero_dirs=[Direction.LEFT],
    ),
]

TIE_BENCHMARKS_5X5 = [
    ModelBenchmark(
        pov_game_state=PovGameState(
            GameState(
                grid=np.array(
                    [
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                    ],
                    dtype=bool,
                ),
                players=(Player(0, 0, True), Player(4, 2, True)),
            ),
            hero_index=0,
        ),
        expected_value=0.0,
    ),
    ModelBenchmark(
        pov_game_state=PovGameState(
            GameState(
                grid=np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1],
                    ],
                    dtype=bool,
                ),
                players=(Player(3, 0, True), Player(3, 4, True)),
            ),
            hero_index=0,
        ),
        expected_value=0.0,
    ),
]

# TODO: Add perspective switches
WIN_LOSS_BENCHMARKS_5X5 = [
    ModelBenchmark(
        pov_game_state=PovGameState(
            GameState(
                grid=np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0],
                    ],
                    dtype=bool,
                ),
                players=(Player(4, 3, True), Player(2, 2, True)),
            ),
            hero_index=0,
        ),
        expected_value=1.0,
    ),
    ModelBenchmark(
        pov_game_state=PovGameState(
            GameState(
                grid=np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0],
                    ],
                    dtype=bool,
                ),
                players=(Player(2, 2, True), Player(4, 3, True)),
            ),
            hero_index=0,
        ),
        expected_value=-1.0,
    ),
    ModelBenchmark(
        pov_game_state=PovGameState(
            GameState(
                grid=np.array(
                    [
                        [1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [0, 1, 0, 1, 1],
                        [0, 1, 0, 1, 1],
                    ],
                    dtype=bool,
                ),
                players=(Player(2, 0, True), Player(0, 3, True)),
            ),
            hero_index=0,
        ),
        expected_value=1.0,
    ),
    ModelBenchmark(
        pov_game_state=PovGameState(
            GameState(
                grid=np.array(
                    [
                        [1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [0, 1, 0, 1, 1],
                        [0, 1, 0, 1, 1],
                    ],
                    dtype=bool,
                ),
                players=(Player(0, 3, True), Player(2, 0, True)),
            ),
            hero_index=0,
        ),
        expected_value=-1.0,
    ),
]


def run_benchmark(
    bench: Benchmark, dir_fn: callable, run_symmetries: bool = True
) -> float:

    if run_symmetries:
        benchmarks = []

        for do_lr_flip, n_rot_90 in itertools.product([True, False], range(4)):
            benchmarks.append(Benchmark.transform(bench, do_lr_flip, n_rot_90))

    else:
        benchmarks = [bench]

    total_score = 0.0

    for b in benchmarks:
        pov_game_state, opposing_dirs, expected_hero_dirs = (
            b.pov_game_state,
            b.opposing_dirs,
            b.expected_hero_dirs,
        )

        opponent_index = 0 if pov_game_state.hero_index == 1 else 1

        is_tactic = expected_hero_dirs is not None

        if is_tactic:
            assert len(opposing_dirs) == len(expected_hero_dirs)

        # Score is based on how far through the opposing dirs we got
        for i in range(len(opposing_dirs)):

            hero_dir = dir_fn(pov_game_state)

            if is_tactic:

                if hero_dir != expected_hero_dirs[i]:
                    break

            directions = [None, None]
            directions[pov_game_state.hero_index] = hero_dir
            directions[opponent_index] = opposing_dirs[i]

            pov_game_state = PovGameState(next(pov_game_state.game_state, directions), pov_game_state.hero_index)

            status_info = get_status(pov_game_state.game_state)

            if status_info.status != GameStatus.IN_PROGRESS:
                assert not is_tactic

                assert status_info.winner_index != pov_game_state.hero_index
                assert status_info.status != GameStatus.TIE
                break
        else:
            i += 1

        # print(f"Made it through {i} moves of {len(opposing_dirs)} move benchmark.")
        total_score += i / len(opposing_dirs)

    return total_score / len(benchmarks)


def run_model_benchmark(bench: ModelBenchmark, model: TronModel, run_symmetries=True):

    if run_symmetries:
        pov_game_states = []

        for do_lr_flip, n_rot_90 in itertools.product([True, False], range(4)):
            pov_game_states.append(
                PovGameState(
                    GameState.transform(
                        bench.pov_game_state.game_state, do_lr_flip, n_rot_90
                    ),
                    bench.pov_game_state.hero_index,
                )
            )

    else:
        pov_game_states = [bench.pov_game_state]

    total_diff = 0.0
    for pov_game_state in pov_game_states:
        eval = model.run_inference(pov_game_state)

        # print(f"{bench.expected_value=}, actual eval={round(eval, 3)}")
        total_diff += abs(bench.expected_value - eval)

    return total_diff / len(pov_game_states)


# TODO: Test by asserting match score is same with p1/p2 dir fn args switched
def match(p1_dir_fn: callable, p2_dir_fn: callable, starting_positions=list[GameState]):

    p1_wins = p2_wins = ties = 0

    print(f"Playing match...")
    for i in tqdm(range(len(starting_positions))):

        white_pos = starting_positions[i]

        black_players = (white_pos.players[1], white_pos.players[0])
        black_grid = white_pos.grid.copy()

        black_pos = GameState(black_grid, black_players)

        for start_game_state in [white_pos, black_pos]:

            game_state = start_game_state

            status_info = get_status(game_state)

            while status_info.status == GameStatus.IN_PROGRESS:

                p1_dir = p1_dir_fn(PovGameState(game_state, hero_index=0))
                p2_dir = p2_dir_fn(PovGameState(game_state, hero_index=1))

                game_state = next(game_state, directions=(p1_dir, p2_dir))

                status_info = get_status(game_state)

            if status_info.status == GameStatus.WINNER:
                if status_info.winner_index == 0:
                    p1_wins += 1
                else:
                    p2_wins += 1
            elif status_info.status == GameStatus.TIE:
                ties += 1

    return p1_wins, p2_wins, ties
