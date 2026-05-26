from typing import Callable, Union, Optional
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
    PovGameState,
    from_2d_game_state,
    from_bitboard,
)

from tron.enums import PovGameResult

from tron.game_2d import GameState2D, Player2D

from tron.ai.tron_model import TronModel
from tron.ai.training import get_label_magnitude, ModelExample, LabeledExample


@dataclass(frozen=True)
class ValueBenchmark:

    pov_game_state: PovGameState
    steps_until_terminal: int
    hero_expected_result: PovGameResult


@dataclass(frozen=True)
class Tactic:

    pov_game_state: PovGameState
    opposing_dirs: list[Direction]
    expected_hero_dirs: Optional[list[Direction]] = None


    def __post_init__(self):
        assert len(self.opposing_dirs) > 0


    @staticmethod
    def transform(bench: "Tactic", do_lr_flip: bool, n_rot_90: int) -> "Tactic":

        game_2d: GameState2D = from_bitboard(bench.pov_game_state.game_state)

        t_game_state = from_2d_game_state(
            GameState2D.transform(game_2d, do_lr_flip, n_rot_90)
        )

        t_pov_game_state = PovGameState(
            t_game_state,
            hero_index=bench.pov_game_state.hero_index,
            opponent_index=bench.pov_game_state.opponent_index,
        )

        t_opposing_dirs = Direction.transform(bench.opposing_dirs, do_lr_flip, n_rot_90)

        if bench.expected_hero_dirs is not None:
            t_expected_hero_dirs = Direction.transform(
                bench.expected_hero_dirs, do_lr_flip, n_rot_90
            )
        else:
            t_expected_hero_dirs = None

        return Tactic(t_pov_game_state, t_opposing_dirs, t_expected_hero_dirs)


@dataclass(frozen=True)
class TacticResult:

    tactic: Tactic
    pov_game_states: list[PovGameState]
    actual_hero_dirs: list[Direction]
    correct_moves: int


@dataclass(frozen=True)
class ValueBenchmarkResult:

    benchmark: ValueBenchmark
    expected_value: float
    predicted_value: float


# TODO: Add "null" p1 moves - test bot's "will to live"

SPATIAL_TACTICS_5X5 = (
    Tactic(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(0, 0, True), Player2D(0, 4, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        opposing_dirs=([Direction.DOWN] * 4) + [Direction.LEFT] + ([Direction.UP] * 4),
    ),
    Tactic(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(1, 2, True), Player2D(3, 0, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        opposing_dirs=([Direction.RIGHT] * 4)
        + [Direction.DOWN]
        + ([Direction.LEFT] * 4),
    ),
)

DECISIVE_TACTICS_5X5 = (
    Tactic(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(1, 1, True), Player2D(2, 4, True)),
                ),
            ),
            hero_index=0,
            opponent_index=1,
        ),
        opposing_dirs=[Direction.DOWN],
        expected_hero_dirs=[Direction.LEFT],
    ),
)

TIES_5X5 = (
    ValueBenchmark(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(0, 0, True), Player2D(4, 2, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        steps_until_terminal=4,
        hero_expected_result=PovGameResult.TIE,
    ),
    ValueBenchmark(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(3, 0, True), Player2D(3, 4, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        steps_until_terminal=2,
        hero_expected_result=PovGameResult.TIE,
    ),
)

# TODO: Add perspective switches
DECISIVE_5X5 = (
    ValueBenchmark(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(4, 3, True), Player2D(2, 2, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        steps_until_terminal=1,
        hero_expected_result=PovGameResult.WINNER,
    ),
    ValueBenchmark(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(2, 2, True), Player2D(4, 3, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        steps_until_terminal=1,
        hero_expected_result=PovGameResult.LOSER,
    ),
    ValueBenchmark(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(2, 0, True), Player2D(0, 3, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        steps_until_terminal=2,
        hero_expected_result=PovGameResult.WINNER,
    ),
    ValueBenchmark(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(
                GameState2D(
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
                    players=(Player2D(0, 3, True), Player2D(2, 0, True)),
                )
            ),
            hero_index=0,
            opponent_index=1,
        ),
        steps_until_terminal=2,
        hero_expected_result=PovGameResult.LOSER,
    ),
)


def run_tactic(
    tactic: Tactic,
    dir_fn: Callable[[PovGameState], Direction],
    run_symmetries: bool = True,
) -> list[TacticResult]:

    if run_symmetries:
        tactics = []

        for do_lr_flip, n_rot_90 in itertools.product([True, False], range(4)):
            tactics.append(Tactic.transform(tactic, do_lr_flip, n_rot_90))

    else:
        tactics = [tactic]

    total_score = 0.0

    results: list[TacticResult] = []

    for t in tactics:
        pov_game_state, opposing_dirs, expected_hero_dirs = (
            t.pov_game_state,
            t.opposing_dirs,
            t.expected_hero_dirs,
        )

        actual_pov_game_states = [pov_game_state]
        actual_hero_dirs = []

        # Score is based on how far through the opposing dirs we got
        # i will hold the number of correct moves made
        for i in range(len(opposing_dirs)):

            hero_dir = dir_fn(pov_game_state)

            actual_hero_dirs.append(hero_dir)

            directions = [None, None]
            directions[pov_game_state.hero_index] = hero_dir
            directions[pov_game_state.opponent_index] = opposing_dirs[i]

            pov_game_state = PovGameState(
                next(pov_game_state.game_state, directions),
                pov_game_state.hero_index,
                pov_game_state.opponent_index,
            )

            actual_pov_game_states.append(pov_game_state)

            if expected_hero_dirs is not None:

                if hero_dir != expected_hero_dirs[i]:
                    break

            status_info = get_status(pov_game_state.game_state)

            if status_info.status != GameStatus.IN_PROGRESS:
                assert (
                    expected_hero_dirs is None
                ), "Tactics with expected hero dirs should not reach terminal state"

                assert status_info.winner_index != pov_game_state.hero_index
                assert status_info.status != GameStatus.TIE
                break
        else:
            i += 1

        results.append(TacticResult(t, actual_pov_game_states, actual_hero_dirs, i))

    return results


def run_value_benchmark(
    bench: ValueBenchmark, model: TronModel, run_symmetries=True
) -> list[ModelExample]:

    if run_symmetries:
        value_benchmarks = []

        for do_lr_flip, n_rot_90 in itertools.product([True, False], range(4)):

            value_benchmarks.append(
                ValueBenchmark(
                    PovGameState(
                        GameState.transform(
                            bench.pov_game_state.game_state, do_lr_flip, n_rot_90
                        ),
                        bench.pov_game_state.hero_index,
                        bench.pov_game_state.opponent_index,
                    ),
                    steps_until_terminal=bench.steps_until_terminal,
                    hero_expected_result=bench.hero_expected_result,
                ),
            )

    else:
        value_benchmarks = [bench]

    results = []
    for vb in value_benchmarks:

        if vb.hero_expected_result == PovGameResult.TIE:
            label = 0.0
        else:

            label = get_label_magnitude(vb.steps_until_terminal)

            if vb.hero_expected_result == PovGameResult.LOSER:
                label *= -1

        labeled_example = LabeledExample(vb.pov_game_state, label=label)

        results.append(
            ModelExample(labeled_example, model.run_inference(vb.pov_game_state))
        )

    return results


# TODO: Test by asserting match score is same with p1/p2 dir fn args switched
def match(p1_dir_fn: callable, p2_dir_fn: callable, starting_positions=list[GameState]):

    p1_wins = p2_wins = ties = 0

    print(f"Playing match...")
    for i in tqdm(range(len(starting_positions))):

        white_pos = starting_positions[i]

        if len(white_pos.players) > 2:
            raise NotImplementedError()

        black_players = (white_pos.players[1], white_pos.players[0])

        black_pos = GameState(
            white_pos.num_rows, white_pos.num_cols, white_pos.board, black_players
        )

        for start_game_state in [white_pos, black_pos]:

            game_state = start_game_state

            status_info = get_status(game_state)

            while status_info.status == GameStatus.IN_PROGRESS:

                p1_dir = p1_dir_fn(
                    PovGameState(game_state, hero_index=0, opponent_index=1)
                )
                p2_dir = p2_dir_fn(
                    PovGameState(game_state, hero_index=1, opponent_index=0)
                )

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
