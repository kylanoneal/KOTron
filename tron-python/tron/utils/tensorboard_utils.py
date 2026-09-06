import random

from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter


import tron

from tron.game import (
    GameState,
    GameStatus,
    PovGameState,
    GameStatus,
    Direction,
    next,
    get_possible_directions,
)


from tron.ai.tron_model import TronModel

from tron.ai.training import (
    TrainingResult,
    get_sos_info,
)


from tron.ai.benchmarks import (
    Tactic,
    # SPATIAL_TACTICS_5X5,
    # DECISIVE_TACTICS_5X5,
    # TIES_5X5,
    # DECISIVE_5X5,
    match,
    run_tactic,
    run_value_benchmark,
)

from tron.utils.viz_utils import (
    render_model_example_image,
    render_tactic_benchmark_image,
)


@dataclass
class ValueBenchmarkContext:
    model: TronModel
    description: str


@dataclass
class TacticalBenchmarkContext:
    dir_fn: Callable[[PovGameState], Direction]
    description: str


@dataclass
class TacticGroup:
    tactics: list[Tactic]
    description: str


@dataclass
class MatchContext:
    p1_bc: TacticalBenchmarkContext
    p2_bc: TacticalBenchmarkContext
    starting_positions: list[GameState]


def is_one_move_from_terminal(game_state: GameState):

    assert len(game_state.players) == 2
    assert tron.get_status(game_state).status == GameStatus.IN_PROGRESS

    p1_dirs = get_possible_directions(game_state, 0)
    p2_dirs = get_possible_directions(game_state, 1)

    if len(p1_dirs) == 0 or len(p2_dirs) == 0:
        return True
    elif len(p1_dirs) == 1 and len(p2_dirs) == 1:
        next_state = next(game_state, (p1_dirs[0], p2_dirs[0]))

        if tron.get_status(next_state).status != GameStatus.IN_PROGRESS:
            return True
    else:
        return False


def model_tensorboard_update(
    i: int,
    tb_writer: SummaryWriter,
    model: TronModel,
    model_desc: str,
    training_result: TrainingResult,
    validation_loss: float,
    make_visualizations: bool,
):

    print(
        f"{training_result.avg_loss=:.3f}, {training_result.avg_prediction_magnitude=:.3f}"
    )

    sos_dict, total_sos = get_sos_info(model)
    print("\nSum of squares (weights/biases):")
    for param, sos_val in sos_dict.items():
        print(f"{param:40s} {sos_val}")

    tb_writer.add_scalar(f"weights_sos/{model_desc}", total_sos, i)
    tb_writer.add_scalar(f"avg_train_loss/{model_desc}", training_result.avg_loss, i)
    tb_writer.add_scalar(
        f"avg_pred_magnitude/{model_desc}", training_result.avg_prediction_magnitude, i
    )

    tb_writer.add_scalar(f"avg_val_loss/{model_desc}", validation_loss, i)

    if make_visualizations:

        terminal_examples = []
        in_prog_examples = []

        for example in training_result.model_examples:

            game_state = example.labeled_example.pov_game_state.game_state

            if is_one_move_from_terminal(game_state):
                terminal_examples.append(example)
            else:
                in_prog_examples.append(example)

        tb_writer.add_image(
            f"training_examples/terminal/{model_desc}",
            render_model_example_image(random.sample(terminal_examples, k=20)),
            global_step=i,
            dataformats="HWC",
        )

        tb_writer.add_image(
            f"training_examples/in_progress/{model_desc}",
            render_model_example_image(random.sample(in_prog_examples, k=20)),
            global_step=i,
            dataformats="HWC",
        )


def benchmark(
    i,
    tb_writer,
    make_visualizations: bool,
    value_contexts: list[ValueBenchmarkContext],
    tactical_contexts: list[TacticalBenchmarkContext],
    match_contexts: list[MatchContext],
    tactic_groups: list[TacticGroup],
):
    # # TODO: formalize
    # value_benchmark_info = [(TIES_5X5, "ties"), (DECISIVE_5X5, "decisive")]

    # # Model value benchmarks
    # for vc in value_contexts:

    #     for value_benchmarks, bench_description in value_benchmark_info:

    #         results = []

    #         for vb in value_benchmarks:
    #             results.extend(run_value_benchmark(vb, vc.model))

    #         if make_visualizations:

    #             tb_writer.add_image(
    #                 f"value_benchmarks/{bench_description}/{vc.description}",
    #                 render_model_example_image(results),
    #                 global_step=i,
    #                 dataformats="HWC",
    #             )

    #         avg_diff = sum(
    #             [abs(r.labeled_example.label - r.prediction) for r in results]
    #         ) / len(results)

    #         tb_writer.add_scalar(
    #             f"Avg. Value Diff ({bench_description}) ({vc.description})", avg_diff, i
    #         )

    for tc in tqdm(tactical_contexts, "Running tactics..."):

        for tactic_group in tactic_groups:

            passes = fails = 0

            results = []

            for t in tactic_group.tactics:

                results.extend(run_tactic(t, tc.dir_fn))

            for r in results:

                if r.correct_moves == len(r.tactic.opposing_dirs):
                    passes += 1
                else:
                    fails += 1

            tb_writer.add_scalar(
                f"{tactic_group.description} tactics pass rate ({tc.description})",
                passes / (passes + fails),
                i,
            )

            if make_visualizations:

                tb_writer.add_image(
                    f"tactics/{tactic_group.description}/{tc.description}",
                    render_tactic_benchmark_image(results),
                    global_step=i,
                    dataformats="HWC",
                )

    # Match score

    # for mc in match_contexts:

    #     p1_wins, p2_wins, ties = match(
    #         mc.p1_bc.dir_fn, mc.p2_bc.dir_fn, mc.starting_positions
    #     )

    #     p1_match_score = (p1_wins + ties * 0.5) / (p1_wins + p2_wins + ties)

    #     tb_writer.add_scalar(
    #         f"{mc.p1_bc.description} match score vs. {mc.p2_bc.description})",
    #         p1_match_score,
    #         i,
    #     )
