from typing import Optional
from dataclasses import dataclass

import tron
from tron.game import GameState, StatusInfo, GameStatus, Direction

from tron.ai.tron_model import TronModel, PovGameState


@dataclass
class MinimaxResult:
    evaluation: float
    principal_variation: Optional[Direction] = None


@dataclass
class MinimaxArgs:
    game_state: GameState
    depth: int
    is_hero: bool
    hero_move: Direction


@dataclass
class MinimaxContext:
    model: TronModel
    hero_index: int
    opponent_index: int
    win_magnitude: float = 10_000.0
    debug_stack: Optional[list[MinimaxArgs]] = None

    def __post_init__(self):

        assert (0 <= self.hero_index < 2) and (0 <= self.opponent_index < 2)

        assert self.hero_index != self.opponent_index

        assert self.win_magnitude > 1.0


def basic_minimax(
    game_state: GameState,
    depth: int,
    is_hero: bool,
    hero_move: Optional[Direction] = None,
    context: MinimaxContext = None,
) -> MinimaxResult:

    assert depth >= 0
    assert context is not None, "Context must be passed"

    if context.debug_stack is not None:

        context.debug_stack.append(
            MinimaxArgs(
                game_state,
                depth,
                is_hero,
                hero_move,
            )
        )

    if is_hero:
        assert hero_move is None
    else:
        assert hero_move is not None

    hero_index = context.hero_index
    opponent_index = context.opponent_index

    status_info: StatusInfo = tron.get_status(game_state)

    if status_info.status != GameStatus.IN_PROGRESS:
        assert is_hero

    if status_info.status == GameStatus.TIE:
        return MinimaxResult(0.0, None)
    elif status_info.status == GameStatus.WINNER:

        eval_magnitude = context.win_magnitude * (depth + 1)

        eval = (
            eval_magnitude
            if status_info.winner_index == hero_index
            else eval_magnitude * -1
        )
        return MinimaxResult(eval, None)

    if depth == 0:
        assert is_hero

        model_eval = context.model.run_inference(
            PovGameState(game_state, hero_index, opponent_index)
        )

        assert abs(model_eval) < context.win_magnitude

        return MinimaxResult(
            model_eval,
            None,
        )

    if is_hero:

        possible_directions = tron.get_possible_directions(game_state, hero_index)

        possible_directions = (
            possible_directions if len(possible_directions) > 0 else [Direction.UP]
        )

        max_eval = -float("inf")

        for direction in possible_directions:
            mm_result = basic_minimax(
                game_state,
                depth,
                is_hero=False,
                hero_move=direction,
                context=context,
            )

            if mm_result.evaluation > max_eval:
                max_eval = mm_result.evaluation
                best_dir = direction
        return MinimaxResult(max_eval, best_dir)
    else:

        possible_directions = tron.get_possible_directions(game_state, opponent_index)
        possible_directions = (
            possible_directions if len(possible_directions) > 0 else [Direction.UP]
        )

        min_eval = float("inf")
        for direction in possible_directions:

            directions = [None, None]
            directions[hero_index] = hero_move
            directions[opponent_index] = direction

            child_state = tron.next(game_state, directions=tuple(directions))

            mm_result = basic_minimax(
                child_state, depth - 1, is_hero=True, context=context
            )

            if mm_result.evaluation < min_eval:
                min_eval = mm_result.evaluation
                best_dir = direction

        return MinimaxResult(min_eval, best_dir)
