from typing import Optional
from dataclasses import dataclass
from enum import Enum

import tron
from tron.game import GameState, StatusInfo, GameStatus, Direction

from tron.ai.tron_model import TronModel, PovGameState

# @dataclass
# class MinimaxResult:
#     game: GameState
#     dir: Direction
#     evaluation: float
#     is_hero: bool
#     depth: int  # NOTE: Need a better way to store "steps until terminal" than this
#     sub_results: list["MinimaxResult"]


class GameResult(Enum):
    P1_WIN = "p1_win"
    P2_WIN = "p2_win"
    TIE = "tie"


@dataclass
class OracleInfo:
    result: GameResult
    steps_to_result: int
    pvs: Optional[tuple[tuple[Direction]]] = None
    slower_pvs: Optional[tuple[tuple[Direction]]] = None
    non_pvs: Optional[tuple[tuple[Direction]]] = None


@dataclass
class MinimaxContext:
    model: TronModel
    hero_index: int
    opponent_index: int
    oracle_table: dict[GameState, OracleInfo]
    win_magnitude: float = 10_000.0

    def __post_init__(self):

        assert (0 <= self.hero_index < 2) and (0 <= self.opponent_index < 2)

        assert self.hero_index != self.opponent_index

        assert self.win_magnitude > 1.0


# NOTE: This could be done more cleverly
def is_better_or_equal_result(
    current_best_oracle: OracleInfo,
    new_oracle: OracleInfo,
    mm_context: MinimaxContext,
    is_hero: bool,
):

    if current_best_oracle is None:
        return True

    # Switch perspective
    hero_index = mm_context.hero_index if is_hero else mm_context.opponent_index

    if hero_index == 0:

        # Win for hero
        if new_oracle.result == GameResult.P1_WIN:

            if current_best_oracle.result == GameResult.P1_WIN:

                return new_oracle.steps_to_result <= current_best_oracle.steps_to_result
            else:
                return True

        elif new_oracle.result == GameResult.TIE:

            if current_best_oracle.result == GameResult.P1_WIN:
                return False
            else:
                return True

        # Loss for hero
        elif new_oracle.result == GameResult.P2_WIN:

            if current_best_oracle.result == GameResult.P2_WIN:
                return new_oracle.steps_to_result >= current_best_oracle.steps_to_result
            else:
                return False

    elif hero_index == 1:

        # Win for hero
        if new_oracle.result == GameResult.P2_WIN:

            if current_best_oracle.result == GameResult.P2_WIN:

                return new_oracle.steps_to_result <= current_best_oracle.steps_to_result
            else:
                return True

        elif new_oracle.result == GameResult.TIE:

            if current_best_oracle.result == GameResult.P2_WIN:
                return False
            else:
                return True

        # Loss for hero
        elif new_oracle.result == GameResult.P1_WIN:

            if current_best_oracle.result == GameResult.P1_WIN:
                return new_oracle.steps_to_result >= current_best_oracle.steps_to_result
            else:
                return False


def oracle_minimax(
    game_state: GameState,
    depth: int,
    is_hero: bool,
    hero_move: Optional[Direction] = None,
    context: MinimaxContext = None,
) -> OracleInfo:

    assert depth > 0, "Oracle minimax should not reach depth 0"
    assert context is not None, "Context must be passed"

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

        oracle_info = OracleInfo(
            GameResult.TIE, 0, pvs=None, slower_pvs=None, non_pvs=None
        )

        return oracle_info

    elif status_info.status == GameStatus.WINNER:

        oracle_info = OracleInfo(
            GameResult.P1_WIN if status_info.winner_index == 0 else GameResult.P2_WIN,
            0,
            pvs=None,
            slower_pvs=None,
            non_pvs=None,
        )

        return oracle_info
    
    


    # Maximizing
    if is_hero:


        # Lookup and return oracle info if we already have it
        oracle_info_lookup = context.oracle_table.get(game_state)

        if oracle_info_lookup is not None:
            return oracle_info_lookup
        
        # Otherwise proceed with minimax

        possible_directions = tron.get_possible_directions(game_state, hero_index)

        possible_directions = (
            possible_directions if len(possible_directions) > 0 else [Direction.UP]
        )

        best_oracle_info: OracleInfo = None

        for direction in possible_directions:

            oracle_info = oracle_minimax(
                game_state,
                depth,
                is_hero=False,
                hero_move=direction,
                context=context,
            )

            if is_better_or_equal_result(
                best_oracle_info, oracle_info, context, is_hero
            ):

                best_oracle_info = oracle_info

        new_oracle_info = OracleInfo(
            best_oracle_info.result,
            steps_to_result=best_oracle_info.steps_to_result + 1,
        )
        # Hero perspective updates oracle table
        context.oracle_table[game_state] = new_oracle_info

        return new_oracle_info

    # Minimizing
    else:

        possible_directions = tron.get_possible_directions(game_state, opponent_index)
        possible_directions = (
            possible_directions if len(possible_directions) > 0 else [Direction.UP]
        )

        best_oracle_info: OracleInfo = None

        for direction in possible_directions:

            directions = [None, None]
            directions[hero_index] = hero_move
            directions[opponent_index] = direction

            child_state = tron.next(game_state, directions=tuple(directions))

            oracle_info = oracle_minimax(
                child_state, depth - 1, is_hero=True, context=context
            )

            if is_better_or_equal_result(
                best_oracle_info, oracle_info, context, is_hero
            ):

                best_oracle_info = oracle_info

        return best_oracle_info
