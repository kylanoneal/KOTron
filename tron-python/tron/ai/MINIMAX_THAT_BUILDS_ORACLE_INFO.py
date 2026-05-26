from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto

import tron
from tron.game import GameState, StatusInfo, GameStatus, Direction, Player

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
    HERO_WIN = auto()
    OPPO_WIN = auto()
    TIE = auto()


class SpecialCase(Enum):
    ONE_TIE_ONE_WIN = auto()
    DIFF_STEPS_TO_SAME_RESULT = auto()
    OPPOSITE_RESULT = auto()


@dataclass(frozen=True)
class Response:
    pvs: list[Direction]
    slower_pvs: list[Direction]
    non_pvs: list[Direction]


@dataclass(frozen=True)
class Move:
    dir: Direction
    response: Response


@dataclass
class OracleInfo:
    result: GameResult
    steps_to_result: int

    # NOTE: Hero goes first, then opponent responds
    hero_player: Optional[Player] = None
    oppo_player: Optional[Player] = None
    pvs: Optional[list[Move]] = None
    slower_pvs: Optional[list[Move]] = None
    non_pvs: Optional[list[Move]] = None
    response: Optional[Response] = None
    special_case: Optional[SpecialCase] = None


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


class ResultComparison(Enum):
    BETTER = auto()
    EQUAL = auto()
    SLOWER_WIN_OR_QUICKER_LOSS = auto()
    WORSE = auto()


def swap_perspective(oracle_info: OracleInfo):

    if oracle_info.result == GameResult.TIE:

        new_result = GameResult.TIE
    else:

        new_result = (
            GameResult.HERO_WIN
            if oracle_info.result == GameResult.OPPO_WIN
            else GameResult.OPPO_WIN
        )

    return OracleInfo(
        new_result,
        steps_to_result=oracle_info.steps_to_result,
        hero_player=oracle_info.oppo_player,
        oppo_player=oracle_info.hero_player,
    )


# NOTE: This could be done more cleverly
def compare_results(
    current_best_oracle: OracleInfo,
    new_oracle: OracleInfo,
    is_hero: bool,
):

    # Handle equal results:
    if current_best_oracle.result == new_oracle.result:

        # Don't care about steps to result for ties
        if current_best_oracle.result == GameResult.TIE:
            return ResultComparison.EQUAL
        else:
            if current_best_oracle.steps_to_result == new_oracle.steps_to_result:
                return ResultComparison.EQUAL


    perspective_win = GameResult.HERO_WIN if is_hero else GameResult.OPPO_WIN
    perspective_loss = GameResult.OPPO_WIN if is_hero else GameResult.HERO_WIN

    # Win for hero
    if new_oracle.result == perspective_win:

        if current_best_oracle.result == perspective_win:

            if new_oracle.steps_to_result < current_best_oracle.steps_to_result:
                return ResultComparison.BETTER
            elif new_oracle.steps_to_result > current_best_oracle.steps_to_result:
                return ResultComparison.SLOWER_WIN_OR_QUICKER_LOSS
            else:
                raise AssertionError("Should've already returned")
        else:
            return ResultComparison.BETTER

    elif new_oracle.result == GameResult.TIE:

        if current_best_oracle.result == perspective_win:
            return ResultComparison.WORSE
        elif current_best_oracle.result == perspective_loss:
            return ResultComparison.BETTER
        else:
            raise AssertionError("Should've already returned")

    # Loss for hero
    elif new_oracle.result == perspective_loss:

        if current_best_oracle.result == perspective_loss:

            if new_oracle.steps_to_result > current_best_oracle.steps_to_result:
                return ResultComparison.BETTER
            elif new_oracle.steps_to_result < current_best_oracle.steps_to_result:
                return ResultComparison.SLOWER_WIN_OR_QUICKER_LOSS
            else:
                raise AssertionError("Should've already returned")
        else:
            return ResultComparison.WORSE


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
            GameResult.TIE, 0,
        )

        return oracle_info

    elif status_info.status == GameStatus.WINNER:

        oracle_info = OracleInfo(
            GameResult.HERO_WIN if status_info.winner_index == hero_index else GameResult.OPPO_WIN,
            0,
        )

        return oracle_info

    # Maximizing
    if is_hero:

        # Lookup and return oracle info if we already have it
        oracle_info_lookup = context.oracle_table.get(game_state)

        if oracle_info_lookup is not None:

            if oracle_info_lookup.hero_player == game_state.players[hero_index]:
                assert oracle_info_lookup.hero_player == game_state.players[hero_index]
                assert oracle_info_lookup.oppo_player == game_state.players[opponent_index]

                # Perspective matches
                return oracle_info_lookup
            else:

                # Perspective swap
                assert oracle_info_lookup.hero_player == game_state.players[opponent_index]
                assert oracle_info_lookup.oppo_player == game_state.players[hero_index]

                return swap_perspective(oracle_info_lookup)

        # Otherwise proceed with minimax

        possible_directions = tron.get_possible_directions(game_state, hero_index)

        possible_directions = (
            possible_directions if len(possible_directions) > 0 else [Direction.UP]
        )

        best_oracle_info: OracleInfo = None

        oracle_infos = []

        for direction in possible_directions:

            oracle_info = oracle_minimax(
                game_state,
                depth,
                is_hero=False,
                hero_move=direction,
                context=context,
            )

            oracle_infos.append(oracle_info)

            if best_oracle_info is None:
                best_oracle_info = oracle_info
            else:
                compare_result = compare_results(
                    best_oracle_info, oracle_info, is_hero
                )

                if compare_result == ResultComparison.BETTER:
                    best_oracle_info = oracle_info

        pvs: list[Move] = []
        slower_pvs: list[Move] = []
        non_pvs: list[Move] = []

        for i, oracle_info in enumerate(oracle_infos):

            compare_result = compare_results(
                best_oracle_info, oracle_info, is_hero
            )

            move = Move(possible_directions[i], oracle_info.response)

            if compare_result == ResultComparison.EQUAL:
                pvs.append(move)
            elif compare_result == ResultComparison.SLOWER_WIN_OR_QUICKER_LOSS:
                slower_pvs.append(move)
            elif compare_result == ResultComparison.WORSE:
                non_pvs.append(move)
            else:
                raise AssertionError()

        new_oracle_info = OracleInfo(
            best_oracle_info.result,
            steps_to_result=best_oracle_info.steps_to_result + 1,
            hero_player=game_state.players[hero_index],
            oppo_player=game_state.players[opponent_index],
            pvs=pvs,
            slower_pvs=slower_pvs,
            non_pvs=non_pvs,
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
        oracle_infos = []

        for direction in possible_directions:

            directions = [None, None]
            directions[hero_index] = hero_move
            directions[opponent_index] = direction

            child_state = tron.next(game_state, directions=tuple(directions))

            oracle_info = oracle_minimax(
                child_state, depth - 1, is_hero=True, context=context
            )

            oracle_infos.append(oracle_info)

            if best_oracle_info is None:
                best_oracle_info = oracle_info
            else:
                compare_result = compare_results(
                    best_oracle_info, oracle_info, is_hero
                )

                if compare_result == ResultComparison.BETTER:
                    best_oracle_info = oracle_info

        pvs: list[Direction] = []
        slower_pvs: list[Direction] = []
        non_pvs: list[Direction] = []

        for i, oracle_info in enumerate(oracle_infos):

            compare_result = compare_results(
                best_oracle_info, oracle_info, is_hero
            )

            if compare_result == ResultComparison.EQUAL:
                pvs.append(possible_directions[i])
            elif compare_result == ResultComparison.SLOWER_WIN_OR_QUICKER_LOSS:
                slower_pvs.append(possible_directions[i])
            elif compare_result == ResultComparison.WORSE:
                non_pvs.append(possible_directions[i])
            else:
                raise AssertionError()

        new_oracle_info = OracleInfo(
            best_oracle_info.result,
            steps_to_result=best_oracle_info.steps_to_result,
            response=Response(pvs, slower_pvs, non_pvs),
        )

        return new_oracle_info
