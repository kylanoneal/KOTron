import pytest

import tron.game as game
from tron.ai import minimax
from tron.enums import Direction, GameStatus


def board_from_indices(*indices: int) -> int:
    board = 0
    for index in indices:
        board |= game.BIT_MASKS[index]
    return board


def make_game_state(
    num_rows: int,
    num_cols: int,
    player_specs: tuple[tuple[int, bool], tuple[int, bool]],
    extra_walls: tuple[int, ...] = (),
) -> game.GameState:
    players = tuple(
        game.Player(idx=player_idx, can_move=can_move)
        for player_idx, can_move in player_specs
    )
    board = board_from_indices(*extra_walls, *(player.idx for player in players))

    return game.GameState(
        num_rows=num_rows,
        num_cols=num_cols,
        board=board,
        players=players,
    )


class RecordingModel:
    def __init__(self, default: float = 0.0, evaluations: dict | None = None):
        self.default = default
        self.evaluations = evaluations or {}
        self.calls = []

    def run_inference(self, pov_game_state: game.PovGameState) -> float:
        self.calls.append(pov_game_state)
        return self.evaluations.get(pov_game_state.game_state, self.default)


def make_context(
    model: RecordingModel | None = None,
    hero_index: int = 0,
    opponent_index: int = 1,
    win_magnitude: float = 100.0,
    debug_stack: list[minimax.MinimaxArgs] | None = None,
) -> minimax.MinimaxContext:
    return minimax.MinimaxContext(
        model=model or RecordingModel(),
        hero_index=hero_index,
        opponent_index=opponent_index,
        win_magnitude=win_magnitude,
        debug_stack=debug_stack,
    )


def in_progress_status() -> game.StatusInfo:
    return game.StatusInfo(GameStatus.IN_PROGRESS)


def test_minimax_result_defaults_to_no_principal_variation():
    result = minimax.MinimaxResult(evaluation=1.25)

    assert result.evaluation == 1.25
    assert result.principal_variation is None


def test_tie_returns_zero_without_model_call_and_records_debug_stack():
    game_state = make_game_state(1, 2, ((0, False), (1, False)))
    model = RecordingModel(default=99.0)
    debug_stack = []
    context = make_context(model=model, debug_stack=debug_stack)

    result = minimax.basic_minimax(
        game_state,
        depth=3,
        is_hero=True,
        context=context,
    )

    assert result == minimax.MinimaxResult(0.0, None)
    assert model.calls == []
    assert debug_stack == [
        minimax.MinimaxArgs(
            game_state=game_state,
            depth=3,
            is_hero=True,
            hero_move=None,
        )
    ]


@pytest.mark.parametrize(
    ("player_specs", "hero_index", "depth", "expected_evaluation"),
    [
        (((0, True), (1, False)), 0, 2, 30.0),
        (((0, False), (1, True)), 0, 2, -30.0),
        (((0, False), (1, True)), 1, 4, 50.0),
    ],
)
def test_winner_returns_depth_weighted_terminal_score(
    player_specs: tuple[tuple[int, bool], tuple[int, bool]],
    hero_index: int,
    depth: int,
    expected_evaluation: float,
):
    game_state = make_game_state(1, 2, player_specs)
    context = make_context(
        model=RecordingModel(default=-99.0),
        hero_index=hero_index,
        opponent_index=1 - hero_index,
        win_magnitude=10.0,
    )

    result = minimax.basic_minimax(
        game_state,
        depth=depth,
        is_hero=True,
        context=context,
    )

    assert result == minimax.MinimaxResult(expected_evaluation, None)
    assert context.model.calls == []


def test_depth_zero_in_progress_state_uses_model_from_hero_pov():
    game_state = make_game_state(2, 2, ((0, True), (3, True)))
    model = RecordingModel(default=7.5)
    context = make_context(model=model, hero_index=1, opponent_index=0)

    result = minimax.basic_minimax(
        game_state,
        depth=0,
        is_hero=True,
        context=context,
    )

    assert result == minimax.MinimaxResult(7.5, None)
    assert len(model.calls) == 1
    assert model.calls[0] == game.PovGameState(game_state, 1, 0)


@pytest.mark.parametrize(
    ("is_hero", "hero_move"),
    [
        (True, Direction.UP),
        (False, None),
    ],
)
def test_basic_minimax_rejects_invalid_phase_arguments(
    is_hero: bool,
    hero_move: Direction | None,
):
    game_state = make_game_state(2, 2, ((0, True), (3, True)))
    context = make_context()

    with pytest.raises(AssertionError):
        minimax.basic_minimax(
            game_state,
            depth=1,
            is_hero=is_hero,
            hero_move=hero_move,
            context=context,
        )


def test_terminal_state_must_be_evaluated_on_hero_turn():
    game_state = make_game_state(1, 2, ((0, False), (1, False)))
    context = make_context()

    with pytest.raises(AssertionError):
        minimax.basic_minimax(
            game_state,
            depth=1,
            is_hero=False,
            hero_move=Direction.UP,
            context=context,
        )


def test_depth_zero_requires_hero_turn_for_non_terminal_state():
    game_state = make_game_state(2, 2, ((0, True), (3, True)))
    context = make_context()

    with pytest.raises(AssertionError):
        minimax.basic_minimax(
            game_state,
            depth=0,
            is_hero=False,
            hero_move=Direction.RIGHT,
            context=context,
        )


def test_hero_maximizes_after_opponent_minimizes(monkeypatch: pytest.MonkeyPatch):
    root_state = ("root",)
    leaf_values = {
        (Direction.RIGHT, Direction.LEFT): 10.0,
        (Direction.RIGHT, Direction.DOWN): -5.0,
        (Direction.DOWN, Direction.LEFT): 3.0,
        (Direction.DOWN, Direction.DOWN): 4.0,
    }

    monkeypatch.setattr(
        minimax.tron,
        "get_status",
        lambda game_state: in_progress_status(),
    )
    monkeypatch.setattr(
        minimax.tron,
        "get_possible_directions",
        lambda game_state, player_index: (
            [Direction.RIGHT, Direction.DOWN]
            if player_index == 0
            else [Direction.LEFT, Direction.DOWN]
        ),
    )
    monkeypatch.setattr(
        minimax.tron,
        "next",
        lambda game_state, directions: tuple(directions),
    )

    model = RecordingModel(evaluations=leaf_values)
    debug_stack = []
    context = make_context(model=model, debug_stack=debug_stack)

    result = minimax.basic_minimax(
        root_state,
        depth=1,
        is_hero=True,
        context=context,
    )

    assert result == minimax.MinimaxResult(3.0, Direction.DOWN)
    assert [call.game_state for call in model.calls] == [
        (Direction.RIGHT, Direction.LEFT),
        (Direction.RIGHT, Direction.DOWN),
        (Direction.DOWN, Direction.LEFT),
        (Direction.DOWN, Direction.DOWN),
    ]
    assert debug_stack == [
        minimax.MinimaxArgs(root_state, 1, True, None),
        minimax.MinimaxArgs(root_state, 1, False, Direction.RIGHT),
        minimax.MinimaxArgs((Direction.RIGHT, Direction.LEFT), 0, True, None),
        minimax.MinimaxArgs((Direction.RIGHT, Direction.DOWN), 0, True, None),
        minimax.MinimaxArgs(root_state, 1, False, Direction.DOWN),
        minimax.MinimaxArgs((Direction.DOWN, Direction.LEFT), 0, True, None),
        minimax.MinimaxArgs((Direction.DOWN, Direction.DOWN), 0, True, None),
    ]


def test_opponent_phase_returns_minimizing_direction(monkeypatch: pytest.MonkeyPatch):
    root_state = ("root",)
    leaf_values = {
        (Direction.RIGHT, Direction.LEFT): 10.0,
        (Direction.RIGHT, Direction.DOWN): -5.0,
    }

    monkeypatch.setattr(
        minimax.tron,
        "get_status",
        lambda game_state: in_progress_status(),
    )
    monkeypatch.setattr(
        minimax.tron,
        "get_possible_directions",
        lambda game_state, player_index: [Direction.LEFT, Direction.DOWN],
    )
    monkeypatch.setattr(
        minimax.tron,
        "next",
        lambda game_state, directions: tuple(directions),
    )

    context = make_context(model=RecordingModel(evaluations=leaf_values))

    result = minimax.basic_minimax(
        root_state,
        depth=1,
        is_hero=False,
        hero_move=Direction.RIGHT,
        context=context,
    )

    assert result == minimax.MinimaxResult(-5.0, Direction.DOWN)


def test_no_legal_moves_falls_back_to_up_for_both_players(
    monkeypatch: pytest.MonkeyPatch,
):
    root_state = ("root",)
    next_calls = []

    monkeypatch.setattr(
        minimax.tron,
        "get_status",
        lambda game_state: in_progress_status(),
    )
    monkeypatch.setattr(
        minimax.tron,
        "get_possible_directions",
        lambda game_state, player_index: [],
    )

    def fake_next(game_state, directions):
        next_calls.append(tuple(directions))
        return ("leaf", tuple(directions))

    monkeypatch.setattr(minimax.tron, "next", fake_next)

    context = make_context(model=RecordingModel(default=12.0))

    result = minimax.basic_minimax(
        root_state,
        depth=1,
        is_hero=True,
        context=context,
    )

    assert result == minimax.MinimaxResult(12.0, Direction.UP)
    assert next_calls == [(Direction.UP, Direction.UP)]


def test_depth_one_search_uses_real_game_helpers_for_legal_principal_variation():
    game_state = make_game_state(
        3,
        3,
        ((4, True), (8, True)),
        extra_walls=(1,),
    )
    context = make_context(model=RecordingModel(default=0.25))

    result = minimax.basic_minimax(
        game_state,
        depth=1,
        is_hero=True,
        context=context,
    )

    assert result.evaluation >= -context.win_magnitude
    assert result.principal_variation in game.get_possible_directions(game_state, 0)
