import pytest
import torch

from AI.pytorch_game_utils import get_position_evaluation, GameResult

# TODO:
device = "cuda"


def test_get_position_evaluation():


    non_neutral_tie_result = get_position_evaluation(
        lambda x: x, game_progress=1.0, game_result=GameResult.TIE, is_tie_neutral=False
    )

    assert non_neutral_tie_result.dtype == torch.float32
    assert non_neutral_tie_result == -1

    assert (
        get_position_evaluation(
            lambda x: x,
            game_progress=1.0,
            game_result=GameResult.TIE,
            is_tie_neutral=True,
        )
        == 0
    )

    with pytest.raises(ValueError):
        get_position_evaluation(
            lambda x: x,
            game_progress=1.5,
            game_result=GameResult.TIE,
            is_tie_neutral=True,
        )

    with pytest.raises(AssertionError):
        get_position_evaluation(
            lambda x: x * 2,
            game_progress=1.0,
            game_result=GameResult.WIN
        )

    with pytest.raises(AssertionError):
        get_position_evaluation(
            lambda x: x * 2,
            game_progress=1.0,
            game_result=GameResult.WIN
        )

