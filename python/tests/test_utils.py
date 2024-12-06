import pytest
import torch
import random

from AI.pytorch_game_utils import (
    get_position_evaluation,
    get_model_input_from_raw_info,
    GameResult,
)
from game.ko_tron import KOTron

# TODO:
device = "cuda"


# TODO: Clean this shit up
def test_get_model_input_from_raw_info():

    for i in range(100):
        game = KOTron(num_players=2, dimension=random.randint(4, 40), random_starts=True)

        player_num = 0

        result = get_model_input_from_raw_info(game.grid, game.get_heads(), 0, 69420)
        print(game)

        heads = game.get_heads()

        hero_val = result[0][1][heads[0][0]][heads[0][1]]
        assert hero_val == 1

        antag_val = result[0][2][heads[1][0]][heads[1][1]]
        assert antag_val == -1


        # Opposite case
        result = get_model_input_from_raw_info(game.grid, game.get_heads(), 1, 69420)
        print(game)

        heads = game.get_heads()

        hero_val = result[0][1][heads[1][0]][heads[1][1]]
        assert hero_val == 1

        antag_val = result[0][2][heads[0][0]][heads[0][1]]
        assert antag_val == -1

        for x in range(game.dimension):
            for y in range(game.dimension):
                if game.grid[x][y] != 0:
                    assert result[0][0][x][y] == 1
                else:
                    assert result[0][0][x][y] == 0

    print("bp")


def test_get_position_evaluation():

    non_neutral_tie_result = get_position_evaluation(
        game_progress=1.0, game_result=GameResult.TIE, is_tie_neutral=False
    )

    assert non_neutral_tie_result.dtype == torch.float32
    assert non_neutral_tie_result == -1

    assert (
        get_position_evaluation(
            game_progress=1.0,
            game_result=GameResult.TIE,
            is_tie_neutral=True,
        )
        == 0
    )

    with pytest.raises(ValueError):
        get_position_evaluation(
            game_progress=1.5,
            game_result=GameResult.TIE,
            is_tie_neutral=True,
        )

    with pytest.raises(AssertionError):
        get_position_evaluation(
            game_progress=1.0,
            game_result=GameResult.WIN,
            decay_fn=lambda x: x * 2,
        )


if __name__ == "__main__":
    test_get_model_input_from_raw_info()
