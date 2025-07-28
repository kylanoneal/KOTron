import pytest
import numpy as np
from tron.game import GameState



def validate_game_state(game: GameState):

    assert type(game.players == tuple)
    assert type(game.grid) == np.ndarray
    assert game.grid.dtype == bool
    assert len(game.grid.shape) == 2

    num_rows, num_cols = game.grid.shape

    player_heads = set()
    for player in game.players:

        assert game.grid[player.row][player.col]
        assert 0 <= player.row < num_rows
        assert 0 <= player.col < num_cols

        head_tup = (player.row, player.col)

        assert head_tup not in player_heads
        
        player_heads.add(head_tup)

def test_new_game():


    with pytest.raises(ValueError):
        GameState.new_game(obstacle_density=0.81)

    with pytest.raises(NotImplementedError):
        GameState.new_game(num_players=3)