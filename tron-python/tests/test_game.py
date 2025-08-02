import sys
import pytest
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, FrozenInstanceError
from tron.game import GameState, Player



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

def test_init():

    valid_players = tuple([Player(0, 0, True), Player(1,1, True)])


    # None args
    with pytest.raises(TypeError):
        GameState(grid=None, players=None)

    # Int grid
    with pytest.raises(TypeError):
        GameState(grid=np.zeros(shape=(10, 10), dtype=np.uint8), players=valid_players)

    # Players on same square
    with pytest.raises(ValueError):
        same_spot_players = tuple([Player(0, 0, True), Player(0,0, True)])
        GameState(grid=np.zeros(shape=(10, 10), dtype=bool), players=same_spot_players)   

    # List of players
    with pytest.raises(TypeError):
        list_players = list(valid_players)
        GameState(grid=np.zeros(shape=(10, 10), dtype=bool), players=list_players)

    # Grid not True at player heads
    with pytest.raises(ValueError):
        GameState(grid=np.zeros(shape=(10, 10), dtype=bool), players=valid_players)

    # Negative player index
    with pytest.raises(IndexError):
        neg_ind_players = tuple([Player(0,0,True), Player(-1, -1, True)])
        GameState(grid=np.ones(shape=(10, 10), dtype=bool), players=neg_ind_players)

    # Out of bounds player
    with pytest.raises(IndexError):
        out_of_bounds_players = tuple([Player(0,0,True), Player(0, sys.maxsize, True)])
        GameState(grid=np.ones(shape=(10, 10), dtype=bool), players=out_of_bounds_players)


    GameState(np.ones(shape=(100, 100), dtype=bool), valid_players)

    empty_grid = np.zeros(shape=(10, 10), dtype=bool)
    for player in valid_players:
        empty_grid[player.row][player.col] = True

    GameState(empty_grid, valid_players)


def test_mutability():

    game = GameState.new_game(random_starts=True)

    with pytest.raises(FrozenInstanceError):
        game.grid = np.zeros(shape=(10,10), dtype=bool)

    with pytest.raises(FrozenInstanceError):
        game.players[0].row = 5

    with pytest.raises(TypeError):
        game.players[0] = Player(5, 5, False)

    hash_1 = hash(game)
    game_copy = deepcopy(game)
    
    game.grid[0][0] = not game.grid[0][0]

    assert hash_1 != hash(game)
    assert game_copy != game




def test_new_game():


    with pytest.raises(ValueError):
        GameState.new_game(obstacle_density=0.81)

    with pytest.raises(NotImplementedError):
        GameState.new_game(num_players=3)