import zmq
import numpy as np
from tron_io import tron_pb2  # This is the generated file from your .proto schema

from game.tron import GameState, Player

def to_proto(game_data: list[list[GameState]]):
    # Create an instance of GameState
    games_pb = tron_pb2.Games()
    

    for game in game_data:

        game_pb = games_pb.games.add()

        for game_state in game:

            game_state_pb = game_pb.game_states.add()
        
            # Fill the grid into the GameState message
            for row in game_state.grid:
                grid_row = game_state_pb.grid.add()  # Add a new row
                grid_row.cells.extend(row)         # Extend the row with the boolean values
            
            for player in game_state.players:
                player_pb = game_state_pb.players.add()
                player_pb.row = player.row
                player_pb.col = player.col
                player_pb.can_move = player.can_move

    return games_pb.SerializeToString()

                
def from_proto(serialized_data):


    # Deserialize the byte string back into a GameState message
    games_pb = tron_pb2.Games()
    games_pb.ParseFromString(serialized_data)

    game_data = []
    for game_pb in games_pb.games:

        game = []

        for game_state_pb in game_pb.game_states:
        
            grid_list = []
            for grid_row in game_state_pb.grid:
                # Each grid_row.cells is already a sequence of booleans,
                # so we convert it to a list and append it to grid_list.
                grid_list.append(list(grid_row.cells))
            
            grid = np.array(grid_list, dtype=bool)

            players = []
            for player_pb in game_state_pb.players:
                players.append(Player(player_pb.row, player_pb.col, player_pb.can_move))

            game.append(GameState(grid, tuple(players)))
        game_data.append(game) 

    return game_data
