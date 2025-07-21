import torch
import numpy as np
from abc import ABC, abstractmethod

from tron.game import  GameState


class TronModelAbstract(ABC):

    @abstractmethod
    def get_model_input(
        self, game_states: list[GameState], player_index: int
    ) -> torch.Tensor:
        pass

    def run_inference(self, game_states: list[GameState], player_index: int) -> np.ndarray:

        # NOTE: Maybe should have a way to set requries_grad = False for infernce time
        model_input = self.get_model_input(game_states, player_index)

        self.model.eval()

        return self.model(model_input).detach().cpu().numpy()

class RandomTronModel(TronModelAbstract):

    def get_model_input(self, game_states, player_index):
        pass
    
    def run_inference(self, game_states, player_index):
        return np.random.uniform(-1.0, 1.0, size=len(game_states))
    

class CnnTronModel(TronModelAbstract):
    def __init__(self, model, device):

        self.device = device
        self.model = model

    def get_model_input(
        self, game_states: list[GameState], player_index: int
    ) -> torch.Tensor:

        bool_array = (
            np.stack([game.grid for game in game_states], axis=0)
            .astype(np.float32)
            .reshape((len(game_states), 1, 10, 10))
        )

        pos_array = np.zeros((len(game_states), 2, 10, 10))

        # NOTE: Assuming 2 players
        opponent_index = 1 if player_index == 0 else 0

        for i, game in enumerate(game_states):

            pos_array[
                i, 0, game.players[player_index].row, game.players[player_index].col
            ] = 1.0
            pos_array[
                i, 1, game.players[opponent_index].row, game.players[opponent_index].col
            ] = -1.0

        combined_array = np.concatenate((bool_array, pos_array), axis=1)

        tensor_output = torch.tensor(combined_array, dtype=torch.float32).to(
            self.device
        )

        return tensor_output
    

class OneHotTransformerTronModel(TronModelAbstract):
    def __init__(self, model, device):

        self.device = device
        self.model = model

    def get_model_input(
        self, game_states: list[GameState], player_index: int
    ) -> torch.Tensor:
        

        # NOTE: Assuming 2 players
        if len(game_states[0].players) > 2:
            raise NotImplementedError()

        opponent_index = 1 if player_index == 0 else 0
        
        num_rows, num_cols = len(game_states[0].grid), len(game_states[0].grid[0])

        # Shape: (batch_size, 4, tokens)
        # np_input = np.zeros((len(game_states), 4, num_rows * num_cols))

        np_input = (
            np.stack([np.where(np.expand_dims(game.grid.flatten(), axis=1), [0, 1, 0, 0], [1, 0, 0, 0]) for game in game_states], axis=0)
            .astype(np.float32)
        )

        assert np_input.shape == (len(game_states), num_rows * num_cols, 4)

        for i, game_state in enumerate(game_states):

            hero_player = game_state.players[player_index]
            opponent_player = game_state.players[opponent_index]

            flattened_hero_index = (hero_player.row * num_cols) + hero_player.col
            flattened_opponent_index = (opponent_player.row * num_cols) + opponent_player.col

            np_input[i, flattened_hero_index, :] = [0, 0, 1, 0]
            np_input[i, flattened_opponent_index, :] = [0, 0, 0, 1]

                
        tensor_output = torch.tensor(np_input, dtype=torch.float32).to(
            self.device
        )

        return tensor_output
    

class EmbeddingTransformerTronModel(TronModelAbstract):
    def __init__(self, model, device):

        self.device = device
        self.model = model

    def get_model_input(
        self, game_states: list[GameState], player_index: int
    ) -> torch.Tensor:
        

        # NOTE: Assuming 2 players
        if len(game_states[0].players) > 2:
            raise NotImplementedError()

        opponent_index = 1 if player_index == 0 else 0
        
        num_rows, num_cols = len(game_states[0].grid), len(game_states[0].grid[0])

        # Shape: (batch_size, 4, tokens)
        # np_input = np.zeros((len(game_states), 4, num_rows * num_cols))

        np_input = (
            np.stack([np.where(game.grid.flatten(), 1, 0) for game in game_states], axis=0)
            .astype(np.uint8)
        )

        assert np_input.shape == (len(game_states), num_rows * num_cols)

        for i, game_state in enumerate(game_states):

            hero_player = game_state.players[player_index]
            opponent_player = game_state.players[opponent_index]

            flattened_hero_index = (hero_player.row * num_cols) + hero_player.col
            flattened_opponent_index = (opponent_player.row * num_cols) + opponent_player.col

            np_input[i, flattened_hero_index] = 2
            np_input[i, flattened_opponent_index] = 3

                
        tensor_output = torch.tensor(np_input, dtype=torch.long).to(
            self.device
        )

        return tensor_output





