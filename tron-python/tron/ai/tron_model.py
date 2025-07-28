import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tron.game import GameState, Player


@dataclass
class HeroGameState:
    game_state: GameState
    hero_index: int


class TronModel(torch.nn.Module, ABC):

    @abstractmethod
    def get_model_input(
        self, hero_game_states: list[HeroGameState]
    ) -> torch.Tensor:
        pass


    @abstractmethod
    def run_inference(self, hero_game_states: list[HeroGameState]) -> np.ndarray:
        pass

class RandomTronModel(TronModel):
    def get_model_input(self, hero_game_states: list[HeroGameState]):
        raise NotImplementedError()
    
    def run_inference(self, hero_game_states: list[HeroGameState]) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=len(hero_game_states))
    


# class NnueTronModel(TronModelAbstract):
#     def __init__(self, num_rows, num_cols, device, torch_model):

#         self.num_rows = num_rows
#         self.num_cols = num_cols
#         self.num_cells = num_rows * num_cols
#         self.device = device
#         self.torch_model = torch_model
#         self.reset_acc()


#     def reset_acc(self):
#         p1 = Player(0, 0, True)
#         p2 = Player(1,1, True)
        
#         # NOTE: Sketch
#         self.prev_game_state = GameState.from_players((p1, p2), num_rows=self.num_rows, num_cols=self.num_cols)
#         self.prev_hero_index = 0

#         active_indices = [self.emb_idx_wall(0,0), self.emb_idx_wall(1,1), self.emb_idx_hero_head(0,0), self.emb_idx_opponent_head(1,1)]
#         active_indices = torch.tensor(active_indices, dtype=torch.long).to(self.device)

#         self.acc = self.torch_model.init_accumulator(active_indices)



#     def emb_idx_wall(self, row, col):
#         return row * self.num_cols + col
#     def emb_idx_hero_head(self, row, col):
#         return (self.num_cells) + (row * self.num_cols + col)
#     def emb_idx_opponent_head(self, row, col):
#         return (self.num_cells * 2) + (row * self.num_cols + col)


#     def run_inference(self, game_states: list[GameState], hero_index: int) -> np.ndarray:

#         # 1. Compute indices to add/remove
#         # 2. Update self.acc
#         # 3. Update last game state
#         # 4. Forward pass with self.acc

#         # To expand to batch inference, sequentially do the above

#         evals = []

#         for game_state in game_states:


#             remove_mask = self.prev_game_state.grid & (~game_state.grid)
#             add_mask = game_state.grid & (~self.prev_game_state.grid)

#             # get row/col pairs for each case
#             remove_grid_indices = np.argwhere(remove_mask).tolist()
#             add_grid_indices = np.argwhere(add_mask).tolist()

#             # get emb 
#             remove_emb_indices = [self.emb_idx_wall(row, col) for row, col in remove_grid_indices]
#             add_emb_indices = [self.emb_idx_wall(row, col) for row, col in add_grid_indices]

#             # Previous state's player heads
#             prev_hero_player = self.prev_game_state.players[self.prev_hero_index]
#             prev_hero_emb_index = self.emb_idx_hero_head(prev_hero_player.row, prev_hero_player.col)

#             prev_opponent_index = 0 if self.prev_hero_index == 1 else 1
#             prev_opponent_player = self.prev_game_state.players[prev_opponent_index]
#             prev_opponent_emb_index = self.emb_idx_opponent_head(prev_opponent_player.row, prev_opponent_player.col)

#             remove_emb_indices.extend([prev_hero_emb_index, prev_opponent_emb_index])
#             # Current state's player heads

#             hero_player = game_state.players[hero_index]
#             hero_emb_index = self.emb_idx_hero_head(hero_player.row, hero_player.col)

#             opponent_index = 0 if hero_index == 1 else 1
#             opponent_player = game_state.players[opponent_index]
#             opponent_emb_index = self.emb_idx_opponent_head(opponent_player.row, opponent_player.col)

#             add_emb_indices.extend([hero_emb_index, opponent_emb_index])

#             # Update accumulator and prev variables
#             self.acc = self.torch_model.update_acc(self.acc, remove_emb_indices, add_emb_indices)

#             evals.append(self.torch_model(self.acc).item())

#             self.prev_game_state = game_state
#             self.prev_hero_index = hero_index

#         return np.array(evals)

#         # evals = []
#         # for game_state in game_states:
#         #     if len(game_state.players) > 2:
#         #         raise NotImplementedError()

#         #     num_rows, num_cols = game_state.grid.shape

#         #     assert num_rows == self.num_rows
#         #     assert num_cols == self.num_cols

#         #     hero_player = game_state.players[hero_index]
#         #     hero_emb_index = self.emb_idx_hero_head(hero_player.row, hero_player.col)

#         #     opponent_index = 0 if hero_index == 1 else 1
#         #     opponent_player = game_state.players[opponent_index]
#         #     opponent_emb_index = self.emb_idx_opponent_head(opponent_player.row, opponent_player.col)

#         #     indices = [hero_emb_index, opponent_emb_index]

#         #     for row in range(num_rows):
#         #         for col in range(num_cols):

#         #             if game_state.grid[row][col]:
#         #                 indices.append(self.emb_idx_wall(row, col))
            
#         #     acc = self.torch_model.init_accumulator(torch.tensor(indices, dtype=torch.long))
#         #     evals.append(self.torch_model(acc).item())


#         # return np.array(evals)

#     def get_model_input(
#         self, game_states: list[GameState], hero_index: int
#     ) -> torch.Tensor:

#         accs = []
        
#         for game_state in game_states:
#             if len(game_state.players) > 2:
#                 raise NotImplementedError()

#             num_rows, num_cols = game_state.grid.shape

#             assert num_rows == self.num_rows
#             assert num_cols == self.num_cols

#             hero_player = game_state.players[hero_index]
#             hero_emb_index = self.emb_idx_hero_head(hero_player.row, hero_player.col)

#             opponent_index = 0 if hero_index == 1 else 1
#             opponent_player = game_state.players[opponent_index]
#             opponent_emb_index = self.emb_idx_opponent_head(opponent_player.row, opponent_player.col)

#             indices = [hero_emb_index, opponent_emb_index]

#             for row in range(num_rows):
#                 for col in range(num_cols):

#                     if game_state.grid[row][col]:
#                         indices.append(self.emb_idx_wall(row, col))

#             accs.append(self.torch_model.init_accumulator(torch.tensor(indices, dtype=torch.long).to(self.device)))

#         return torch.stack(accs)
    
# class RandomTronModel(TronModelAbstract):

#     def get_model_input(self, game_states, player_index):
#         pass
    
#     def run_inference(self, game_states, player_index):
#         return np.random.uniform(-1.0, 1.0, size=len(game_states))
    

# class CnnTronModel(TronModelAbstract):
#     def __init__(self, model, device):

#         self.device = device
#         self.model = model

#     def get_model_input(
#         self, game_states: list[GameState], player_index: int
#     ) -> torch.Tensor:

#         bool_array = (
#             np.stack([game.grid for game in game_states], axis=0)
#             .astype(np.float32)
#             .reshape((len(game_states), 1, 10, 10))
#         )

#         pos_array = np.zeros((len(game_states), 2, 10, 10))

#         # NOTE: Assuming 2 players
#         opponent_index = 1 if player_index == 0 else 0

#         for i, game in enumerate(game_states):

#             pos_array[
#                 i, 0, game.players[player_index].row, game.players[player_index].col
#             ] = 1.0
#             pos_array[
#                 i, 1, game.players[opponent_index].row, game.players[opponent_index].col
#             ] = -1.0

#         combined_array = np.concatenate((bool_array, pos_array), axis=1)

#         tensor_output = torch.tensor(combined_array, dtype=torch.float32).to(
#             self.device
#         )

#         return tensor_output
    

# class OneHotTransformerTronModel(TronModelAbstract):
#     def __init__(self, model, device):

#         self.device = device
#         self.model = model

#     def get_model_input(
#         self, game_states: list[GameState], player_index: int
#     ) -> torch.Tensor:
        

#         # NOTE: Assuming 2 players
#         if len(game_states[0].players) > 2:
#             raise NotImplementedError()

#         opponent_index = 1 if player_index == 0 else 0
        
#         num_rows, num_cols = len(game_states[0].grid), len(game_states[0].grid[0])

#         # Shape: (batch_size, 4, tokens)
#         # np_input = np.zeros((len(game_states), 4, num_rows * num_cols))

#         np_input = (
#             np.stack([np.where(np.expand_dims(game.grid.flatten(), axis=1), [0, 1, 0, 0], [1, 0, 0, 0]) for game in game_states], axis=0)
#             .astype(np.float32)
#         )

#         assert np_input.shape == (len(game_states), num_rows * num_cols, 4)

#         for i, game_state in enumerate(game_states):

#             hero_player = game_state.players[player_index]
#             opponent_player = game_state.players[opponent_index]

#             flattened_hero_index = (hero_player.row * num_cols) + hero_player.col
#             flattened_opponent_index = (opponent_player.row * num_cols) + opponent_player.col

#             np_input[i, flattened_hero_index, :] = [0, 0, 1, 0]
#             np_input[i, flattened_opponent_index, :] = [0, 0, 0, 1]

                
#         tensor_output = torch.tensor(np_input, dtype=torch.float32).to(
#             self.device
#         )

#         return tensor_output
    

# class EmbeddingTransformerTronModel(TronModelAbstract):
#     def __init__(self, model, device):

#         self.device = device
#         self.model = model

#     def get_model_input(
#         self, game_states: list[GameState], player_index: int
#     ) -> torch.Tensor:
        

#         # NOTE: Assuming 2 players
#         if len(game_states[0].players) > 2:
#             raise NotImplementedError()

#         opponent_index = 1 if player_index == 0 else 0
        
#         num_rows, num_cols = len(game_states[0].grid), len(game_states[0].grid[0])

#         # Shape: (batch_size, 4, tokens)
#         # np_input = np.zeros((len(game_states), 4, num_rows * num_cols))

#         np_input = (
#             np.stack([np.where(game.grid.flatten(), 1, 0) for game in game_states], axis=0)
#             .astype(np.uint8)
#         )

#         assert np_input.shape == (len(game_states), num_rows * num_cols)

#         for i, game_state in enumerate(game_states):

#             hero_player = game_state.players[player_index]
#             opponent_player = game_state.players[opponent_index]

#             flattened_hero_index = (hero_player.row * num_cols) + hero_player.col
#             flattened_opponent_index = (opponent_player.row * num_cols) + opponent_player.col

#             np_input[i, flattened_hero_index] = 2
#             np_input[i, flattened_opponent_index] = 3

                
#         tensor_output = torch.tensor(np_input, dtype=torch.long).to(
#             self.device
#         )

#         return tensor_output





