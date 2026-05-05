import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


from tron.game import GameState, Player, PovGameState, from_bitboard
from tron.ai.tron_model import TronModel

class CnnTronModel(TronModel):

    PADDING = 1

    def __init__(self, num_rows, num_cols):

        assert num_rows == num_cols
        self.num_rows = num_rows
        self.num_cols = num_cols

        # FIX PADDING TO BE IMPLICIT HERE
        super(CnnTronModel, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 10, kernel_size=3, stride=1, padding=0
        )  # changed stride to 2
        self.conv2 = nn.Conv2d(
            10, 20, kernel_size=3, stride=1, padding=1
        )  # changed stride to 2
        output_dim1 = self.conv_output_size(num_rows, 3, 1, 1)  # after first conv layer
        output_dim2 = self.conv_output_size(
            output_dim1, 3, 1, 1
        )  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2**2, 128)  # input features for fc1
        self.fc_value = nn.Linear(128, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        padded_game_grid = F.pad(
            x[:, 0:1], pad=(1, 1, 1, 1), mode="constant", value=1.0
        )
        padded_heads = F.pad(x[:, 1:], pad=(1, 1, 1, 1), mode="constant", value=0.0)
        concat = torch.concat((padded_game_grid, padded_heads), dim=1)


        x = F.relu(self.conv1(concat))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten the tensor

        x = F.leaky_relu(self.fc1(x))
        
        value_estimate = self.fc_value(x).squeeze(1)

        return value_estimate
    

    def get_model_input(
        self, pov_game_states: list[PovGameState]
    ) -> torch.Tensor:



        bool_array = (
            np.stack([from_bitboard(h.game_state).grid for h in pov_game_states], axis=0)
            .astype(np.float32)
            .reshape((len(pov_game_states), 1, self.num_rows, self.num_cols))
        )

        pos_array = np.zeros((len(pov_game_states), 2, self.num_rows, self.num_cols))

        for i, pov_game_state in enumerate(pov_game_states):

            game_state = from_bitboard(pov_game_state.game_state)
            hero_index = pov_game_state.hero_index
            opponent_index = 0 if hero_index == 1 else 1

            pos_array[
                i, 0, game_state.players[hero_index].row, game_state.players[hero_index].col
            ] = 1.0
            pos_array[
                i, 1, game_state.players[opponent_index].row, game_state.players[opponent_index].col
            ] = -1.0

        combined_array = np.concatenate((bool_array, pos_array), axis=1)

        tensor_output = torch.tensor(combined_array, dtype=torch.float32)

        return tensor_output
    
    def run_inference(self, pov_game_state: PovGameState) -> np.ndarray:

        model_input = self.get_model_input([pov_game_state])

        output = self(model_input)

        return output.detach().item()