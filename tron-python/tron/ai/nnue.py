
from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tron import GameState, Player
from tron.ai.training import HeroGameState


class TronModel(nn.Module, ABC):

    @abstractmethod
    def get_model_input(
        self, game_states: list[GameState], player_index: int
    ) -> torch.Tensor:
        pass


    @abstractmethod
    def run_inference(self, game_states: list[GameState], player_index: int) -> np.ndarray:
        pass


# --- 2. Define the efficient‐updatable net ---
class NnueTronModel(TronModel):
    def __init__(self, num_rows=10, num_cols=10, acc_dim=128, hidden_dim=64):
        super().__init__()
        # Embedding table: feature → acc_dim vector
        self.embedding = nn.Embedding(num_rows * num_cols * 3, acc_dim)
        # Tiny MLP on top of the accumulator
        self.fc1 = nn.Linear(acc_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols
        self.reset_acc()


    def init_accumulator(self, active_indices: list[int]):
        """
        Build accumulator from scratch by summing embeddings
        active_indices: list or 1D tensor of feature indices that are “on”
        """
        active_indices = torch.tensor(active_indices, dtype=torch.long)
        emb = self.embedding(active_indices)           # [#active × acc_dim]
        return emb.sum(dim=0)               # → [acc_dim]

    def update_acc(self, acc, to_remove, to_add):
        """
        Efficient delta‐update:
          acc ← acc - E[to_remove] + E[to_add]
        to_remove, to_add: single indices or lists of indices
        """
        # wrap into LongTensor
        rem = torch.tensor(to_remove, dtype=torch.long)
        add = torch.tensor(to_add, dtype=torch.long)

        emb_rem = self.embedding(rem).sum(dim=0)
        emb_add = self.embedding(add).sum(dim=0)
        return acc - emb_rem + emb_add

    def forward(self, acc):
        # 3. Clamp and run MLP
        # x = torch.clamp(acc, min=0.0, max=127.0)  # mimic 8-bit clamp
        h = F.relu(self.fc1(acc))
        out = self.fc2(h)
        return out.squeeze(-1)
    

    def reset_acc(self):

        game_state = GameState.new_game(num_players=2, num_rows=self.num_rows, num_cols=self.num_cols, random_starts=True)

        hero_index = 0

        active_indices = self.get_active_indices(HeroGameState(game_state, hero_index))

        self.acc = self.init_accumulator(active_indices)
        self.prev_game_state = game_state
        self.prev_hero_index = hero_index


    def emb_idx_wall(self, row, col):
        return row * self.num_cols + col
    def emb_idx_hero_head(self, row, col):
        return (self.num_cells) + (row * self.num_cols + col)
    def emb_idx_opponent_head(self, row, col):
        return (self.num_cells * 2) + (row * self.num_cols + col)


    def run_inference(self, hero_game_states: list[HeroGameState]) -> np.ndarray:
        with torch.no_grad():
            evals = []

            for hero_game_state in hero_game_states:

                if len(hero_game_state.game_state.players) != 2:
                    raise NotImplementedError()
                
                num_rows, num_cols = hero_game_state.game_state.grid.shape

                assert num_rows == self.num_rows
                assert num_cols == self.num_cols
                
                hero_index = hero_game_state.hero_index
                opponent_index = 0 if hero_index == 1 else 1
                game_state = hero_game_state.game_state

                remove_mask = self.prev_game_state.grid & (~game_state.grid)
                add_mask = game_state.grid & (~self.prev_game_state.grid)

                # get row/col pairs for each case
                remove_grid_indices = np.argwhere(remove_mask).tolist()
                add_grid_indices = np.argwhere(add_mask).tolist()

                # get emb 
                remove_emb_indices = [self.emb_idx_wall(row, col) for row, col in remove_grid_indices]
                add_emb_indices = [self.emb_idx_wall(row, col) for row, col in add_grid_indices]

                # Previous state's player heads
                prev_hero_player = self.prev_game_state.players[self.prev_hero_index]
                prev_hero_emb_index = self.emb_idx_hero_head(prev_hero_player.row, prev_hero_player.col)

                prev_opponent_index = 0 if self.prev_hero_index == 1 else 1
                prev_opponent_player = self.prev_game_state.players[prev_opponent_index]
                prev_opponent_emb_index = self.emb_idx_opponent_head(prev_opponent_player.row, prev_opponent_player.col)

                remove_emb_indices.extend([prev_hero_emb_index, prev_opponent_emb_index])
                # Current state's player heads

                hero_player = game_state.players[hero_index]
                hero_emb_index = self.emb_idx_hero_head(hero_player.row, hero_player.col)

                opponent_player = game_state.players[opponent_index]
                opponent_emb_index = self.emb_idx_opponent_head(opponent_player.row, opponent_player.col)

                add_emb_indices.extend([hero_emb_index, opponent_emb_index])

                # Update accumulator and prev variables
                self.acc = self.update_acc(self.acc, remove_emb_indices, add_emb_indices)

                evals.append(self(self.acc).item())

                self.prev_game_state = game_state
                self.prev_hero_index = hero_index

        return np.array(evals)


    def get_active_indices(self, hero_game_state: HeroGameState) -> list[int]:

        if len(hero_game_state.game_state.players) != 2:
            raise NotImplementedError()
        
        hero_index = hero_game_state.hero_index
        opponent_index = 0 if hero_index == 1 else 1
        game_state = hero_game_state.game_state

        num_rows, num_cols = game_state.grid.shape

        assert num_rows == self.num_rows
        assert num_cols == self.num_cols

        hero_player = game_state.players[hero_index]
        hero_emb_index = self.emb_idx_hero_head(hero_player.row, hero_player.col)

        opponent_player = game_state.players[opponent_index]
        opponent_emb_index = self.emb_idx_opponent_head(opponent_player.row, opponent_player.col)

        indices = [hero_emb_index, opponent_emb_index]

        for row in range(num_rows):
            for col in range(num_cols):

                if game_state.grid[row][col]:
                    indices.append(self.emb_idx_wall(row, col))

        return indices

    def get_model_input(
        self, hero_game_states: list[HeroGameState]
    ) -> torch.Tensor:

        accs = []
        
        for hero_game_state in hero_game_states:

            active_indices = self.get_active_indices(hero_game_state)

            accs.append(self.init_accumulator(active_indices))

        return torch.stack(accs)
    
