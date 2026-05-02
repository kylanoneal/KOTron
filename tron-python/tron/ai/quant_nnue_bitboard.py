from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from warnings import warn
from tron.ai.tron_model import TronModel

from tron.game import PovGameState, get_wall_indices


# --- 2. Define the efficient‐updatable net ---
class NnueTronModel(TronModel):
    def __init__(self, num_rows, num_cols, acc_dim=128):
        super().__init__()
        # Embedding table: feature → acc_dim vector
        self.embedding = nn.Embedding(num_rows * num_cols * 3, acc_dim)
        # Tiny MLP on top of the accumulator
        self.fc1 = nn.Linear(acc_dim, 1)

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols

    def load_state_dict(self, state_dict, strict: bool = True):

        # Call super and return the IncompatibleKeys namedtuple
        rv = super().load_state_dict(state_dict, strict=strict)


        return rv

    def init_accumulator(self, active_indices: list[int]):
        """
        Build accumulator from scratch by summing embeddings
        active_indices: list or 1D tensor of feature indices that are “on”
        """
        active_indices = torch.tensor(active_indices, dtype=torch.long)
        emb = self.embedding(active_indices)  # [#active × acc_dim]
        return emb.sum(dim=0)  # → [acc_dim]

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

        x = torch.clamp(acc, min=0.0, max=1.0)
        out = self.fc1(x)
        return out.squeeze(-1)

    def emb_idx_wall(self, idx):
        return idx

    def emb_idx_hero_head(self, idx):
        return (self.num_cells) + idx

    def emb_idx_opponent_head(self, idx):
        return (self.num_cells * 2) + idx

    # TODO: Make static?
    def get_active_indices(self, pov_game_state: PovGameState) -> list[int]:

        if len(pov_game_state.game_state.players) != 2:
            raise NotImplementedError()

        game_state = pov_game_state.game_state

        hero_emb_index = self.emb_idx_hero_head(game_state.players[pov_game_state.hero_index].idx)

        opponent_emb_index = self.emb_idx_opponent_head(
            game_state.players[pov_game_state.opponent_index].idx
        )

        indices = [hero_emb_index, opponent_emb_index] + get_wall_indices(pov_game_state.game_state)



        return indices

    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:

        accs = []

        for pov_game_state in pov_game_states:

            active_indices = self.get_active_indices(pov_game_state)

            accs.append(self.init_accumulator(active_indices))

        return torch.stack(accs)
    
    def run_inference(self, pov_game_state: PovGameState) -> float:

        raise NotImplementedError("Should be using quantized version.")


class QuantizedNnueTronModel(TronModel):

    def __init__(self, model: NnueTronModel, scale=256):

        super().__init__()
        self.raw_model = model

        self.scale = scale

        self.embed_weights = torch.round(model.embedding.weight * scale).to(torch.int32)

        self.linear_weights = torch.round(model.fc1.weight * scale).to(torch.int32)
        self.linear_bias = torch.round(model.fc1.bias * scale * scale).to(torch.int32)

    def run_inference_acc(self, acc) -> float:

        # 2. Clamp to [0, scale]
        acc = torch.clamp(acc, 0, self.scale)
        # print(f"After clamp: {acc.sum().item()/ 1024=}")

        acc = acc.to(dtype=torch.int32)

        # print(f"After int32 cast: {acc.sum().item()/ 1024=}")

        # 3. Linear layer in integer domain
        #    (1 x acc_dim) @ (acc_dim) -> scalar
        y_int = (self.linear_weights @ acc) + self.linear_bias  # still int32

        # print(f"After linear: {y_int.sum().item()/ 1024 / 1024=}")

        # 4. Rescale back to float
        y = y_int.float() / (self.scale * self.scale)

        return y.item()

    def initilize_acc(self, pov_game_state):

        indices = self.raw_model.get_active_indices(pov_game_state)
        # 1. Sum embeddings (int accumulator)
        acc = self.embed_weights[indices].sum(dim=0)  # [acc_dim], int32

        return acc

    def run_inference(self, pov_game_state: PovGameState) -> float:

        acc = self.initilize_acc(pov_game_state)

        return self.run_inference_acc(acc)

    def update_acc(
        self,
        prev_acc,
        hero_index: int,
        opponent_index: int,
        prev_game_state,
        new_game_state,
    ):

        prev_hero = prev_game_state.players[hero_index]
        prev_oppo = prev_game_state.players[opponent_index]

        new_hero = new_game_state.players[hero_index]
        new_oppo = new_game_state.players[opponent_index]

        subtract_indices = [
            self.raw_model.emb_idx_hero_head(prev_hero.idx),
            self.raw_model.emb_idx_opponent_head(prev_oppo.idx),
        ]

        add_indices = [
            self.raw_model.emb_idx_wall(new_hero.idx),
            self.raw_model.emb_idx_wall(new_oppo.idx),
            self.raw_model.emb_idx_hero_head(new_hero.idx),
            self.raw_model.emb_idx_opponent_head(new_oppo.idx),
        ]

        acc = (
            prev_acc
            - self.embed_weights[subtract_indices].sum(dim=0)
            + self.embed_weights[add_indices].sum(dim=0)
        )

        return acc

    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:
        raise RuntimeError("Quantized NNUE is not used for training.")
