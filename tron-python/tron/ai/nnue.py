import torch
import torch.nn as nn

from tron.ai.tron_model import TronModel

from tron.game import GameState, PovGameState, get_wall_indices


# --- 2. Define the efficient‐updatable net ---
class NnueTronModel(TronModel):
    def __init__(self, num_rows: int, num_cols: int, acc_dim: int):
        super().__init__()
        # Embedding table: feature → acc_dim vector
        self.embedding = nn.Embedding(num_rows * num_cols * 3, acc_dim)
        # Tiny MLP on top of the accumulator
        self.fc1 = nn.Linear(acc_dim, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc_value = nn.Linear(16, 1)

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols

    def initialize_acc(self, pov_game_state: PovGameState):
        """
        Build accumulator from scratch by summing embeddings

        """
        active_indices = torch.tensor(
            self.get_active_indices(pov_game_state), dtype=torch.long
        )
        acc = self.embedding(active_indices).sum(dim=0)

        return acc

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

        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=1.0)

        x = self.fc2(x)
        x = torch.clamp(x, min=0.0, max=1.0)

        out = self.fc_value(x)

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

        hero_emb_index = self.emb_idx_hero_head(
            game_state.players[pov_game_state.hero_index].idx
        )

        opponent_emb_index = self.emb_idx_opponent_head(
            game_state.players[pov_game_state.opponent_index].idx
        )

        indices = [hero_emb_index, opponent_emb_index] + get_wall_indices(
            pov_game_state.game_state
        )

        return indices

    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:

        accs = []

        for pov_game_state in pov_game_states:

            accs.append(self.initialize_acc(pov_game_state))

        return torch.stack(accs)

    def run_inference(self, pov_game_state: PovGameState) -> float:

        with torch.no_grad():
            acc = self.initialize_acc(pov_game_state)
            output = self(acc)
            return output.item()


# TODO: Figure out most efficient dtypes to use
class QuantizedNnueTronModel(TronModel):

    def __init__(self, model: NnueTronModel, scale=256):

        assert isinstance(model, NnueTronModel)
        raise NotImplementedError("We have more than 2 layers now")

        super().__init__()
        self.raw_model = model

        self.scale = scale

        self.embed_weights = torch.round(model.embedding.weight * scale).to(dtype=torch.int64)

        self.linear_weights = torch.round(model.fc1.weight * scale).to(dtype=torch.int64)
        self.linear_bias = torch.round(model.fc1.bias * scale * scale).to(dtype=torch.int64)

        assert self.embed_weights.dtype == torch.int64
        assert self.linear_weights.dtype == torch.int64
        assert self.linear_bias.dtype == torch.int64

    def run_inference_acc(self, acc) -> float:

        assert acc.dtype == torch.int64
        # 2. Clamp to [0, scale]
        acc = torch.clamp(acc, 0, self.scale)
        # print(f"After clamp: {acc.sum().item()/ 1024=}")

        # 3. Linear layer in integer domain
        #    (1 x acc_dim) @ (acc_dim) -> scalar
        y_int = (self.linear_weights @ acc) + self.linear_bias 

        # print(f"After linear: {y_int.sum().item()/ 1024 / 1024=}")

        # 4. Rescale back to float
        y = y_int.float() / (self.scale * self.scale)

        return y.item()

    def initialize_acc(self, pov_game_state):

        indices = self.raw_model.get_active_indices(pov_game_state)
        # 1. Sum embeddings (int accumulator)
        acc = self.embed_weights[indices].sum(dim=0)  # [acc_dim], int64


        assert acc.dtype == torch.int64

        return acc

    def run_inference(self, pov_game_state: PovGameState) -> float:

        acc = self.initialize_acc(pov_game_state)

        return self.run_inference_acc(acc)

    def update_acc(
        self,
        prev_acc: torch.tensor,
        prev_game_state: GameState,
        next_pov_game_state: PovGameState,
    ):

        assert prev_acc.dtype == torch.int64

        hero_index = next_pov_game_state.hero_index
        opponent_index = next_pov_game_state.opponent_index

        next_game_state = next_pov_game_state.game_state

        prev_hero = prev_game_state.players[hero_index]
        prev_oppo = prev_game_state.players[opponent_index]

        new_hero = next_game_state.players[hero_index]
        new_oppo = next_game_state.players[opponent_index]


        # NOTE: will double count a wall if the player is inactive
        assert len(next_pov_game_state.game_state.players) == 2
        assert new_hero.can_move
        assert new_oppo.can_move

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
