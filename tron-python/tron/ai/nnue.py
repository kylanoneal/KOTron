import torch
import torch.nn as nn

from tron.ai.tron_model import TronModel

from tron.game import GameState, PovGameState, get_wall_indices


# --- 2. Define the efficient‐updatable net ---
class NnueTronModel(TronModel):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        acc_dim: int,
        fc_layer_neuron_counts:tuple, # (8, 16),
        clamp_val: float #= 1.0,
    ):
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_cells = num_rows * num_cols

        self.clamp_val = clamp_val

        # Real features:
        #   [0, num_cells)              -> walls
        #   [num_cells, 2*num_cells)    -> hero head
        #   [2*num_cells, 3*num_cells)  -> opponent head
        #
        # Plus one padding feature.
        num_features = (self.num_cells * 3) + 1
        self.padding_idx = num_features - 1

        # Can only be num cells plus 2 (all walls filled and 2 players)
        self.max_features = self.num_cells + 2

        # EmbeddingBag directly sums variable-length feature sets.
        self.embedding = nn.EmbeddingBag(
            num_embeddings=num_features,
            embedding_dim=acc_dim,
            padding_idx=self.padding_idx,
            mode="sum",
        )
        # Tiny MLP on top of the accumulator

        self.fc_layers = nn.ModuleList()

        prev_neuron_count = acc_dim
        for neuron_count in fc_layer_neuron_counts:
            self.fc_layers.append(nn.Linear(prev_neuron_count, neuron_count))
            prev_neuron_count = neuron_count

        self.fc_value = nn.Linear(prev_neuron_count, 1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:

        # indices shape: [batch_size, max_features]
        # x shape: [batch_size, acc_dim]
        x = self.embedding(indices)
        x = torch.clamp(x, min=0.0, max=self.clamp_val)

        for layer in self.fc_layers:

            x = layer(x)
            x = torch.clamp(x, min=0.0, max=self.clamp_val)

        out = self.fc_value(x)

        return out.squeeze(-1)

    def initialize_acc(self, pov_game_state: PovGameState):
        """
        Build accumulator from scratch by summing embeddings

        """

        raise NotImplementedError("Does this ever need to happen?")
        active_indices = torch.tensor(
            self.get_active_indices(pov_game_state), dtype=torch.long
        )
        acc = self.embedding(active_indices).sum(dim=0)

        return acc

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

    def get_model_input(self, pov_game_states: list[PovGameState]) -> tuple:

        inputs: list[int] = []

        for pov_game_state in pov_game_states:
            curr_active_indices = self.get_active_indices(pov_game_state)
            curr_pad_indices = [self.padding_idx] * (
                self.max_features - len(curr_active_indices)
            )

            input = curr_active_indices + curr_pad_indices

            assert len(input) == self.max_features

            torch_input = torch.tensor(input, dtype=torch.long)

            inputs.append(torch_input)

        return torch.stack(inputs)

    def run_inference(self, pov_game_state: PovGameState) -> float:

        self.eval()

        with torch.no_grad():

            model_input = self.get_model_input([pov_game_state])

            output = self(model_input)

            return output.detach().item()


# TODO: Figure out most efficient dtypes to use
class QuantizedNnueTronModel(TronModel):

    def __init__(self, model: NnueTronModel, scale=256):

        assert isinstance(model, NnueTronModel)
        raise NotImplementedError(
            "We have more than 2 layers now and using embedding bag"
        )

        super().__init__()
        self.raw_model = model

        self.scale = scale

        self.embed_weights = torch.round(model.embedding.weight * scale).to(
            dtype=torch.int64
        )

        self.linear_weights = torch.round(model.fc1.weight * scale).to(
            dtype=torch.int64
        )
        self.linear_bias = torch.round(model.fc1.bias * scale * scale).to(
            dtype=torch.int64
        )

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

        raise NotImplementedError(
            "Changed how running inference on NNUE's work, rethink this"
        )
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
