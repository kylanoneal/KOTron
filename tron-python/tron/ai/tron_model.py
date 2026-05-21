import torch
import random

from abc import ABC, abstractmethod

from tron.game import GameState, PovGameState


class TronModel(torch.nn.Module, ABC):

    @abstractmethod
    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:
        pass

    @abstractmethod
    def run_inference(self, pov_game_state: PovGameState) -> float:
        pass

    def run_inference_acc(self, acc: torch.Tensor) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement run_inference_acc"
        )
    
    def update_acc(
        self,
        prev_acc: torch.Tensor,
        prev_game_state: GameState,
        next_pov_game_state: PovGameState,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement update_acc"
        )




class RandomTronModel(TronModel):
    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:
        raise NotImplementedError()

    def run_inference(self, pov_game_state: PovGameState) -> float:

        # Should be deterministic across Python process restarts in the same environment
        seed = hash(pov_game_state)
        rng = random.Random(seed)
        return rng.normalvariate(0, 1)


    # Could do this or similar for maximum reproducibility:

    # def _stable_seed_from_pov_game_state(pov: PovGameState) -> int:
    #     game = pov.game_state

    #     key = (
    #         game.num_rows,
    #         game.num_cols,
    #         game.board,
    #         tuple((p.idx, p.can_move) for p in game.players),
    #         pov.hero_index,
    #         pov.opponent_index,
    #     )

    #     key_bytes = repr(key).encode("utf-8")
    #     digest = hashlib.sha256(key_bytes).digest()

    #     return int.from_bytes(digest[:8], byteorder="big", signed=False)


# For searches that go all the way to the end of a game, so never should
# be calling it's methods
class DummyTronModel(TronModel):
    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:
        raise RuntimeError("Can't train dummy model")

    def run_inference(self, pov_game_state: PovGameState) -> float:
        raise RuntimeError("Can't run inference on dummy model")