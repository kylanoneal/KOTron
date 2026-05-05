import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tron.game import GameState, Player, PovGameState, from_bitboard


class TronModel(torch.nn.Module, ABC):

    @abstractmethod
    def get_model_input(self, pov_game_states: list[PovGameState]) -> torch.Tensor:
        pass

    @abstractmethod
    def run_inference(self, pov_game_state: PovGameState) -> float:
        pass

    def run_inference_acc(self, acc: torch.tensor) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement run_inference_acc"
        )
    
    def update_acc(
        self,
        prev_acc: torch.tensor,
        prev_game_state: GameState,
        next_pov_game_state: PovGameState,
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement update_acc"
        )


class RandomTronModel(TronModel):
    def get_model_input(self, pov_game_states: list[PovGameState]):
        raise NotImplementedError()

    def run_inference(self, pov_game_state: PovGameState) -> float:

        hash_tup = (pov_game_state.game_state, pov_game_state.hero_index)

        seed = int(hash(hash_tup))
        rng = random.Random(seed)
        return rng.normalvariate(0, 1)



