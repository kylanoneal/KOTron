import random
import torch
import numpy as np
from abc import ABC, abstractmethod


from game.ko_tron import KOTron, DirectionUpdate


class TronModelAbstract(ABC):

    @abstractmethod
    def get_model_input(
        self, game_states: list[KOTron], player_index: int
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def run_inference(
        self, game_states: list[KOTron], player_index: int
    ) -> list[float]:
        pass


class StandardTronModel(TronModelAbstract):
    def __init__(self, model, device):

        self.device = device
        self.model = model

    def get_model_input(
        self, game_states: list[KOTron], player_index: int
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

        tensor_output = torch.tensor(
            combined_array, dtype=torch.float32
        ).to(self.device)

        return tensor_output

    def run_inference(self, game_states: list[KOTron], player_index: int) -> np.ndarray:

        # NOTE: Maybe should have a way to set requries_grad = False for infernce time
        model_input = self.get_model_input(
            game_states, player_index
        )

        self.model.eval()

        return self.model(model_input).detach().cpu().numpy()


def choose_direction_model_naive(
    game, player_index, model: TronModelAbstract
) -> DirectionUpdate:

    if not game.players[player_index].can_move:
        # Maybe return None instead?
        return DirectionUpdate(
            direction=game.players[player_index].direction, player_index=player_index
        )
    else:
        possible_directions = KOTron.get_possible_directions(game, player_index)

        if len(possible_directions) == 0:
            # Maybe return None instead?
            return DirectionUpdate(
                direction=game.players[player_index].direction,
                player_index=player_index,
            )
        else:
            game_states_to_eval = [
                KOTron.next(game, [DirectionUpdate(direction, player_index)])
                for direction in possible_directions
            ]

            evaluations = model.run_inference(game_states_to_eval, player_index)

            return DirectionUpdate(
                direction=possible_directions[np.argmax(evaluations)],
                player_index=player_index,
            )



def choose_direction_random(
    game, player_index
) -> DirectionUpdate:

    if not game.players[player_index].can_move:
        # Maybe return None instead?
        return DirectionUpdate(
            direction=game.players[player_index].direction, player_index=player_index
        )
    else:
        possible_directions = KOTron.get_possible_directions(game, player_index)

        if len(possible_directions) == 0:
            # Maybe return None instead?
            return DirectionUpdate(
                direction=game.players[player_index].direction,
                player_index=player_index,
            )
        else:
            return DirectionUpdate(direction = random.choice(possible_directions), player_index=player_index)

#      evaluations: Vec<f32> = model
#         .run_inference(&game_states_to_eval, player_index)
#         .expect("Failed to run inference.")

#     // Update direction to possible_moves[argmax(evaluations)]
#     return DirectionUpdate :
#         player_index: player_index,
#         direction: possible_moves[argmax(&evaluations)],

#  else :
#     // Return current direction if it can't move
#     DirectionUpdate :
#         player_index: player_index,
#         direction: game.players[player_index].direction,
