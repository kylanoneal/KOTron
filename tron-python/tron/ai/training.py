from collections import OrderedDict
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

import tron
from tron.game import  GameState, GameStatus, Player
from tron.ai.tron_model import TronModel, HeroGameState


# NOTE: Where does this belong?
def get_weights_sum_of_squares(model):
    device = next(model.parameters()).device
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    return total_sum_of_squares.item()

def print_state_and_sos(model, decimals=6):
    # """Print each entry in state_dict and the sum of squares of learnable params."""
    # # 1) Raw / shaped state_dict (optional: huge tensors truncated by repr)
    # for k, v in model.state_dict().items():
    #     print(f"{k:40s} shape={tuple(v.shape)} dtype={v.dtype}")

    # 2) Sum of squares per parameter tensor
    sos = OrderedDict()
    with torch.no_grad():
        for name, p in model.named_parameters():
            val = (p.detach() ** 2).sum().item()
            sos[name] = round(val, decimals)

    # Pretty print
    print("\nSum of squares (weights/biases):")
    for k, v in sos.items():
        print(f"{k:40s} {v}")

    print(f"\nTOTAL: {round(sum(sos.values()), decimals)}")
    return sos


def collate_fn(x):
    inputs, labels = zip(*x)
    labels = torch.tensor(labels)

    return inputs, labels



def affine_aug(game_state: GameState):

    grid = game_state.grid.copy()
    player_grid = np.zeros_like(grid, dtype=np.uint8)

    for i, player in enumerate(game_state.players):
        player_grid[player.row][player.col] = i + 1

    do_flip = random.random() > 0.5

    if do_flip:

        grid = np.fliplr(grid)
        player_grid = np.fliplr(player_grid)
 

    n_rot_90 = random.randrange(0,4)

    grid = np.rot90(grid, k=n_rot_90)
    player_grid = np.rot90(player_grid, k=n_rot_90)

    transformed_players = []

    for i, player in enumerate(game_state.players):
        ind = np.argwhere(player_grid == i + 1)
        assert len(ind) == 1 and len(ind[0]) == 2
        row, col = ind[0]

        transformed_players.append(Player(row, col, player.can_move))

    return GameState(grid, tuple(transformed_players))

def make_dataloader(
    game_data: list[list[GameState]],
    batch_size: int,
    shuffle: bool = True,
    include_ties = True,
    do_affine = True,
    keep_rate = 0.5
) -> DataLoader:
    
    dataset = []

    for game_states in game_data:

        terminal_status = tron.get_status(game_states[-1])

        if not include_ties and terminal_status.status == GameStatus.TIE:
            continue

        assert not terminal_status.status == GameStatus.IN_PROGRESS

        if terminal_status.status == GameStatus.WINNER:
            assert terminal_status.winner_index is not None
            assert 0 <= terminal_status.winner_index < 2


        # TODO: "Think about how the first couple moves of the game should be represented"
        # TODO: Add rotation augmentation

        # game_progs = [
        #     turn_index / (len(game_states) - 1)
        #     for turn_index in range(len(game_states))
        # ]

        # NOTE: Assumes 2 players
        for player_index in range(2):
                
            # NOTE: DONT INCLUDE TERMINAL STATE!!!

            num_active_turns = len(game_states) - 1
            for i, game_state in enumerate(game_states[:-1]):

            #for game_state, game_prog in zip(game_states, game_progs):

                assert len(game_state.players) == 2
                assert tron.get_status(game_state).status == GameStatus.IN_PROGRESS

                # if game_prog < 0.7:
                #     print(f"Skipping game prog of : {game_prog}")
                #     continue

                if terminal_status.status == GameStatus.WINNER:
                    # eval = (
                    #     game_prog
                    #     if terminal_status.winner_index == player_index
                    #     else -game_prog
                    # )

                    dist_from_end = num_active_turns - (i + 1)

                    eval = 10 * (0.9 ** dist_from_end)

                    if terminal_status.winner_index != player_index:
                        eval *= -1
                else:
                    eval = 0.0

                if random.random() < keep_rate:

                    game_state_to_add = affine_aug(game_state) if do_affine else game_state

                    dataset.append((HeroGameState(game_state_to_add, hero_index=player_index), np.float32(eval)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_loop(
    model: TronModel,
    train_dataloader: DataLoader,
    optimizer,
    criterion,
    device,
    epochs=1,
):

    model.train()

    cum_loss = 0.0
    cum_magnitude = 0.0

    # Iterate through the DataLoader in a training loop
    for epoch in range(epochs):

        cum_epoch_loss = 0.0
        cum_epoch_magnitude = 0.0

        for _inputs, _labels in train_dataloader:
            # Move data to GPU if available

            inputs = model.get_model_input(_inputs).to(device)
            labels = _labels.to(device)

            # if np.random.random() < 0.01:
            #     print(f"Mean labels: {labels.mean()}")
            #     print(f"mean Abs labels: {labels.abs().mean()}\n")


            optimizer.zero_grad()

            # Forward pass, loss computation, backward pass, optimizer step, etc.
            outputs = model(inputs)

            cum_epoch_magnitude += torch.sum(torch.abs(outputs)).item() / len(outputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            cum_epoch_loss += loss.item()

        epoch_avg_loss = cum_epoch_loss / len(train_dataloader)
        epoch_avg_magnitude = cum_epoch_magnitude / len(train_dataloader)

        cum_loss += epoch_avg_loss
        cum_magnitude += epoch_avg_magnitude

    average_loss = cum_loss / epochs
    average_magnitude = cum_magnitude / epochs
    return average_loss, average_magnitude
