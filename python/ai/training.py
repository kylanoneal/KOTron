import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from game import tron
from game.tron import GameState, GameStatus
from ai.tron_model import TronModelAbstract


# NOTE: Where does this belong?
def get_weights_sum_of_squares(model):
    device = next(model.parameters()).device
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    return total_sum_of_squares.item()


def make_dataloader(
    game_data: list[list[GameState]],
    model: TronModelAbstract,
    batch_size: int,
    shuffle: bool = True,
    include_ties = True
) -> DataLoader:

    dataset = []

    for game_states in game_data:

        terminal_status = tron.get_status(game_states[-1])

        if not include_ties and terminal_status.status == GameStatus.TIE:
            continue

        assert not terminal_status.status == GameStatus.IN_PROGRESS

        # TODO: "Think about how the first couple moves of the game should be represented"
        # TODO: Add rotation augmentation

        game_progs = [
            turn_index / (len(game_states) - 1)
            for turn_index in range(len(game_states))
        ]

        # NOTE: Assumes 2 players
        for player_index in range(2):

            model_inputs = model.get_model_input(game_states, player_index)

            assert len(game_progs) == model_inputs.shape[0]

            for game_prog, model_input in zip(game_progs, model_inputs):

                # if game_prog < 0.7:
                #     print(f"Skipping game prog of : {game_prog}")
                #     continue

                if terminal_status.winner_index is not None:
                    eval = (
                        game_prog
                        if terminal_status.winner_index == player_index
                        else -game_prog
                    )
                else:
                    eval = 0.0

                dataset.append((model_input, np.float32(eval)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_loop(
    model: torch.nn.Module,
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

        for inputs, labels in train_dataloader:
            # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

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
