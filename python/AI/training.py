import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from game.ko_tron import KOTron, GameStatus
from AI.algos import TronModelAbstract

# NOTE: Where does this belong?
def get_weights_sum_of_squares(model):
    device = next(model.parameters()).device
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    return total_sum_of_squares.item()


def make_dataloader(
    games_list: list[list[KOTron]], model: TronModelAbstract, batch_size: int, shuffle: bool = True
) -> DataLoader:

    dataset = []

    for game_states in games_list:

        last_game_state = game_states[-1]

        assert not last_game_state.status == GameStatus.IN_PROGRESS

        if not last_game_state.status == GameStatus.TIE:
            if last_game_state.status == GameStatus.P1_WIN:
                winning_player_index = 0
            elif last_game_state.status == GameStatus.P2_WIN:
                winning_player_index = 1
            else:
                raise NotImplementedError()
        else:
            winning_player_index = None

        game_progs = [
            turn_index / (len(game_states) - 1) for turn_index in range(len(game_states))
        ]

        # NOTE: Assumes 2 players
        for player_index in range(2):

            model_inputs = model.get_model_input(game_states, player_index)

            for game_prog, model_input in zip(game_progs, model_inputs):
                if winning_player_index is not None:
                    eval = (
                        game_prog
                        if winning_player_index == player_index
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

            optimizer.zero_grad()

            # Forward pass, loss computation, backward pass, optimizer step, etc.
            outputs = model(inputs)

            cum_epoch_magnitude += (torch.sum(torch.abs(outputs)).item() / len(outputs))

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            cum_epoch_loss += loss.item()

        epoch_avg_loss = cum_epoch_loss / len(train_dataloader)
        epoch_avg_magnitude = cum_epoch_magnitude / len(train_dataloader)
        
        cum_loss += epoch_avg_loss
        cum_magnitude += epoch_avg_magnitude
        print(f"Epoch: {epoch + 1}, Avg. Train Loss: {epoch_avg_loss:.4f}")

    average_loss = cum_loss / epochs
    average_magnitude = cum_magnitude / epochs
    return average_loss, average_magnitude

