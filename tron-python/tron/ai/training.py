from collections import OrderedDict
import torch
import random
import numpy as np
from typing import Optional
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader

import tron
from tron.game import (
    GameState,
    GameState2D,
    GameStatus,
    Player,
    Direction,
    from_2d_game_state,
    from_bitboard,
)
from tron.ai.tron_model import TronModel, PovGameState

from tron.enums import PovGameResult


@dataclass(frozen=True)
class LabeledExample:
    pov_game_state: PovGameState
    label: float


@dataclass(frozen=True)
class ModelExample:
    labeled_example: LabeledExample
    prediction: float


@dataclass(frozen=True)
class TrainingResult:
    model_examples: tuple[ModelExample]
    avg_loss: float
    avg_prediction_magnitude: float


@dataclass(frozen=True)
class TrainValSplit:
    train_examples: tuple[LabeledExample]
    val_examples: tuple[LabeledExample]


# NOTE: Where does this belong?
def get_weights_sum_of_squares(model):
    device = next(model.parameters()).device
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    return total_sum_of_squares.item()


def get_sos_info(model):

    # 2) Sum of squares per parameter tensor
    sos_dict = OrderedDict()
    with torch.no_grad():
        for name, p in model.named_parameters():
            val = (p.detach() ** 2).sum().item()
            sos_dict[name] = round(val, 2)

    total_sos = round(sum(sos_dict.values(), 2))

    return sos_dict, total_sos


def make_batches(
    items: tuple[LabeledExample],
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
) -> tuple[tuple[LabeledExample]]:

    if shuffle:
        rng = random.Random(seed)

        items = list(items)
        rng.shuffle(items)

        items = tuple(items)

    return tuple([items[i : i + batch_size] for i in range(0, len(items), batch_size)])


def make_k_folds(
    items: tuple[LabeledExample], # TODO: Fix this type hint once you decide on a struct here
    k: int,
    shuffle: bool = True,
    seed: int = 0,
) -> tuple[TrainValSplit]:

    if k <= 1:
        raise ValueError("k must be greater than 1")

    if k > len(items):
        raise ValueError("k cannot be greater than len(items)")

    rng = random.Random(seed)

    items = list(items)
    if shuffle:
        rng.shuffle(items)

    fold_sizes = [len(items) // k] * k

    # Distribute remainder across the first folds
    for i in range(len(items) % k):
        fold_sizes[i] += 1

    folds = []
    start = 0

    for fold_size in fold_sizes:
        end = start + fold_size
        val_items = items[start:end]
        train_items = items[:start] + items[end:]

        folds.append(TrainValSplit(train_items, val_items))

        start = end

    return tuple(folds)


def get_label_magnitude(steps_until_terminal: int):

    assert steps_until_terminal >= 1
    assert isinstance(steps_until_terminal, int)

    return 0.8 ** (steps_until_terminal - 1)



# def make_dataset(
#     game_data: tuple[tuple[GameState]],
#     batch_size: Optional[int] = None,
#     shuffle: bool = True,
#     include_ties=True,
#     do_affine=True,
#     keep_rate=0.5,
# ) -> tuple[tuple[LabeledExample]]:

#     dataset = []

#     for game_states in game_data:

#         terminal_status = tron.get_status(game_states[-1])

#         if not include_ties and terminal_status.status == GameStatus.TIE:
#             continue

#         assert not terminal_status.status == GameStatus.IN_PROGRESS

#         if terminal_status.status == GameStatus.WINNER:
#             assert terminal_status.winner_index is not None
#             assert 0 <= terminal_status.winner_index < 2


#         # NOTE: Assumes 2 players
#         for player_index in range(2):

#             opponent_index = 0 if player_index == 1 else 1

#             # NOTE: DONT INCLUDE TERMINAL STATE!!!

#             num_active_turns = len(game_states) - 1
#             for i, game_state in enumerate(game_states[:-1]):


#                 assert len(game_state.players) == 2
#                 assert tron.get_status(game_state).status == GameStatus.IN_PROGRESS

#                 if terminal_status.status == GameStatus.WINNER:

#                     steps_until_terminal = num_active_turns - i

#                     assert steps_until_terminal >= 1

#                     label = get_label_magnitude(steps_until_terminal)

#                     if terminal_status.winner_index != player_index:
#                         label *= -1
#                 else:
#                     label = 0.0

#                 if random.random() < keep_rate:

#                     game_state_to_add = (
#                         GameState.transform(
#                             game_state,
#                             do_lr_flip=random.random() > 0.5,
#                             n_rot_90=random.randrange(0, 4),
#                         )
#                         if do_affine
#                         else game_state
#                     )

#                     dataset.append(
#                         (
#                             LabeledExample(
#                                 pov_game_state=PovGameState(
#                                     game_state_to_add,
#                                     hero_index=player_index,
#                                     opponent_index=opponent_index,
#                                 ),
#                                 label=label,
#                             )
#                         )
#                     )

#     if shuffle:
#         random.shuffle(dataset)

#     if batch_size is not None:
#         dataset = make_batches(
#             dataset, batch_size, shuffle=False
#         )  # Shuffle already handled

#     return tuple(dataset)


# def train(
#     model: TronModel,
#     dataset: tuple[tuple[LabeledExample]],
#     optimizer,
#     criterion,
#     device=torch.device("cpu"),
#     epochs=1,
# ):

#     model.train()

#     # TODO: Instead of computing stats like avg_loss in train loop, just
#     # provide per example loss (for example) and let something else handle
#     # the stats

#     cum_loss = 0.0
#     cum_magnitude = 0.0

#     model_examples: tuple[tuple[ModelExample]] = []

#     # Iterate through the DataLoader in a training loop
#     for epoch in range(epochs):

#         cum_epoch_loss = 0.0
#         cum_epoch_magnitude = 0.0

#         for batch in dataset:

#             inputs = model.get_model_input([ex.pov_game_state for ex in batch])
#             labels = torch.tensor([ex.label for ex in batch])

#             if device.type == "cuda":
#                 # Move data to GPU if available

#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#             # if np.random.random() < 0.01:
#             #     print(f"Mean labels: {labels.mean()}")
#             #     print(f"mean Abs labels: {labels.abs().mean()}\n")

#             optimizer.zero_grad()

#             # Forward pass, loss computation, backward pass, optimizer step, etc.
#             outputs = model(inputs)

#             for train_example, o in zip(batch, outputs):

#                 model_examples.append(ModelExample(train_example, o.item()))

#             cum_epoch_magnitude += torch.sum(torch.abs(outputs)).item() / len(outputs)

#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             cum_epoch_loss += loss.item()

#         epoch_avg_loss = cum_epoch_loss / len(dataset)
#         epoch_avg_magnitude = cum_epoch_magnitude / len(dataset)

#         cum_loss += epoch_avg_loss
#         cum_magnitude += epoch_avg_magnitude

#     average_loss = cum_loss / epochs
#     average_magnitude = cum_magnitude / epochs

#     return TrainingResult(
#         model_examples,
#         avg_loss=average_loss,
#         avg_prediction_magnitude=average_magnitude,
#     )

def train(
    model: TronModel,
    train_loader: DataLoader,
    optimizer,
    criterion,
    epochs: int = 1,
) -> TrainingResult:
    model.train()

    cum_loss = 0.0
    cum_magnitude = 0.0

    # Important: if your DataLoader only gives tensors, you no longer have
    # the original LabeledExample objects available here.
    model_examples: list[ModelExample] = []

    for epoch in range(epochs):
        cum_epoch_loss = 0.0
        cum_epoch_magnitude = 0.0
        num_epoch_examples = 0

        for inputs, labels in train_loader:

            batch_size = inputs.shape[0]

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # If criterion returns mean batch loss, multiply by batch_size
            # so the final average is truly per-example.
            cum_epoch_loss += loss.item() * batch_size

            cum_epoch_magnitude += torch.sum(torch.abs(outputs)).item()

            num_epoch_examples += batch_size

        epoch_avg_loss = cum_epoch_loss / num_epoch_examples
        epoch_avg_magnitude = cum_epoch_magnitude / num_epoch_examples

        cum_loss += epoch_avg_loss
        cum_magnitude += epoch_avg_magnitude

    average_loss = cum_loss / epochs
    average_magnitude = cum_magnitude / epochs

    return TrainingResult(
        tuple(model_examples),
        avg_loss=average_loss,
        avg_prediction_magnitude=average_magnitude,
    )

def validate(
    model: TronModel,
    validation_loader: DataLoader,
    criterion,
) -> float:
    model.eval()

    cum_loss = 0.0
    num_examples = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.shape[0]

            # If criterion returns mean loss over the batch, scale back up
            # so the final average is per-example.
            cum_loss += loss.item() * batch_size
            num_examples += batch_size

    avg_loss = cum_loss / num_examples

    return avg_loss