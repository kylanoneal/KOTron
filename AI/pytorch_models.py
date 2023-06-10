import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

import json

from typing import Callable, Optional

from AI.model_architectures import *

MODEL_INPUT_DIMENSION = 40

device = "cpu" if torch.cuda.is_available() else "cpu"


def collect_feat(move_list):
    # Change racer values to 1

    processed_moves = []
    for grid, head, dir in move_list:
        padded_grid = get_processed_grid(grid, head)

        # print("Fully processed grid:", padded_grid)
        # print("One count:", one_count, "two count:", two_count)
        processed_moves.append((padded_grid, head, dir))

    print("Total moves collected:", len(processed_moves))
    with open("kylan_moves_vs_troy.json", "w") as file:
        json.dump(processed_moves, file)


def get_processed_grid(game, player_num):
    # print("Binary GRid:", binary_grid)
    # Create a new 42x42 list filled with padding value 0
    padded_body_grid = [[1] * (MODEL_INPUT_DIMENSION) for _ in range(MODEL_INPUT_DIMENSION)]

    # padded_head_grid = [[0] * (MODEL_INPUT_DIMENSION) for _ in range(MODEL_INPUT_DIMENSION)]

    pad_offset = (MODEL_INPUT_DIMENSION - game.dimension) // 2

    # Create body grid
    for i in range(game.dimension):
        for j in range(game.dimension):
            val = 0 if game.collision_table[i][j] == 0 else 1
            padded_body_grid[i + pad_offset][j + pad_offset] = val

    # Create head grid
    for curr_player_num, (x, y) in enumerate(game.get_heads()):
        # print("Head value on body grid:", padded_body_grid[x+pad_offset][y+pad_offset])
        # print("at inverse val on body grid:", padded_body_grid[y+pad_offset][x+pad_offset])
        # print("Head value on collision grid:", game.collision_table[x][y])
        # print("at inverse val on collision grid:", game.collision_table[y][x])
        padded_body_grid[x + pad_offset][y + pad_offset] = 100 if player_num == curr_player_num else -100

    # print("BODY GRID: \n\n")
    # print_readable_grid(padded_body_grid)
    # print("HEAD GRID:\n\n")
    # print_readable_grid(padded_head_grid)
    return padded_body_grid


def print_readable_grid(grid):
    transposed_grid = list(zip(*grid))

    # Print the transposed grid
    for row in transposed_grid:
        print(row)


def get_relevant_info_from_game_state(game_state):
    return game_state.collision_table, game_state.get_heads()


def get_model_evaluation(decay_fn: Callable[[float], float], game_progress: float, won_lost_tied: int):
    return torch.tensor(won_lost_tied * decay_fn(game_progress), dtype=torch.float32)


def get_model_input_from_raw_info(grid, heads, player_num, model_type: Optional[type] = None, is_part_of_batch=False):
    if len(grid) != MODEL_INPUT_DIMENSION:
        raise NotImplementedError
    else:
        processed_grid = np.where(np.array(grid) == 0, 0, 1)

        for p_num, (x, y) in enumerate(heads):
            processed_grid[x][y] = 10 if player_num == p_num else -10

    tensor_output = torch.tensor(processed_grid, dtype=torch.float32).unsqueeze(0).to(device)
    if not is_part_of_batch:
        tensor_output = tensor_output.unsqueeze(0)

    if model_type is EvaluationAttentionConvNet:
        return tensor_output, torch.tensor(heads[player_num])
    else:
        return tensor_output


# def get_model_input_from_game_state(game_state, player_num):
#     # Old and jank
#     padded_grids = get_processed_grid(game_state, player_num)
#
#     tensor_grids = torch.tensor(padded_grids, dtype=torch.float32).to(device)
#     tensor_grids = tensor_grids.view(1, MODEL_INPUT_DIMENSION, MODEL_INPUT_DIMENSION)
#     tensor_grids = tensor_grids.unsqueeze(0)
#
#     return tensor_grids


def get_imitation_model_inference(model, game_state, player_num):
    model_output = model(get_model_input_from_game_state(game_state, player_num))
    direction = torch.argmax(model_output).item()
    return direction


def get_reinforcement_model_inference(model, game_state, player_num):
    action_probs, eval = model(get_model_input_from_game_state(game_state, player_num))
    direction = torch.argmax(action_probs).item()
    return direction


def get_dataloader():
    with open('kylan_moves_vs_troy.json', 'r') as file:
        # Load the JSON data into a Python object
        data = json.load(file)

    torch_grids = [torch.tensor(grid, dtype=torch.float32).to(device) for grid, pos, label in data]
    torch_grids_conv = [grid.view(1, 42, 42) for grid in torch_grids]

    head_positions = [torch.tensor(pos, dtype=torch.int32).to(device) for grid, pos, label in data]
    attention_input = [(conv_grid, pos) for conv_grid, pos in zip(torch_grids_conv, head_positions)]

    labels = [torch.tensor(label).to(device) for grid, pos, label in data]

    return DataLoader(list(zip(torch_grids_conv, labels)), batch_size=32, shuffle=True)


def train_loop(model, data_loader, epochs=10, lr=0.005):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for batch, (X, y) in enumerate(data_loader):
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 2000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(data_loader.dataset)}]")


def evaluate_model(model, data_loader):
    model.eval()
    correct_preds, total_preds = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += y.size(0)
            correct_preds += (predicted == y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = 100 * correct_preds / total_preds
    return accuracy, all_preds, all_labels


def plot_confusion_matrix(labels, preds, class_names):
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)

    # Create a heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5, 2.5, 3.5], class_names)
    plt.yticks([0.5, 1.5, 2.5, 3.5], class_names)
    plt.show()


def get_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


class DirectionNetConv(nn.Module):
    def __init__(self):
        super(DirectionNetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(30 * 10 * 10, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    data_loader = get_dataloader()
    # Train the model
    model = DirectionNetConv().to(device)
    train_loop(model, data_loader, epochs=50, lr=0.001)

    accuracy, all_preds, all_labels = evaluate_model(model, data_loader)
    print(f'Accuracy of the model on test data: {accuracy}%')

    torch.save(model, 'old-monte-models/model.pth')
    class_names = ['Up', 'Right', 'Down', 'Left']
    plot_confusion_matrix(all_labels, all_preds, class_names)
