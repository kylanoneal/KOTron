import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import json

device = "cuda" if torch.cuda.is_available() else "cpu"

import json

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


def get_processed_grid(grid, head):
    binary_grid = [[1 if element != 0 else 0 for element in row] for row in grid]
    binary_grid[head[0]][head[1]] = -1
    # print("Binary GRid:", binary_grid)
    # Create a new 42x42 list filled with padding value 0
    padded_grid = [[1] * 42 for _ in range(42)]

    for i in range(40):
        for j in range(40):
            padded_grid[i + 1][j + 1] = binary_grid[i][j]

    return padded_grid

def get_model_input_from_game_state(game_state, player_num):
    head = game_state.players[player_num].head
    head_x, head_y = head
    padded_grid = get_processed_grid(game_state.collision_table, head)
    #print("HEAD X AND Y:", head_x, head_y)
    #print("Collision table at those indexes:", game_state.collision_table[head_x][head_y])
    #print("at inverse indexes:", game_state.collision_table[head_y][head_x])
    padded_grid[head_x][head_y] = -1
    tensor_grid = torch.tensor(padded_grid, dtype=torch.float32).to(device)
    tensor_grid = tensor_grid.view(1, 42, 42)
    tensor_grid = tensor_grid.unsqueeze(1)
    return tensor_grid

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
        self.fc1 = nn.Linear(30*10*10, 1000)
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

    torch.save(model, 'model.pth')
    class_names = ['Up', 'Right', 'Down', 'Left']
    plot_confusion_matrix(all_labels, all_preds, class_names)
