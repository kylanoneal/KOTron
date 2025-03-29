import torch
from torch import nn
import torch.nn.functional as F


# Stride == 1
class FastNet(nn.Module):

    def __init__(self, grid_dim=10):

        super(FastNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0)

        output_dim1 = self.conv_output_size(
            grid_dim, kernel_size=3, padding=0, stride=1
        )
        output_dim2 = self.conv_output_size(
            output_dim1, kernel_size=3, padding=0, stride=1
        )

        fc1_input_neurons = 10 * output_dim2**2

        print(f"Conv1 output size: {output_dim1}")
        print(f"Conv2 output size: {output_dim2}")
        print(f"FC1 input size: {fc1_input_neurons}")

        self.fc1 = nn.Linear(fc1_input_neurons, 50)  # input features for fc1
        self.fc_value = nn.Linear(50, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc_value(x).squeeze(1)

        return x


# Stride == 1
class EvaluationNetConv3OneStride(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(EvaluationNetConv3OneStride, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 10, kernel_size=5, stride=1, padding=0
        )  # changed stride to 2
        self.conv2 = nn.Conv2d(
            10, 20, kernel_size=3, stride=1, padding=1
        )  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 1)  # after first conv layer
        output_dim2 = self.conv_output_size(
            output_dim1, 3, 1, 1
        )  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2**2, 250)  # input features for fc1
        self.fc_value = nn.Linear(250, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        padded_game_grid = F.pad(
            x[:, 0:1], pad=(2, 2, 2, 2), mode="constant", value=1.0
        )
        padded_heads = F.pad(x[:, 1:], pad=(2, 2, 2, 2), mode="constant", value=0.0)
        concat = torch.concat((padded_game_grid, padded_heads), dim=1)


        x = F.relu(self.conv1(concat))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten the tensor

        x = F.relu(self.fc1(x))
        
        value_estimate = self.fc_value(x).squeeze(1)

        return value_estimate

# Stride == 1
class LeakyReLU(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(LeakyReLU, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 10, kernel_size=5, stride=1, padding=0
        )  # changed stride to 2
        self.conv2 = nn.Conv2d(
            10, 20, kernel_size=3, stride=1, padding=1
        )  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 1)  # after first conv layer
        output_dim2 = self.conv_output_size(
            output_dim1, 3, 1, 1
        )  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2**2, 250)  # input features for fc1
        self.fc_value = nn.Linear(250, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        padded_game_grid = F.pad(
            x[:, 0:1], pad=(2, 2, 2, 2), mode="constant", value=1.0
        )
        padded_heads = F.pad(x[:, 1:], pad=(2, 2, 2, 2), mode="constant", value=0.0)
        concat = torch.concat((padded_game_grid, padded_heads), dim=1)


        x = F.relu(self.conv1(concat))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten the tensor

        x = F.leaky_relu(self.fc1(x))
        
        value_estimate = self.fc_value(x).squeeze(1)

        return value_estimate
