import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F


class DirectionValueNetConvMaxPool(nn.Module):
    def __init__(self, input_dim=40, input_channels=1):
        super(DirectionValueNetConvMaxPool, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 30, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=5, stride=1, padding=0)
        #self.fc1 = nn.Linear(30 * (input_dim // 4) ** 2, 1000)
        self.fc1 = nn.Linear(2940, 1000)
        # self.fc2 = nn.Linear(1000, 4)
        self.fc_value = nn.Linear(1000, 1)  # new linear layer to output the value estimate

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        # action_probs = F.softmax(self.fc2(x), dim=1)  # apply softmax to output action probabilities
        value_estimate = self.fc_value(x)
        return value_estimate

class DirectionValueNetConv(nn.Module):
    def __init__(self, input_dim=42, input_channels=1):
        #FIX PADDING TO BE IMPLICIT HERE
        super(DirectionValueNetConv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 30, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        self.conv2 = nn.Conv2d(30, 60, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        output_dim1 = self.conv_output_size(input_dim, 5, 0, 2)  # after first conv layer
        output_dim2 = self.conv_output_size(output_dim1, 5, 0, 2)  # after second conv layer
        self.fc1 = nn.Linear(60 * output_dim2 ** 2, 1000)  # input features for fc1
        self.fc_value = nn.Linear(1000, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        value_estimate = self.fc_value(x)
        return value_estimate

# class DirectionNetConv(nn.Module):
#     def __init__(self):
#         super(DirectionNetConv, self).__init__()
#         self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(15, 30, kernel_size=5, stride=1, padding=2)
#         self.fc1 = nn.Linear(30*10*10, 1000)
#         self.fc2 = nn.Linear(1000, 4)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
