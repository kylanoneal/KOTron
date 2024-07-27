import torch
from torch import nn
import torch.nn.functional as F


class EvaluationNetConv3(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(EvaluationNetConv3, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1)  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 2)  # after first conv layer
        output_dim2 = self.conv_output_size(output_dim1, 3, 1, 2)  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2 ** 2, 250)  # input features for fc1
        self.fc_value = nn.Linear(250, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        padded_game_grid = F.pad(x[:, 0:1], pad=(self.PADDING, self.PADDING, self.PADDING, self.PADDING), mode='constant', value=1)
        padded_heads = F.pad(x[:, 1:], pad=(self.PADDING, self.PADDING, self.PADDING, self.PADDING), mode='constant', value=0)
        concat = torch.concat((padded_game_grid, padded_heads), axis=1)

        x = F.relu(self.conv1(concat))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten the tensor

        x = F.relu(self.fc1(x))
        value_estimate = self.fc_value(x).squeeze(1)
        return value_estimate


# class EvaluationNetConv3(nn.Module):
#     PADDING = 2
#
#     def __init__(self, grid_dim=40):
#         # FIX PADDING TO BE IMPLICIT HERE
#         super(EvaluationNetConv3, self).__init__()
#         self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=0)  # changed stride to 2
#         self.conv2 = nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1)  # changed stride to 2
#         output_dim1 = self.conv_output_size(grid_dim, 5, 2, 2)  # after first conv layer
#         output_dim2 = self.conv_output_size(output_dim1, 3, 1, 2)  # after second conv layer
#         self.fc1 = nn.Linear(10 * output_dim2 ** 2, 250)  # input features for fc1
#         self.fc_value = nn.Linear(250, 1)
#
#     @staticmethod
#     def conv_output_size(input_size, kernel_size, padding, stride):
#         return ((input_size - kernel_size + 2 * padding) // stride) + 1
#
#     def forward(self, x):
#         x = F.pad(x, pad=(self.PADDING, self.PADDING, self.PADDING, self.PADDING), mode='constant', value=1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = F.relu(self.fc1(x))
#         value_estimate = self.fc_value(x).squeeze(1)
#         return value_estimate


class EvaluationNetConv2(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(EvaluationNetConv2, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        self.conv2 = nn.Conv2d(15, 30, kernel_size=3, stride=2, padding=1)  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 2)  # after first conv layer
        output_dim2 = self.conv_output_size(output_dim1, 3, 1, 2)  # after second conv layer
        self.fc1 = nn.Linear(30 * output_dim2 ** 2, 500)  # input features for fc1
        self.fc_value = nn.Linear(500, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        x = F.pad(x, pad=(self.PADDING, self.PADDING, self.PADDING, self.PADDING), mode='constant', value=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        value_estimate = self.fc_value(x).squeeze(1)
        return value_estimate


class EvaluationNetConv1(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(EvaluationNetConv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 2)  # after first conv layer
        output_dim2 = self.conv_output_size(output_dim1, 5, 0, 2)  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2 ** 2, 500)  # input features for fc1
        self.fc_value = nn.Linear(500, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        x = F.pad(x, pad=(self.PADDING, self.PADDING, self.PADDING, self.PADDING), mode='constant', value=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        value_estimate = self.fc_value(x).squeeze(1)
        return value_estimate


def get_attn_mask(shape, heads, sigma=10.0):
    mask = torch.zeros(shape)

    batch_size, _, height, width = shape

    for b in range(batch_size):
        # print("player positions at b:", player_positions[b])
        # Don't hardcode padding

        x_center, y_center = ((heads[b] / (40 - 1) * (height - 1)).tolist())

        # Don't hardcode size
        x = torch.arange(0, height, dtype=torch.float32)  # x dimension
        y = torch.arange(0, width, dtype=torch.float32)  # y dimension

        x = x - x_center
        y = y - y_center

        xx, yy = torch.meshgrid(x, y, indexing='ij')

        mask[b] = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return mask


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, padding=2):
        super(AttentionModule, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2)

    def forward(self, x, player_positions):
        x = F.pad(x, pad=(2, 2, 2, 2), mode='constant', value=1)
        out = F.relu(self.conv(x))
        mask = get_attn_mask(out.shape, player_positions).to(device)
        out = out * mask

        return out


class EvaluationAttentionConvNet(nn.Module):
    def __init__(self):
        super(EvaluationAttentionConvNet, self).__init__()

        self.conv1_attn = AttentionModule(1, 10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=0)

        self.fc1 = nn.Linear(1280, 500)
        self.fc_value = nn.Linear(500, 1)

    def forward(self, model_input):
        # print("Model input inside model:", model_input)
        # print("len model input:", len(model_input))

        x, heads = model_input
        # print("shape of head:", heads.shape)
        # head_x, head_y = head

        # print("shape of x:", x.shape)

        # print("Forward called, X[0] shape", x[0].shape, " x[1].shape: ", x[1].shape)
        x = F.relu(self.conv1_attn(x, heads))
        # print("shape after first conv:", x.shape)
        x = F.relu(self.conv2(x))
        # print("shape after second conv:", x.shape)

        # print("size of x after both conv layers:", x.shape)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # print("size of x after flatten:", x.shape)
        x = F.relu(self.fc1(x))
        # print("shape after first FC:", x.shape)

        x = self.fc_value(x).squeeze(1)

        return x
