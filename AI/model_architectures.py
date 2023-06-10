import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F


class EvaluationNetConvMaxPool(nn.Module):
    def __init__(self, input_dim=40, input_channels=1):
        super(EvaluationNetConvMaxPool, self).__init__()
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

class EvaluationNetConv(nn.Module):
    def __init__(self, input_dim=42, input_channels=1):
        #FIX PADDING TO BE IMPLICIT HERE
        super(EvaluationNetConv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=0)  # changed stride to 2
        output_dim1 = self.conv_output_size(input_dim, 5, 0, 2)  # after first conv layer
        output_dim2 = self.conv_output_size(output_dim1, 5, 0, 2)  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2 ** 2, 500)  # input features for fc1
        self.fc_value = nn.Linear(500, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        x = F.pad(x, pad=(2,2,2,2), mode='constant', value=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        value_estimate = self.fc_value(x).squeeze(1)
        return value_estimate


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, padding=2):
        super(AttentionModule, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2)

    def forward(self, x, player_positions):

        x = F.pad(x, pad=(2,2,2,2), mode='constant', value=1)
        out = F.relu(self.conv(x))

        mask = torch.zeros_like(out)

        batch_size, _, height, width = out.shape

        for b in range(batch_size):

            #Don't hardcode padding
            x_center, y_center = (player_positions[b] + 2).tolist()


            #Don't hardcode size
            x = torch.arange(0, 40 + 2, dtype=torch.float32)  # x dimension
            y = torch.arange(0, 40 + 2, dtype=torch.float32)  # y dimension

            x = x - x_center
            y = y - y_center

            xx, yy = torch.meshgrid(x, y)
            sigma = 10.0
            mask[b] = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)

        out = out * mask

        return out


class EvaluationAttentionConvNet(nn.Module):
    def __init__(self):
        super(EvaluationAttentionConvNet, self).__init__()

        self.conv1 = AttentionModule(1, 10)
        self.conv2 = AttentionModule(10, 20)

        self.fc1 = nn.Linear(20 * 40 * 40, 500)
        self.fc_value = nn.Linear(500, 1)

    def forward(self, model_input):
        #print("Model input inside model:", model_input)
        #print("len model input:", len(model_input))


        x, heads = model_input
        #print("shape of head:", heads.shape)
        #head_x, head_y = head

        #print("shape of x:", x.shape)

        # print("Forward called, X[0] shape", x[0].shape, " x[1].shape: ", x[1].shape)
        x = F.relu(self.conv1(x, heads))
        #print("shape after first conv:", x.shape)
        x = F.relu(self.conv2(x, heads))
        #print("shape after second conv:", x.shape)

        # print("size of x after both conv layers:", x.shape)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        #print("shape after flatten:", x.shape)
        # print("size of x after flatten:", x.shape)
        x = F.relu(self.fc1(x))
        #print("shape after first FC:", x.shape)

        x = F.relu(self.fc_value(x)).squeeze(1)

        return x
