# experiments/common/model.py

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 style model for 1x32x32 input.

    Input:
        [N, 1, 32, 32]

    Shape:
        32x32
        -> conv1 5x5 valid: 6x28x28
        -> pool: 6x14x14
        -> conv2 5x5 valid: 16x10x10
        -> pool: 16x5x5
        -> flatten: 400
        -> fc1: 120
        -> fc2: 84
        -> fc3: 10
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), 400)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x