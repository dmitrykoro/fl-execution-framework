import torch
import torch.nn as nn
import torch.nn.functional as functional


class Cifar10Network(nn.Module):
    """CNN for CIFAR-10: 32x32 RGB input, 10 output classes.

    Same architecture as Cifar100Network with fc2 reduced to 10 outputs.
    BatchNorm omitted for compatibility with the framework's set_parameters.
    """

    def __init__(self) -> None:
        super(Cifar10Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 10)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)