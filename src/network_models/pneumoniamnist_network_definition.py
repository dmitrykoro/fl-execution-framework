import torch
import torch.nn as nn
import torch.nn.functional as functional


class PneumoniamnistNetwork(nn.Module):
    def __init__(self) -> None:
        super(PneumoniamnistNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
