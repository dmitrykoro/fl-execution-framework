import torch
import torch.nn as nn
import torch.nn.functional as functional


class TissueMNISTNetwork(nn.Module):
    def __init__(self):
        super(TissueMNISTNetwork, self).__init__()
        # TissueMNIST: grayscale (1 channel), 8 output classes
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 8)  # 8 tissue classes

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x  # raw logits for CrossEntropyLoss

    def _initialize_weights(self):
        """Random weight initialization"""
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
