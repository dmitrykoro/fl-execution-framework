import torch
import torch.nn as nn
import torch.nn.functional as functional


class Cifar100Network(nn.Module):
    """CNN for CIFAR-100: 32x32 RGB input, 100 output classes.

    BatchNorm is intentionally omitted — the framework's set_parameters uses
    torch.Tensor(v) which mishandles 0-d arrays (num_batches_tracked) produced
    by BatchNorm state dicts.
    """

    def __init__(self) -> None:
        super(Cifar100Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # After 3x MaxPool(2): 32 -> 16 -> 8 -> 4 => 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 100)

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
