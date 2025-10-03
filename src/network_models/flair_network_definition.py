import torch
import torch.nn as nn
import torch.nn.functional as F


class FlairNetwork(nn.Module):
    def __init__(self) -> None:
        super(FlairNetwork, self).__init__()
        # Increase the number of filters and convolutional layers
        self.conv1 = nn.Conv2d(
            3, 32, 3, padding=1
        )  # Increased number of filters for more feature extraction
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(
            64, 128, 3, padding=1
        )  # Add an extra convolutional layer

        self.pool = nn.MaxPool2d(2, 2)

        # Placeholder for FC layers - will be initialized on first forward pass
        self.fc1 = None
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)  # Change output to 2 classes

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Extra convolutional layer

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Initialize fc1 on first forward pass
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512)
            if x.is_cuda:
                self.fc1 = self.fc1.cuda()
            # Initialize weights for the new layer
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        """Random weight initialization"""
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")

        # fc1 is initialized dynamically in forward pass
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
