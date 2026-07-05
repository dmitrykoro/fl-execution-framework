import torch
import torch.nn as nn
import torch.nn.functional as functional

class MLPNetwork(nn.Module):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(16, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """Random weight initialization"""

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
