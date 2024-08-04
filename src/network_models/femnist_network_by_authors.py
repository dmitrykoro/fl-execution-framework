import torch
import torch.nn as nn
import torch.nn.functional as F


class FemnistNetworkByAuthors(nn.Module):
    def __init__(self):
        super(FemnistNetworkByAuthors, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits
