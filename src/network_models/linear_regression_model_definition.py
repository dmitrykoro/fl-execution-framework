import torch
import torch.nn as nn
import torch.nn.functional as functional


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
       super(LinearRegressionModel, self).__init__()
       self.linear = nn.Linear(8, 1)

    def forward(self, x):
        return self.linear(x)