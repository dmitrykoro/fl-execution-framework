"""
Dynamic CNN Architecture.

Automatically adapts to different dataset characteristics (image size, channels, number of classes).
Provides a universal CNN model for federated learning experiments across diverse datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCNN(nn.Module):
    """
    Dynamically sized CNN that adapts to dataset characteristics.

    This model automatically adjusts its architecture based on:
    - Number of output classes
    - Input image dimensions (height, width)
    - Number of input channels (grayscale vs RGB)

    Architecture:
        - Conv1: input_channels → 32 (5x5 kernel, padding=2)
        - MaxPool: 2x2
        - Conv2: 32 → 64 (5x5 kernel, padding=2)
        - MaxPool: 2x2
        - FC1: flattened → 512
        - Dropout: 0.5
        - FC2: 512 → num_classes

    Args:
        num_classes: Number of output classes for classification
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        input_height: Height of input images in pixels
        input_width: Width of input images in pixels
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        input_height: int = 28,
        input_width: int = 28,
    ):
        super(DynamicCNN, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=5, padding=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )

        # Calculate flattened size after convolutions and pooling
        # After conv1 (padding=2): H, W (same size)
        # After pool1: H/2, W/2
        # After conv2 (padding=2): H/2, W/2 (same size)
        # After pool2: H/4, W/4
        conv_output_height = input_height // 4
        conv_output_width = input_width // 4
        flattened_size = 64 * conv_output_height * conv_output_width

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (
            f"DynamicCNN("
            f"classes={self.num_classes}, "
            f"input_shape=({self.input_channels}, {self.input_height}, {self.input_width})"
            f")"
        )
