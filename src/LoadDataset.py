from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import torch
import matplotlib.pyplot as plt
import torchvision


class LoadDataset:
    """
    Class to load local dataset into respective loader arrays
    """

    def __init__(self, root_dir, num_clients, batch_size) -> None:
        """
        init function for LoadDataset class
        """
        self.ROOT_DIR = root_dir
        self.NUM_CLIENTS = num_clients
        self.BATCH_SIZE = batch_size

    def load_datasets(self):
        """
        Function to partition and load the dataset for each client.
        """

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale images
            transforms.Resize((28, 28)),                  # Ensure the images are 28x28
            transforms.ToTensor(),                        # Convert images to tensors
            transforms.Normalize((0.5,), (0.5,))          # Normalize the images
        ])

        # Create train/val for each partition and wrap it into DataLoader
        trainloaders = []
        valloaders = []
        # currently we are not using testloaders as of now.
        testloaders = []

        for i in range(self.NUM_CLIENTS):
            client_folder = os.path.join(self.ROOT_DIR, f'client_{i}')

            # assign different transform to different clients based on indexes
            client_dataset = datasets.ImageFolder(root=client_folder, transform=transform)

            # Split into training and validation sets
            val_set_len = len(client_dataset) // 10  # 10% validation set
            train_set_len = len(client_dataset) - val_set_len

            train_dataset, validation_dataset = random_split(client_dataset, [train_set_len, val_set_len], torch.Generator().manual_seed(42))

            # Create DataLoaders
            trainloaders.append(DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(validation_dataset, batch_size=self.BATCH_SIZE))

        return trainloaders, valloaders, testloaders
