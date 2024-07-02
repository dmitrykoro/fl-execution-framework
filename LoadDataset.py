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
        Function to partition and load the dataset for each clients.
        """
        # Change the transform function to add noise to the data
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
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
            len_val = len(client_dataset) // 10  # 10% validation set
            len_train = len(client_dataset) - len_val
            ds_train, ds_val = random_split(client_dataset, [len_train, len_val], torch.Generator().manual_seed(42))

            # Create DataLoaders
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCH_SIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCH_SIZE))
        return trainloaders, valloaders, testloaders

    def display_samples_from_loader(self, loader, title, num_images=5):
        dataiter = iter(loader)
        images, labels = next(dataiter)
        self.show_images(images, labels, title=title)

    def show_images(self, images, labels, title=""):
        num_images = len(images)  # Use the actual number of images in the batch
        plt.figure(figsize=(10, 2))
        plt.suptitle(title)
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(torchvision.transforms.ToPILImage()(images[i]))
            plt.xlabel(f"Label: {labels[i]}")
        plt.show()

# '''
# Uncomment the below code and run the file to test LoadDataset functionality
# '''
# NUM_CLIENTS = 12
# BATCH_SIZE = 4
# ROOT_DIR = './CLIENTS DATA'

# # Loading dataset based on number of clients and batch size
# data_loader = LoadDataset(ROOT_DIR, NUM_CLIENTS, BATCH_SIZE)
# trainloaders, valloaders, testloader = data_loader.load_datasets()

# data_loader.display_samples_from_loader(trainloaders[11], "Sample Images")
