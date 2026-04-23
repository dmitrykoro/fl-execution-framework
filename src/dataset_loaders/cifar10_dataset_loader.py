import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

_GLOBAL_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CIFAR10_CLASSES)}


class _GlobalMappingImageFolder(datasets.ImageFolder):
    """ImageFolder that uses the global CIFAR-10 class→index mapping.

    Clients may be missing class directories under non-IID partitioning;
    this ensures label indices are consistent across all clients.
    """

    def find_classes(self, directory: str):
        present = {d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))}
        classes = sorted(cls for cls in CIFAR10_CLASSES if cls in present)
        class_to_idx = {cls: _GLOBAL_CLASS_TO_IDX[cls] for cls in classes}
        return classes, class_to_idx


class Cifar10DatasetLoader:

    def __init__(
            self,
            transformer: transforms.Compose,
            dataset_dir: str,
            num_of_clients: int,
            batch_size: int,
            training_subset_fraction: float,
    ) -> None:
        self.transformer = transformer
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction

    def load_datasets(self):
        trainloaders, valloaders = [], []
        client_folders = [d for d in os.listdir(self.dataset_dir) if d.startswith("client_")]
        for client_folder in sorted(client_folders, key=lambda s: int(s.split("_")[1])):
            client_dataset = _GlobalMappingImageFolder(
                root=os.path.join(self.dataset_dir, client_folder),
                transform=self.transformer,
            )
            train_len = max(1, int(len(client_dataset) * self.training_subset_fraction))
            val_len = max(0, len(client_dataset) - train_len)
            train_ds, val_ds = random_split(
                client_dataset,
                [train_len, val_len],
                torch.Generator().manual_seed(42),
            )
            trainloaders.append(DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=0, pin_memory=torch.cuda.is_available(),
            ))
            valloaders.append(DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=torch.cuda.is_available(),
            ))
        return trainloaders, valloaders