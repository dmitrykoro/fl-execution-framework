import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Canonical CIFAR-100 fine-label ordering (torchvision CIFAR100.classes)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
]

_GLOBAL_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CIFAR100_CLASSES)}


class _GlobalMappingImageFolder(datasets.ImageFolder):
    """ImageFolder variant that uses the global CIFAR-100 class→index mapping.

    Clients may be missing entire class directories (non-IID partitioning).
    This subclass skips missing/empty directories without remapping labels,
    so gradient updates from all clients refer to the same 100-class label space.
    """

    def find_classes(self, directory: str):
        # PyTorch's supported image extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        present = set()

        # Only add the directory if it is a folder AND contains at least one valid image
        for d in os.listdir(directory):
            d_path = os.path.join(directory, d)
            if os.path.isdir(d_path):
                has_images = any(f.lower().endswith(valid_extensions) for f in os.listdir(d_path))
                if has_images:
                    present.add(d)

        classes = sorted(cls for cls in CIFAR100_CLASSES if cls in present)
        class_to_idx = {cls: _GLOBAL_CLASS_TO_IDX[cls] for cls in classes}
        
        return classes, class_to_idx


class Cifar100DatasetLoader:
    """Drop-in replacement for ImageDatasetLoader for CIFAR-100 Dirichlet partitions."""

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
        trainloaders = []
        valloaders = []

        client_folders = [d for d in os.listdir(self.dataset_dir) if d.startswith("client_")]
        for client_folder in sorted(client_folders, key=lambda s: int(s.split("_")[1])):
            client_dataset = _GlobalMappingImageFolder(
                root=os.path.join(self.dataset_dir, client_folder),
                transform=self.transformer,
            )

            train_len = int(len(client_dataset) * self.training_subset_fraction)
            val_len = len(client_dataset) - train_len

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
