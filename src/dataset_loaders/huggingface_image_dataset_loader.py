import logging
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class HuggingFaceImageDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace image datasets with transforms."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class HuggingFaceImageDatasetLoader:
    """
    Unified dataset loader for HuggingFace image classification datasets.

    Examples:
        MedMNIST BreastMNIST: hf_dataset_path="randall-lab/medmnist",
                             hf_dataset_name="breastmnist"
        CIFAR-10: hf_dataset_path="cifar10",
                 hf_dataset_name=None
    """

    def __init__(
        self,
        hf_dataset_path: str,
        hf_dataset_name: str = None,
        transformer: transforms = None,
        dataset_dir: str = None,  # Not used, kept for compatibility
        num_of_clients: int = 10,
        batch_size: int = 32,
        training_subset_fraction: float = 0.8,
        max_samples: int = None,  # Limit dataset size
    ):
        self.hf_dataset_path = hf_dataset_path
        self.hf_dataset_name = hf_dataset_name
        self.transformer = transformer
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction
        self.max_samples = max_samples

        if self.transformer is None:
            self.transformer = transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

    def _partition_iid(self, full_dataset):
        """Partition dataset uniformly across clients (IID distribution)."""
        client_size = len(full_dataset) // self.num_of_clients
        client_indices = []

        for client_id in range(self.num_of_clients):
            start_idx = client_id * client_size
            end_idx = (
                start_idx + client_size
                if client_id < self.num_of_clients - 1
                else len(full_dataset)
            )
            client_indices.append(list(range(start_idx, end_idx)))

        return client_indices

    def _partition_label_skew_dirichlet(self, full_dataset, alpha=0.5):
        """
        Partition dataset using Dirichlet distribution (Non-IID).
        Lower alpha = more heterogeneous (typical: 0.1-1.0).
        """
        labels = np.array(full_dataset["label"])
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(self.num_of_clients)]

        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            proportions = rng.dirichlet(np.repeat(alpha, self.num_of_clients))
            proportions = (proportions * len(class_indices)).astype(int)

            # Adjust to ensure all samples are assigned
            proportions[-1] = len(class_indices) - proportions[:-1].sum()

            rng.shuffle(class_indices)

            start_idx = 0
            for client_id, count in enumerate(proportions):
                end_idx = start_idx + count
                client_indices[client_id].extend(
                    class_indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx

        # Shuffle each client's indices to mix classes
        for indices in client_indices:
            rng.shuffle(indices)

        return client_indices

    def load_datasets(self):
        """Loads dataset from HuggingFace Hub and partitions into clients."""
        trainloaders = []
        valloaders = []

        if self.hf_dataset_name:
            dataset = load_dataset(
                self.hf_dataset_path, self.hf_dataset_name, trust_remote_code=True
            )
        else:
            dataset = load_dataset(self.hf_dataset_path, trust_remote_code=True)

        full_dataset = dataset["train"]

        # Limit dataset size for memory optimization
        if self.max_samples is not None and len(full_dataset) > self.max_samples:
            original_size = len(full_dataset)
            full_dataset = full_dataset.shuffle(seed=42)
            full_dataset = full_dataset.select(range(self.max_samples))
            logging.info(
                f"Dataset optimization: Limited from {original_size:,} to {self.max_samples:,} samples "
                f"({(self.max_samples / original_size) * 100:.1f}%) for faster processing"
            )

        # Use Non-IID for labeled datasets, IID for unlabeled
        if "label" in full_dataset.column_names:
            client_indices_list = self._partition_label_skew_dirichlet(
                full_dataset, alpha=0.5
            )
        else:
            # Only shuffle if not already shuffled above
            if self.max_samples is None or len(dataset["train"]) <= self.max_samples:
                full_dataset = full_dataset.shuffle(seed=42)
            client_indices_list = self._partition_iid(full_dataset)

        for client_id in range(self.num_of_clients):
            client_dataset = full_dataset.select(client_indices_list[client_id])

            split_dataset = client_dataset.train_test_split(
                test_size=(1 - self.training_subset_fraction), seed=42
            )

            train_dataset = HuggingFaceImageDataset(
                split_dataset["train"], transform=self.transformer
            )
            val_dataset = HuggingFaceImageDataset(
                split_dataset["test"], transform=self.transformer
            )

            trainloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Avoid CUDA fork issues
                pin_memory=torch.cuda.is_available(),  # Fast GPU transfer
            )
            valloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            trainloaders.append(trainloader)
            valloaders.append(valloader)

        return trainloaders, valloaders
