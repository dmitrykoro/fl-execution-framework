from typing import Optional

import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from datasets import load_dataset


class TextClassificationLoader:
    """
    Load and tokenize text classification datasets from HuggingFace Hub.

    Supports IID, Dirichlet, and Pathological partitioning strategies.
    Handles single text inputs and sentence pairs (e.g., for NLI tasks).
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_model: str,
        num_of_clients: int,
        batch_size: int,
        training_subset_fraction: float = 1.0,
        max_seq_length: int = 128,
        text_column: str = "text",
        text2_column: Optional[str] = None,
        label_column: str = "label",
        partitioning_strategy: str = "iid",
        partitioning_params: Optional[dict] = None,
    ) -> None:
        """
        Initialize TextClassificationLoader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "stanfordnlp/sst2")
            tokenizer_model: HuggingFace tokenizer model (e.g., "distilbert-base-uncased")
            num_of_clients: Number of federated learning clients
            batch_size: Batch size for DataLoaders
            training_subset_fraction: Fraction of training data to use (0.0-1.0)
            max_seq_length: Maximum sequence length for tokenization
            text_column: Name of the text column in the dataset
            text2_column: Name of the second text column (for sentence pairs, optional)
            label_column: Name of the label column in the dataset
            partitioning_strategy: "iid", "dirichlet", or "pathological"
            partitioning_params: Strategy-specific parameters (e.g., {"alpha": 0.5})
        """
        self.dataset_name = dataset_name
        self.tokenizer_model = tokenizer_model
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.text2_column = text2_column
        self.label_column = label_column
        self.partitioning_strategy = partitioning_strategy
        self.partitioning_params = partitioning_params or {}

    def load_datasets(self):
        """
        Load, tokenize, and partition dataset from HuggingFace Hub.

        Returns:
            tuple: (trainloaders, valloaders, num_labels)
                - trainloaders: List of PyTorch DataLoaders for training
                - valloaders: List of PyTorch DataLoaders for validation
                - num_labels: Number of classification labels in the dataset
        """
        # Load dataset from HuggingFace Hub
        dataset = load_dataset(self.dataset_name)

        # Get train and test splits
        train_dataset = dataset["train"]
        test_dataset = (
            dataset["validation"] if "validation" in dataset else dataset["test"]
        )

        # Detect number of labels from dataset features
        if hasattr(train_dataset.features[self.label_column], "num_classes"):
            num_labels = train_dataset.features[self.label_column].num_classes
        else:
            # Fallback: count unique labels
            num_labels = len(set(train_dataset[self.label_column]))

        # Apply training subset fraction if needed
        if self.training_subset_fraction < 1.0:
            subset_size = int(len(train_dataset) * self.training_subset_fraction)
            train_dataset = train_dataset.select(range(subset_size))

        # Initialize tokenizer and data collator
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Tokenize datasets
        train_dataset = self._tokenize_dataset(train_dataset, tokenizer)
        test_dataset = self._tokenize_dataset(test_dataset, tokenizer)

        # Partition training data across clients
        client_datasets = self._partition_dataset(train_dataset)

        # Create DataLoaders for each client
        trainloaders = []
        valloaders = []

        for client_dataset in client_datasets:
            trainloader = DataLoader(
                client_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )
            trainloaders.append(trainloader)

            # Use the same test dataset for all clients
            valloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
            valloaders.append(valloader)

        return trainloaders, valloaders, num_labels

    def _tokenize_dataset(self, dataset, tokenizer):
        """
        Tokenize text data and prepare for PyTorch.

        Args:
            dataset: HuggingFace Dataset object
            tokenizer: HuggingFace tokenizer

        Returns:
            Tokenized dataset with PyTorch format
        """

        def tokenize_function(examples):
            # Handle single text or sentence pairs
            if self.text2_column:
                return tokenizer(
                    examples[self.text_column],
                    examples[self.text2_column],
                    truncation=True,
                    max_length=self.max_seq_length,
                )
            else:
                return tokenizer(
                    examples[self.text_column],
                    truncation=True,
                    max_length=self.max_seq_length,
                )

        # Tokenize in batches
        dataset = dataset.map(tokenize_function, batched=True)

        # Rename label column to "labels" (expected by transformers)
        if self.label_column != "labels":
            dataset = dataset.rename_column(self.label_column, "labels")

        # Set format for PyTorch
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        return dataset

    def _partition_dataset(self, dataset):
        """
        Partition dataset across clients based on strategy.

        Args:
            dataset: Tokenized HuggingFace Dataset

        Returns:
            List of datasets, one per client
        """
        if self.partitioning_strategy == "iid":
            return self._partition_iid(dataset)
        elif self.partitioning_strategy == "dirichlet":
            return self._partition_dirichlet(dataset)
        elif self.partitioning_strategy == "pathological":
            return self._partition_pathological(dataset)
        else:
            raise ValueError(
                f"Unknown partitioning strategy: {self.partitioning_strategy}"
            )

    def _partition_iid(self, dataset):
        """
        Partition dataset evenly (IID) across clients.

        Args:
            dataset: HuggingFace Dataset

        Returns:
            List of datasets, one per client
        """
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)

        # Split evenly
        partition_size = len(dataset) // self.num_of_clients
        client_datasets = []

        for i in range(self.num_of_clients):
            start_idx = i * partition_size
            if i == self.num_of_clients - 1:
                # Last client gets remaining samples
                end_idx = len(dataset)
            else:
                end_idx = (i + 1) * partition_size

            client_datasets.append(dataset.select(range(start_idx, end_idx)))

        return client_datasets

    def _partition_dirichlet(self, dataset):
        """
        Partition dataset using Dirichlet distribution (non-IID).

        Creates label imbalance across clients using Dirichlet distribution.

        Args:
            dataset: HuggingFace Dataset

        Returns:
            List of datasets, one per client
        """
        alpha = self.partitioning_params.get("alpha", 0.5)

        # Get labels
        labels = np.array(dataset["labels"])
        num_classes = len(np.unique(labels))

        # Group indices by label
        label_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        # Initialize client indices
        client_indices = [[] for _ in range(self.num_of_clients)]

        # For each class, distribute samples to clients using Dirichlet
        for class_indices in label_indices:
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet([alpha] * self.num_of_clients)
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            client_splits = np.split(class_indices, proportions)

            for client_id, indices in enumerate(client_splits):
                client_indices[client_id].extend(indices.tolist())

        # Create datasets for each client
        client_datasets = []
        for indices in client_indices:
            if len(indices) > 0:
                client_datasets.append(dataset.select(indices))
            else:
                # Empty partition - create minimal dataset with one sample
                client_datasets.append(dataset.select([0]))

        return client_datasets

    def _partition_pathological(self, dataset):
        """
        Partition dataset pathologically (each client gets only K classes).

        Args:
            dataset: HuggingFace Dataset

        Returns:
            List of datasets, one per client
        """
        num_classes_per_client = self.partitioning_params.get(
            "num_classes_per_partition", 2
        )

        # Get labels
        labels = np.array(dataset["labels"])
        num_classes = len(np.unique(labels))

        # Group indices by label
        label_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        # Shuffle label indices
        for indices in label_indices:
            np.random.shuffle(indices)

        # Assign classes to clients in round-robin fashion
        client_datasets = []
        for client_id in range(self.num_of_clients):
            client_indices = []

            # Select K classes for this client
            for k in range(num_classes_per_client):
                class_id = (client_id * num_classes_per_client + k) % num_classes
                indices = label_indices[class_id]

                # Distribute samples from this class
                samples_per_client = len(indices) // (
                    self.num_of_clients // num_classes + 1
                )
                start_idx = (client_id // num_classes) * samples_per_client
                end_idx = start_idx + samples_per_client

                if start_idx < len(indices):
                    client_indices.extend(indices[start_idx:end_idx].tolist())

            if len(client_indices) > 0:
                client_datasets.append(dataset.select(client_indices))
            else:
                # Empty partition - create minimal dataset
                client_datasets.append(dataset.select([0]))

        return client_datasets
