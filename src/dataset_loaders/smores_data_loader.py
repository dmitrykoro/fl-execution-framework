import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import glob
import pandas as pd
import copy
import numpy as np




class SmoresDatasetLoader:

    def __init__(
            self,
            dataset_dir: str,
            num_of_clients: int,
            batch_size: int,
            training_subset_fraction: float,
            sequence_len: int = 12
        ):
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction

        self.sequence_len = sequence_len

    def _create_sequences_single_step(self, data, seq_length: int):
        """
        Creates sequences for single step prediction

        Args:
            data (array-like): The input data.
            seq_length (int): The length of each training sequence.

        Returns:
            np.ndarray: Array of input sequences.
            np.ndarray: Array of target sequences (half the length of input sequences).
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])                        # Input sequence
            targets.append(data[i+seq_length]) # Target is a single step

        return np.array(sequences), np.array(targets)

    def _create_sequences_mutlistep(self, data, seq_length: int):
        """
        Creates sequences and their corresponding target sequences from the input data.
        The target sequence is 1/3 the size of the training sequence.

        Args:
            data (array-like): The input data.
            seq_length (int): The length of each training sequence.

        Returns:
            np.ndarray: Array of input sequences.
            np.ndarray: Array of target sequences (half the length of input sequences).
        """
        target_length = seq_length // 3  # Target is 1/3 the size of the training sequence

        if len(data) < seq_length + target_length:
            raise ValueError("Data length must be at least seq_length + target_length.")

        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length - target_length + 1):
            sequences.append(data[i:i+seq_length])                        # Input sequence
            targets.append(data[i+seq_length:i+seq_length+target_length]) # Target sequence

        return np.array(sequences), np.array(targets)

    def _make_sequence_tensor(self, sequence):
        """
        Make a pytorch Tensor from a numpy sequence

        Args:
            sequence (np.ndarray): Input sequence
            device : pytorch device
        
        Returns:
            torch.FloatTensor with dim (batch, seq_length, features)
        """
        return torch.tensor(sequence, dtype=torch.float32)


    def load_datasets(self):
        """Function to partition and load the dataset for each client."""

        trainloaders = []
        valloaders = []

        client_folders = [d for d in os.listdir(self.dataset_dir) if d.startswith("client_")]

        for client_folder in sorted(client_folders, key=lambda string: int(string.split("_")[1])):

            csv_files = glob.glob(os.path.join(self.dataset_dir, client_folder, "*.csv"))

            # Load data for each client
            client_dataset = pd.read_csv(csv_files[0], index_col=0)

            # Create sequences and labels
            seq, labels = self._create_sequences_single_step(client_dataset.values, self.sequence_len)
            seq_tensor = self._make_sequence_tensor(seq)
            labels_tensor = self._make_sequence_tensor(labels)

            # Creating dataset
            dataset = TensorDataset(seq_tensor, labels_tensor)
            
            train_len = int(len(dataset) * self.training_subset_fraction)
            val_len = len(dataset) - train_len
            train_dataset, val_dataset = random_split(
                dataset,
                [train_len, val_len],
                torch.Generator().manual_seed(42)
            )

            # compute normalization statistics from training set only
            train_features = train_dataset[:][0]
            x_mean = train_features.mean(dim=(0,1))
            x_std = train_features.std(dim=(0,1))
            x_std[x_std == 0] = 1 # avoid divide by zero
            train_targets = train_dataset[:][1]
            y_mean = train_targets.mean(dim=(0,1))
            y_std = train_targets.std(dim=(0,1))
            y_std[y_std == 0] = 1

            # Normalize both features and targets
            def normalize(dataset, x_mean, x_std, y_mean, y_std):
                X, y = dataset[:]
                X = (X - x_mean) / x_std
                y = (y - y_mean) / y_std
                return X, y
            
            # Normalize both train and test sets
            X_train, y_train = normalize(train_dataset, x_mean, x_std, y_mean, y_std)
            X_val, y_val = normalize(val_dataset, x_mean, x_std, y_mean, y_std)
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            # store feature and target mean and std for client use in denormalization for metrics
            train_dataset.x_mean = x_mean
            train_dataset.x_std = x_std
            val_dataset.x_mean = x_mean
            val_dataset.x_std = x_std
            train_dataset.y_mean = y_mean
            train_dataset.y_std = y_std
            val_dataset.y_mean = y_mean
            val_dataset.y_std = y_std

            # store globalized min and max feature
            train_dataset.y_min = y_train.min().item()
            train_dataset.y_max = y_train.max().item()
            val_dataset.y_min = y_train.min().item()
            val_dataset.y_max = y_train.max().item()

            # store feature names for use in snapshot logging
            train_dataset.feature_names = ['DO']
            val_dataset.feature_names = ['DO']

            trainloaders.append(DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Avoid CUDA fork issues
                pin_memory=torch.cuda.is_available()  # Fast GPU transfer
            ))
            valloaders.append(DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            ))
        
        return trainloaders, valloaders

