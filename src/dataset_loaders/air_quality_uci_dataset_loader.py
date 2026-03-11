import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import glob
import pandas as pd
import copy

class AirQualityUCIDatasetLoader:
    _always_used_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

    def __init__(
            self,
            dataset_dir: str,
            num_of_clients: int,
            batch_size: int,
            training_subset_fraction: float,
            target_gas: str = "CO"
    ) -> None:
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction

        self.target_gas = target_gas + "(GT)"
        self.used_columns = copy.deepcopy(AirQualityUCIDatasetLoader._always_used_columns)
        self.used_columns.append(self.target_gas)


    def load_datasets(self):
        """Function to partition and load the dataset for each client."""

        trainloaders = []
        valloaders = []

        client_folders = [d for d in os.listdir(self.dataset_dir) if d.startswith("client_")]

        for client_folder in sorted(client_folders, key=lambda string: int(string.split("_")[1])):

            csv_files = glob.glob(os.path.join(self.dataset_dir, client_folder, "*.csv"))
            
            client_dataset = pd.read_csv(csv_files[0], usecols=self.used_columns)
            X = client_dataset.drop(columns=[self.target_gas]).values
            y = client_dataset[self.target_gas].values
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
            
            train_len = int(len(dataset) * self.training_subset_fraction)
            val_len = len(dataset) - train_len
            train_dataset, val_dataset = random_split(
                dataset,
                [train_len, val_len],
                torch.Generator().manual_seed(42)
            )

            # compute normalization statistics from training set only
            train_features = train_dataset[:][0]
            x_mean = train_features.mean(dim=0)
            x_std = train_features.std(dim=0)
            x_std[x_std == 0] = 1 # avoid divide by zero
            train_targets = train_dataset[:][1]
            y_mean = train_targets.mean(dim=0)
            y_std = train_targets.std(dim=0)
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

            # store feature names for use in snapshot logging
            feature_names = client_dataset.drop(columns=[self.target_gas]).columns.tolist()
            train_dataset.feature_names = feature_names
            val_dataset.feature_names = feature_names

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