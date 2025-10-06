import os
import json
from typing import Dict, List

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class ImageDatasetLoader:
    def __init__(
        self,
        transformer: transforms,
        dataset_dir: str,
        num_of_clients: int,
        batch_size: int,
        training_subset_fraction: float,
        seed: int = 1337,
        splits_filename: str | None = None,
    ) -> None:
        """
        Reproducible CPU dataloader:
          - Deterministic per-client train/val splits cached to disk (per seed)
          - Deterministic DataLoader order via seeded torch.Generator and num_workers=0
        """
        self.transformer = transformer
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = float(training_subset_fraction)
        self.seed = int(seed)
        self._splits_filename = (
            splits_filename or f"_splits_seed_{self.seed}.json"
        )

    # --------- helpers for reproducibility ---------

    def _per_client_seed(self, client_id: int) -> int:
        # stable, bounded 32-bit int per client
        return (self.seed + 1000 * int(client_id)) & 0xFFFF_FFFF

    def _split_cache_path(self) -> str:
        # store alongside the dataset directory
        return os.path.join(self.dataset_dir, self._splits_filename)

    def _load_cached_splits(self) -> Dict[str, Dict[str, List[int]]]:
        p = self._split_cache_path()
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
        return {}

    def _save_cached_splits(self, cache: Dict[str, Dict[str, List[int]]]) -> None:
        p = self._split_cache_path()
        with open(p, "w") as f:
            json.dump(cache, f)

    def _build_loader(self, subset: Subset, shuffle: bool, gen_seed: int) -> DataLoader:
        # Deterministic loader: seeded generator + single-thread workers
        gen = torch.Generator().manual_seed(gen_seed)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=gen if shuffle else None,  # generator only matters when shuffling
            num_workers=0,        # no forking => deterministic on CPU
            pin_memory=False,
            drop_last=False,
        )

    # --------- public API ---------

    def load_datasets(self):
        """Partition and load datasets for each client (deterministic)."""

        trainloaders: List[DataLoader] = []
        valloaders: List[DataLoader] = []

        # Discover client folders and sort numerically by id
        client_folders = [d for d in os.listdir(self.dataset_dir) if d.startswith("client_")]
        client_folders = sorted(client_folders, key=lambda s: int(s.split("_")[1]))

        # (Re)use cached splits if available for this seed
        split_cache: Dict[str, Dict[str, List[int]]] = self._load_cached_splits()

        for client_folder in client_folders:
            client_id = int(client_folder.split("_")[1])

            client_root = os.path.join(self.dataset_dir, client_folder)
            client_dataset = datasets.ImageFolder(root=client_root, transform=self.transformer)

            n_total = len(client_dataset)
            n_train = int(round(n_total * self.training_subset_fraction))
            n_train = max(0, min(n_total, n_train))
            n_val = n_total - n_train

            # Guard against empty datasets gracefully
            if n_total == 0:
                train_subset = Subset(client_dataset, [])
                val_subset = Subset(client_dataset, [])
                trainloaders.append(self._build_loader(train_subset, shuffle=False, gen_seed=self._per_client_seed(client_id)))
                valloaders.append(self._build_loader(val_subset, shuffle=False, gen_seed=self._per_client_seed(client_id)))
                continue

            # Build or load deterministic split indices
            if client_folder in split_cache:
                train_idx = split_cache[client_folder]["train"]
                val_idx = split_cache[client_folder]["val"]
            else:
                # Deterministic permutation per client
                gen_split = torch.Generator().manual_seed(self._per_client_seed(client_id))
                perm = torch.randperm(n_total, generator=gen_split).tolist()
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                split_cache[client_folder] = {"train": train_idx, "val": val_idx}

            train_subset = Subset(client_dataset, train_idx)
            val_subset = Subset(client_dataset, val_idx)

            # Deterministic DataLoaders with NO generator (so Ray can pickle them)
            train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=False,      # <- keep deterministic order from train_idx
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            )
            val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,      # val is never shuffled
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            )

            trainloaders.append(train_loader)
            valloaders.append(val_loader)

        # Persist splits so future runs reuse the exact same indices
        self._save_cached_splits(split_cache)

        return trainloaders, valloaders
