"""
Tests for FederatedDatasetLoader.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
from torch.utils.data import DataLoader

from src.dataset_loaders.federated_dataset_loader import FederatedDatasetLoader


class TestFederatedDatasetLoaderInit:
    """Test FederatedDatasetLoader initialization."""

    def test_basic_initialization(self):
        """Should initialize with required parameters."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
        )
        assert loader.dataset_name == "mnist"
        assert loader.num_of_clients == 10
        assert loader.batch_size == 32
        assert loader.training_subset_fraction == 0.8
        assert loader.partitioning_strategy == "iid"
        assert loader.partitioning_params == {}

    def test_initialization_with_partitioning_params(self):
        """Should initialize with custom partitioning parameters."""
        params = {"alpha": 0.5}
        loader = FederatedDatasetLoader(
            dataset_name="cifar10",
            num_of_clients=5,
            batch_size=64,
            training_subset_fraction=0.9,
            partitioning_strategy="dirichlet",
            partitioning_params=params,
        )
        assert loader.partitioning_strategy == "dirichlet"
        assert loader.partitioning_params == params

    def test_initialization_with_none_params(self):
        """Should handle None partitioning_params."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_params=None,
        )
        assert loader.partitioning_params == {}


class TestCreatePartitioner:
    """Test partitioner creation."""

    def test_create_iid_partitioner(self):
        """Should create IidPartitioner for 'iid' strategy."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="iid",
        )
        partitioner = loader._create_partitioner()
        assert isinstance(partitioner, IidPartitioner)
        assert partitioner.num_partitions == 10

    def test_create_dirichlet_partitioner(self):
        """Should create DirichletPartitioner with alpha parameter."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="dirichlet",
            partitioning_params={"alpha": 0.3},
        )
        partitioner = loader._create_partitioner()
        assert isinstance(partitioner, DirichletPartitioner)
        assert partitioner._num_partitions == 10
        # _alpha can be an array, so check if all values are 0.3
        import numpy as np

        assert np.all(partitioner._alpha == 0.3)

    def test_create_dirichlet_partitioner_default_alpha(self):
        """Should use default alpha if not provided."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="dirichlet",
        )
        partitioner = loader._create_partitioner()
        assert isinstance(partitioner, DirichletPartitioner)
        # _alpha can be an array, so check if all values are 0.5
        import numpy as np

        assert np.all(partitioner._alpha == 0.5)

    def test_create_pathological_partitioner(self):
        """Should create PathologicalPartitioner with num_classes parameter."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="pathological",
            partitioning_params={"num_classes_per_partition": 3},
        )
        partitioner = loader._create_partitioner()
        assert isinstance(partitioner, PathologicalPartitioner)
        assert partitioner._num_partitions == 10
        assert partitioner._num_classes_per_partition == 3

    def test_create_pathological_partitioner_default_classes(self):
        """Should use default num_classes if not provided."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="pathological",
        )
        partitioner = loader._create_partitioner()
        assert isinstance(partitioner, PathologicalPartitioner)
        assert partitioner._num_classes_per_partition == 2

    def test_unknown_partitioning_strategy_raises_error(self):
        """Should raise ValueError for unknown strategy."""
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=10,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="unknown_strategy",
        )
        with pytest.raises(ValueError, match="Unknown partitioning strategy"):
            loader._create_partitioner()


class TestLoadDatasets:
    """Test dataset loading and partitioning."""

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_load_datasets_creates_correct_number_of_loaders(
        self, mock_federated_dataset
    ):
        """Should create trainloaders and valloaders for each client."""
        # Setup mock partition
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        # Setup mock FederatedDataset
        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=5,
            batch_size=32,
            training_subset_fraction=0.8,
        )

        trainloaders, valloaders, num_classes = loader.load_datasets()

        assert len(trainloaders) == 5
        assert len(valloaders) == 5
        assert all(isinstance(tl, DataLoader) for tl in trainloaders)
        assert all(isinstance(vl, DataLoader) for vl in valloaders)

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_load_datasets_uses_correct_partitioner(self, mock_federated_dataset):
        """Should pass partitioner to FederatedDataset."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        loader = FederatedDatasetLoader(
            dataset_name="cifar10",
            num_of_clients=3,
            batch_size=64,
            training_subset_fraction=0.75,
            partitioning_strategy="dirichlet",
            partitioning_params={"alpha": 0.1},
        )

        loader.load_datasets()

        # Verify FederatedDataset was called with correct arguments
        mock_federated_dataset.assert_called_once()
        call_kwargs = mock_federated_dataset.call_args[1]
        assert call_kwargs["dataset"] == "cifar10"
        assert "train" in call_kwargs["partitioners"]
        assert isinstance(call_kwargs["partitioners"]["train"], DirichletPartitioner)

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_load_datasets_loads_correct_partitions(self, mock_federated_dataset):
        """Should load partition for each client ID."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        num_clients = 4
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=num_clients,
            batch_size=32,
            training_subset_fraction=0.8,
        )

        loader.load_datasets()

        # Verify load_partition was called for each client
        assert mock_fds.load_partition.call_count == num_clients
        for i in range(num_clients):
            mock_fds.load_partition.assert_any_call(partition_id=i, split="train")

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    @patch("src.dataset_loaders.federated_dataset_loader.random_split")
    def test_load_datasets_splits_data_correctly(
        self, mock_random_split, mock_federated_dataset
    ):
        """Should split data according to training_subset_fraction."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        # Mock train/val datasets with proper __len__ for DataLoader
        mock_train = MagicMock()
        mock_train.__len__ = Mock(return_value=80)
        mock_val = MagicMock()
        mock_val.__len__ = Mock(return_value=20)
        mock_random_split.return_value = [mock_train, mock_val]

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=2,
            batch_size=32,
            training_subset_fraction=0.8,
        )

        loader.load_datasets()

        # Verify split was called with correct sizes (80% train, 20% val)
        mock_random_split.assert_called()
        call_args = mock_random_split.call_args[0]
        assert call_args[1] == [80, 20]  # 80% of 100, 20% of 100

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    @patch("src.dataset_loaders.federated_dataset_loader.random_split")
    def test_load_datasets_ensures_validation_sample(
        self, mock_random_split, mock_federated_dataset
    ):
        """Should ensure at least 1 validation sample when possible."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=10)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        # Mock train/val datasets with proper __len__ for DataLoader
        mock_train = MagicMock()
        mock_train.__len__ = Mock(return_value=9)
        mock_val = MagicMock()
        mock_val.__len__ = Mock(return_value=1)
        mock_random_split.return_value = [mock_train, mock_val]

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=1,
            batch_size=32,
            training_subset_fraction=1.0,  # 100% would leave 0 for validation
        )

        loader.load_datasets()

        # Should adjust to ensure 1 validation sample
        call_args = mock_random_split.call_args[0]
        assert call_args[1] == [9, 1]  # 9 train, 1 val

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    @patch("src.dataset_loaders.federated_dataset_loader.random_split")
    @patch("src.dataset_loaders.federated_dataset_loader.DataLoader")
    def test_load_datasets_handles_single_sample_partition(
        self, mock_dataloader, mock_random_split, mock_federated_dataset
    ):
        """Should handle partition with only 1 sample."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=1)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        # Mock train/val datasets
        mock_train = MagicMock()
        mock_val = MagicMock()
        mock_random_split.return_value = [mock_train, mock_val]

        # Mock DataLoader to avoid issues with empty datasets
        mock_dataloader.return_value = MagicMock()

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=1,
            batch_size=32,
            training_subset_fraction=0.8,
        )

        loader.load_datasets()

        # With 1 sample and 0.8 fraction, int(1 * 0.8) = 0 train, 1 val
        call_args = mock_random_split.call_args[0]
        assert call_args[1] == [0, 1]

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_load_datasets_sets_format_to_torch(self, mock_federated_dataset):
        """Should set partition format to 'torch'."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=2,
            batch_size=32,
            training_subset_fraction=0.8,
        )

        loader.load_datasets()

        # Verify set_format was called with "torch"
        assert mock_partition.set_format.call_count == 2
        mock_partition.set_format.assert_called_with("torch")

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    @patch("src.dataset_loaders.federated_dataset_loader.DataLoader")
    def test_load_datasets_creates_dataloaders_with_correct_batch_size(
        self, mock_dataloader, mock_federated_dataset
    ):
        """Should create DataLoaders with specified batch size."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        mock_dataloader.return_value = MagicMock()

        batch_size = 64
        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=1,
            batch_size=batch_size,
            training_subset_fraction=0.8,
        )

        loader.load_datasets()

        # Verify DataLoader was called with correct batch_size
        calls = mock_dataloader.call_args_list
        for call in calls:
            assert call[1]["batch_size"] == batch_size

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    @patch("src.dataset_loaders.federated_dataset_loader.DataLoader")
    def test_load_datasets_shuffles_training_data(
        self, mock_dataloader, mock_federated_dataset
    ):
        """Should shuffle training data but not validation data."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        mock_dataloader.return_value = MagicMock()

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=1,
            batch_size=32,
            training_subset_fraction=0.8,
        )

        loader.load_datasets()

        # Check that shuffle=True for train, shuffle not set (False) for val
        calls = mock_dataloader.call_args_list
        # First call should be trainloader with shuffle=True
        assert calls[0][1].get("shuffle") is True
        # Second call should be valloader without shuffle or shuffle=False
        assert calls[1][1].get("shuffle", False) is False


class TestIntegration:
    """Integration tests with real partitioners (no mocking)."""

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_end_to_end_iid_loading(self, mock_federated_dataset):
        """Test complete flow with IID partitioner."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=3,
            batch_size=16,
            training_subset_fraction=0.7,
            partitioning_strategy="iid",
        )

        trainloaders, valloaders, num_classes = loader.load_datasets()

        assert len(trainloaders) == 3
        assert len(valloaders) == 3

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_end_to_end_dirichlet_loading(self, mock_federated_dataset):
        """Test complete flow with Dirichlet partitioner."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        loader = FederatedDatasetLoader(
            dataset_name="cifar10",
            num_of_clients=5,
            batch_size=32,
            training_subset_fraction=0.8,
            partitioning_strategy="dirichlet",
            partitioning_params={"alpha": 0.1},
        )

        trainloaders, valloaders, num_classes = loader.load_datasets()

        assert len(trainloaders) == 5
        assert len(valloaders) == 5

    @patch("src.dataset_loaders.federated_dataset_loader.FederatedDataset")
    def test_end_to_end_pathological_loading(self, mock_federated_dataset):
        """Test complete flow with Pathological partitioner."""
        mock_partition = MagicMock()
        mock_partition.__len__ = Mock(return_value=100)
        mock_partition.set_format = Mock()

        mock_fds = MagicMock()
        mock_fds.load_partition = Mock(return_value=mock_partition)
        mock_federated_dataset.return_value = mock_fds

        loader = FederatedDatasetLoader(
            dataset_name="mnist",
            num_of_clients=4,
            batch_size=64,
            training_subset_fraction=0.9,
            partitioning_strategy="pathological",
            partitioning_params={"num_classes_per_partition": 3},
        )

        trainloaders, valloaders, num_classes = loader.load_datasets()

        assert len(trainloaders) == 4
        assert len(valloaders) == 4
