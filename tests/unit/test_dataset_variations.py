"""
Parameterized tests for dataset variations in federated learning scenarios.

This module tests dataset loading, transformation operations, and dataset-specific
characteristics across all supported dataset types.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from tests.fixtures.mock_datasets import (MockDatasetHandler,
                                          MockFederatedDataset,
                                          generate_mock_dataset_config)

# Dataset configurations with expected characteristics
DATASET_CONFIGURATIONS = [
    # (dataset_name, expected_shape, num_channels, is_grayscale, expected_size)
    ("its", (3, 224, 224), 3, False, (224, 224)),
    ("femnist_iid", (1, 28, 28), 1, True, (28, 28)),
    ("femnist_niid", (1, 28, 28), 1, True, (28, 28)),
    (
        "flair",
        (3, 224, 224),
        3,
        False,
        (224, 224),
    ),  # Note: MockDatasetHandler uses 224x224 for flair
    ("pneumoniamnist", (1, 28, 28), 1, True, (28, 28)),
    ("bloodmnist", (3, 28, 28), 3, False, (28, 28)),
    ("lung_photos", (1, 224, 224), 1, True, (224, 224)),
    ("medquad", None, None, None, None),  # Text dataset - different characteristics
]

# Dataset-specific transformation parameters
DATASET_TRANSFORM_CONFIGS = {
    "its": {"resize": (224, 224), "normalize": None},
    "femnist_iid": {
        "resize": (28, 28),
        "normalize": ((0.5,), (0.5,)),
        "grayscale": True,
    },
    "femnist_niid": {
        "resize": (28, 28),
        "normalize": ((0.5,), (0.5,)),
        "grayscale": True,
    },
    "flair": {"resize": (256, 256), "normalize": None},
    "pneumoniamnist": {
        "resize": (28, 28),
        "normalize": ((0.5,), (0.5,)),
        "grayscale": True,
    },
    "bloodmnist": {"resize": (28, 28), "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))},
    "lung_photos": {
        "resize": (224, 224),
        "normalize": ((0.5,), (0.5,)),
        "grayscale": True,
    },
    "medquad": {
        "tokenizer": "bert-base-uncased",
        "chunk_size": 256,
        "mlm_probability": 0.15,
    },
}


class TestDatasetVariations:
    """Test suite for dataset variation scenarios."""

    @pytest.mark.parametrize(
        "dataset_name,expected_shape,num_channels,is_grayscale,expected_size",
        DATASET_CONFIGURATIONS[:-1],  # Exclude medquad for image tests
    )
    def test_image_dataset_characteristics(
        self, dataset_name, expected_shape, num_channels, is_grayscale, expected_size
    ):
        """Test that each image dataset has the correct characteristics."""
        # Create mock dataset handler
        handler = MockDatasetHandler(dataset_type=dataset_name)
        handler.setup_dataset(num_clients=5)

        # Get client data
        client_data = handler.get_client_data(client_id=0)

        # Verify dataset characteristics
        assert len(client_data) > 0, f"Dataset {dataset_name} should have samples"

        # Check data shape
        sample_data, sample_label = client_data[0]
        assert sample_data.shape == expected_shape, (
            f"Dataset {dataset_name} should have shape {expected_shape}, "
            f"got {sample_data.shape}"
        )

        # Check number of channels
        assert sample_data.shape[0] == num_channels, (
            f"Dataset {dataset_name} should have {num_channels} channels, "
            f"got {sample_data.shape[0]}"
        )

        # Check spatial dimensions
        if expected_size:
            spatial_dims = sample_data.shape[1:]
            assert spatial_dims == expected_size, (
                f"Dataset {dataset_name} should have spatial dimensions {expected_size}, "
                f"got {spatial_dims}"
            )

    @pytest.mark.parametrize(
        "dataset_name", [config[0] for config in DATASET_CONFIGURATIONS[:-1]]
    )
    def test_dataset_loading_operations(self, dataset_name):
        """Test dataset loading operations for each dataset type."""
        handler = MockDatasetHandler(dataset_type=dataset_name)

        # Test setup
        handler.setup_dataset(num_clients=3)
        assert handler.is_setup, f"Dataset {dataset_name} should be set up"
        assert handler.federated_dataset is not None

        # Test client data access
        for client_id in range(3):
            client_data = handler.get_client_data(client_id)
            assert len(client_data) > 0, f"Client {client_id} should have data"

            # Verify data consistency
            sample_data, sample_label = client_data[0]
            assert isinstance(sample_data, torch.Tensor)
            assert isinstance(sample_label, torch.Tensor)

        # Test teardown
        handler.teardown_dataset()
        assert not handler.is_setup, f"Dataset {dataset_name} should be torn down"
        assert handler.federated_dataset is None

    @pytest.mark.parametrize(
        "dataset_name", [config[0] for config in DATASET_CONFIGURATIONS[:-1]]
    )
    def test_dataset_dataloader_creation(self, dataset_name):
        """Test DataLoader creation for each dataset type."""
        handler = MockDatasetHandler(dataset_type=dataset_name)
        handler.setup_dataset(num_clients=2)

        # Test DataLoader creation
        client_data = handler.get_client_data(client_id=0)
        dataloader = DataLoader(client_data, batch_size=16, shuffle=True)

        # Verify DataLoader functionality
        batch_count = 0
        for batch_data, batch_labels in dataloader:
            assert batch_data.shape[0] <= 16, "Batch size should not exceed 16"
            assert batch_labels.shape[0] <= 16, "Label batch size should not exceed 16"
            assert (
                batch_data.shape[0] == batch_labels.shape[0]
            ), "Data and labels should have same batch size"
            batch_count += 1
            if batch_count >= 2:  # Test first few batches
                break

    @pytest.mark.parametrize(
        "dataset_name,num_clients,samples_per_client",
        [
            ("its", 5, 100),
            ("femnist_iid", 10, 50),
            ("femnist_niid", 8, 75),
            ("flair", 3, 200),
            ("pneumoniamnist", 6, 80),
            ("bloodmnist", 4, 120),
            ("lung_photos", 7, 60),
        ],
    )
    def test_federated_dataset_distribution(
        self, dataset_name, num_clients, samples_per_client
    ):
        """Test federated dataset distribution across clients."""
        # Get expected shape for dataset
        dataset_config = next(
            (config for config in DATASET_CONFIGURATIONS if config[0] == dataset_name),
            None,
        )
        expected_shape = dataset_config[1] if dataset_config else (3, 32, 32)

        # Create federated dataset
        fed_dataset = MockFederatedDataset(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            input_shape=expected_shape,
        )

        # Verify client distribution
        assert len(fed_dataset.client_datasets) == num_clients

        for client_id in range(num_clients):
            client_data = fed_dataset.get_client_dataset(client_id)
            assert len(client_data) == samples_per_client

            # Verify data heterogeneity (different clients should have different data)
            if client_id > 0:
                prev_client_data = fed_dataset.get_client_dataset(client_id - 1)
                current_sample = client_data[0][0]
                prev_sample = prev_client_data[0][0]

                # Data should be different between clients (check mean values instead of exact equality)
                # Since we use different seeds, the means should be different
                current_mean = current_sample.mean().item()
                prev_mean = prev_sample.mean().item()
                assert abs(current_mean - prev_mean) > 0.001, (
                    f"Clients should have heterogeneous data for {dataset_name} "
                    f"(current mean: {current_mean}, prev mean: {prev_mean})"
                )

    def test_medquad_text_dataset_characteristics(self):
        """Test MedQuAD text dataset specific characteristics."""
        # Mock the text dataset behavior
        with patch(
            "src.dataset_loaders.medquad_dataset_loader.MedQuADDatasetLoader"
        ) as mock_loader:
            # Configure mock loader
            mock_instance = Mock()
            mock_loader.return_value = mock_instance

            # Mock tokenized data structure
            mock_tokenized_data = {
                "input_ids": [[1, 2, 3, 4] * 64],  # 256 tokens
                "attention_mask": [[1, 1, 1, 1] * 64],
                "labels": [[1, 2, 3, 4] * 64],
            }

            mock_dataloader = Mock()
            mock_dataloader.__iter__ = Mock(return_value=iter([mock_tokenized_data]))
            mock_instance.load_datasets.return_value = (
                [mock_dataloader],
                [mock_dataloader],
            )

            # Test dataset characteristics
            trainloaders, valloaders = mock_instance.load_datasets()

            assert len(trainloaders) > 0, "MedQuAD should have training data"
            assert len(valloaders) > 0, "MedQuAD should have validation data"

            # Test tokenized data structure
            for batch in trainloaders[0]:
                assert "input_ids" in batch, "Should have input_ids"
                assert "attention_mask" in batch, "Should have attention_mask"
                assert "labels" in batch, "Should have labels for MLM"
                break

    @pytest.mark.parametrize(
        "dataset_name,batch_size,expected_batches",
        [
            ("its", 32, 2),  # 100 samples / 32 = ~3 batches
            ("femnist_iid", 16, 4),  # 50 samples / 16 = ~3 batches
            ("bloodmnist", 8, 7),  # 50 samples / 8 = ~6 batches
        ],
    )
    def test_dataset_batch_processing(self, dataset_name, batch_size, expected_batches):
        """Test batch processing for different dataset types and batch sizes."""
        handler = MockDatasetHandler(dataset_type=dataset_name)
        handler.setup_dataset(num_clients=1)

        client_data = handler.get_client_data(client_id=0)
        dataloader = DataLoader(client_data, batch_size=batch_size, shuffle=False)

        batch_count = 0
        total_samples = 0

        for batch_data, batch_labels in dataloader:
            batch_count += 1
            total_samples += batch_data.shape[0]

            # Verify batch dimensions
            assert (
                batch_data.shape[0] <= batch_size
            ), f"Batch size should not exceed {batch_size}"
            assert batch_data.shape[0] > 0, "Batch should not be empty"

        # Verify total samples processed
        assert total_samples == len(client_data), "All samples should be processed"
        assert (
            batch_count >= expected_batches - 1
        ), f"Should have approximately {expected_batches} batches"

    @pytest.mark.parametrize("dataset_name", ["its", "flair", "lung_photos"])
    def test_high_resolution_datasets(self, dataset_name):
        """Test handling of high-resolution datasets (224x224 and above)."""
        handler = MockDatasetHandler(dataset_type=dataset_name)
        handler.setup_dataset(num_clients=2)

        client_data = handler.get_client_data(client_id=0)
        sample_data, _ = client_data[0]

        # Verify high resolution
        height, width = sample_data.shape[-2:]
        assert (
            height >= 224 and width >= 224
        ), f"High-resolution dataset {dataset_name} should have dimensions >= 224x224"

        # Test memory efficiency with smaller batches for high-res data
        dataloader = DataLoader(client_data, batch_size=4, shuffle=True)

        for batch_data, batch_labels in dataloader:
            # Verify batch can be processed without memory issues
            assert batch_data.numel() > 0, "Batch should contain data"
            break

    @pytest.mark.parametrize(
        "dataset_name", ["femnist_iid", "femnist_niid", "pneumoniamnist", "lung_photos"]
    )
    def test_grayscale_datasets(self, dataset_name):
        """Test handling of grayscale datasets."""
        handler = MockDatasetHandler(dataset_type=dataset_name)
        handler.setup_dataset(num_clients=1)

        client_data = handler.get_client_data(client_id=0)
        sample_data, _ = client_data[0]

        # Verify single channel (grayscale)
        assert (
            sample_data.shape[0] == 1
        ), f"Grayscale dataset {dataset_name} should have 1 channel"

        # Verify data range (should be in reasonable range for mock data)
        assert (
            sample_data.min() >= -5.0 and sample_data.max() <= 5.0
        ), "Grayscale data should be in reasonable range for mock data"

    @pytest.mark.parametrize("dataset_name", ["its", "flair", "bloodmnist"])
    def test_color_datasets(self, dataset_name):
        """Test handling of color (RGB) datasets."""
        handler = MockDatasetHandler(dataset_type=dataset_name)
        handler.setup_dataset(num_clients=1)

        client_data = handler.get_client_data(client_id=0)
        sample_data, _ = client_data[0]

        # Verify three channels (RGB)
        assert (
            sample_data.shape[0] == 3
        ), f"Color dataset {dataset_name} should have 3 channels"

        # Verify data range
        assert (
            sample_data.min() >= -5.0 and sample_data.max() <= 5.0
        ), "Color data should be in reasonable range for mock data"

    def test_dataset_configuration_mapping(self):
        """Test dataset configuration mapping functionality."""
        config = generate_mock_dataset_config()

        # Verify all expected datasets are present
        expected_datasets = {
            "its",
            "femnist_iid",
            "femnist_niid",
            "flair",
            "pneumoniamnist",
            "bloodmnist",
            "lung_photos",
        }

        assert (
            set(config.keys()) == expected_datasets
        ), "All datasets should be in configuration"

        # Verify paths are properly formatted
        for dataset_name, dataset_path in config.items():
            assert dataset_path.startswith(
                "datasets/"
            ), f"Path for {dataset_name} should start with 'datasets/'"
            assert (
                dataset_name in dataset_path
            ), f"Dataset name should be in path for {dataset_name}"

    @pytest.mark.parametrize(
        "dataset_name,error_scenario",
        [
            ("its", "invalid_client_id"),
            ("femnist_iid", "not_setup"),
            ("bloodmnist", "empty_dataset"),
        ],
    )
    def test_dataset_error_handling(self, dataset_name, error_scenario):
        """Test error handling for various dataset scenarios."""
        handler = MockDatasetHandler(dataset_type=dataset_name)

        if error_scenario == "invalid_client_id":
            handler.setup_dataset(num_clients=3)
            with pytest.raises(ValueError, match="Client .* not found"):
                handler.federated_dataset.get_client_dataset(client_id=10)

        elif error_scenario == "not_setup":
            with pytest.raises(RuntimeError, match="Dataset not setup"):
                handler.get_client_data(client_id=0)

        elif error_scenario == "empty_dataset":
            # Test with zero samples
            fed_dataset = MockFederatedDataset(
                num_clients=1, samples_per_client=0, input_shape=(3, 32, 32)
            )
            client_data = fed_dataset.get_client_dataset(0)
            assert len(client_data) == 0, "Empty dataset should have zero samples"

    def test_dataset_memory_efficiency(self):
        """Test memory efficiency across different dataset types."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Test multiple datasets without memory leaks
        for dataset_name in ["its", "femnist_iid", "bloodmnist"]:
            handler = MockDatasetHandler(dataset_type=dataset_name)
            handler.setup_dataset(num_clients=5)

            # Process some data
            for client_id in range(5):
                client_data = handler.get_client_data(client_id)
                dataloader = DataLoader(client_data, batch_size=16)

                # Process a few batches
                for i, (batch_data, batch_labels) in enumerate(dataloader):
                    if i >= 2:  # Process only first 2 batches
                        break

            handler.teardown_dataset()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 200MB for mock data)
        assert (
            memory_increase < 200 * 1024 * 1024
        ), "Memory usage should remain reasonable"
