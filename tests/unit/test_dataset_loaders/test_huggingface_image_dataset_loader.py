"""Unit tests for HuggingFace Image Dataset Loader."""

from unittest.mock import MagicMock, Mock

import pytest
import torch
from PIL import Image

from src.dataset_loaders.huggingface_image_dataset_loader import (
    HuggingFaceImageDataset,
    HuggingFaceImageDatasetLoader,
)


@pytest.fixture
def mock_dataset():
    """Creates a mock HuggingFace dataset.

    Returns:
        MagicMock: A mock dataset with standard attributes.
    """
    mock_item = {"image": Image.new("RGB", (32, 32)), "label": 0}

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    mock_ds.__getitem__.return_value = mock_item

    mock_ds.column_names = ["image", "label"]

    mock_ds.train_test_split.return_value = {"train": mock_ds, "test": mock_ds}
    mock_ds.select.return_value = mock_ds
    mock_ds.shuffle.return_value = mock_ds

    return mock_ds


@pytest.fixture
def mock_load_dataset(mocker, mock_dataset):
    """Mocks the load_dataset function.

    Args:
        mocker: Pytest mocker fixture.
        mock_dataset: The mock dataset fixture.

    Returns:
        Mock: The mocked load_dataset function.
    """
    mock = mocker.patch(
        "src.dataset_loaders.huggingface_image_dataset_loader.load_dataset"
    )
    mock.return_value = {"train": mock_dataset}
    return mock


class TestHuggingFaceImageDataset:
    """Tests for the Dataset wrapper."""

    def test_getitem_transform(self):
        """Verifies item retrieval with transforms applied."""
        transform_mock = Mock(return_value=torch.zeros(3, 28, 28))
        hf_ds_mock = MagicMock()
        img = Image.new("RGB", (32, 32))
        hf_ds_mock.__getitem__.return_value = {"image": img, "label": 5}

        ds = HuggingFaceImageDataset(hf_ds_mock, transform=transform_mock)
        item, label = ds[0]

        transform_mock.assert_called_once()
        assert torch.is_tensor(item)
        assert label == 5

    def test_getitem_no_transform(self):
        """Verifies item retrieval without transforms."""
        hf_ds_mock = MagicMock()
        img = Image.new("RGB", (32, 32))
        hf_ds_mock.__getitem__.return_value = {"image": img, "label": 1}

        ds = HuggingFaceImageDataset(hf_ds_mock, transform=None)
        item, label = ds[0]

        assert isinstance(item, Image.Image)
        assert label == 1


class TestHuggingFaceImageDatasetLoader:
    """Tests for the Loader class."""

    def test_init_defaults(self):
        """Verifies initialization with default values."""
        loader = HuggingFaceImageDatasetLoader("test/path")
        assert loader.num_of_clients == 10
        assert loader.batch_size == 32
        assert loader.transformer is not None

    def test_partition_iid(self, mock_load_dataset):
        """Tests IID partitioning logic."""
        loader = HuggingFaceImageDatasetLoader(
            "test/path", num_of_clients=2, max_samples=100
        )
        mock_ds = mock_load_dataset.return_value["train"]
        mock_ds.__len__.return_value = 100
        mock_ds.column_names = ["image"]

        trainloaders, valloaders = loader.load_datasets()

        assert len(trainloaders) == 2
        assert len(valloaders) == 2
        assert mock_ds.select.called

    def test_partition_non_iid_dirichlet(self, mock_load_dataset):
        """Tests Non-IID Dirichlet partitioning."""
        loader = HuggingFaceImageDatasetLoader(
            "test/path", num_of_clients=2, max_samples=100
        )

        mock_ds = mock_load_dataset.return_value["train"]
        mock_ds.column_names = ["image", "label"]

        def getitem(key):
            if key == "label":
                return [0] * 50 + [1] * 50
            return []

        mock_ds.__getitem__ = Mock(side_effect=getitem)
        mock_ds.__len__.return_value = 100

        trainloaders, valloaders = loader.load_datasets()

        assert len(trainloaders) == 2

    def test_load_optimization(self, mock_load_dataset):
        """Tests max_samples optimization limit."""
        loader = HuggingFaceImageDatasetLoader("test/path", max_samples=50)
        mock_ds = mock_load_dataset.return_value["train"]
        mock_ds.__len__.return_value = 1000

        def getitem(key):
            if key == "label":
                return [0] * 1000
            return {"image": Image.new("RGB", (32, 32)), "label": 0}

        mock_ds.__getitem__.side_effect = getitem
        mock_ds.column_names = ["image", "label"]

        loader.load_datasets()

        mock_ds.shuffle.assert_called()
        mock_ds.select.assert_called()
