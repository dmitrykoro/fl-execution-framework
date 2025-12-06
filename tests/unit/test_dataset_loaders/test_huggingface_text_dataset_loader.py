"""Unit tests for HuggingFace Text Dataset Loader."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from src.dataset_loaders.huggingface_text_dataset_loader import (
    HuggingFaceTextDatasetLoader,
)


@pytest.fixture
def mock_tokenizer():
    """Mocks AutoTokenizer.

    Yields:
        Mock: The mocked AutoTokenizer.
    """
    with patch(
        "src.dataset_loaders.huggingface_text_dataset_loader.AutoTokenizer"
    ) as mock:
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock.from_pretrained.return_value = tokenizer
        yield mock


@pytest.fixture
def mock_dataset_pkg():
    """Mocks datasets package.

    Yields:
        MagicMock: The mocked dataset object.
    """
    with patch(
        "src.dataset_loaders.huggingface_text_dataset_loader.load_dataset"
    ) as mock:
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 100
        mock_ds.column_names = ["text", "label"]

        mock_ds.__getitem__.side_effect = (
            lambda x: [0, 1] * 50 if x == "label" else None
        )

        mock_ds.map.return_value = mock_ds
        mock_ds.remove_columns.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.train_test_split.return_value = {"train": mock_ds, "test": mock_ds}

        mock.return_value = {"train": mock_ds}
        yield mock_ds


@pytest.fixture
def mock_dataloader():
    """Mocks torch DataLoader.

    Yields:
        Mock: The mocked DataLoader.
    """
    with patch(
        "src.dataset_loaders.huggingface_text_dataset_loader.DataLoader"
    ) as mock:
        yield mock


class TestHuggingFaceTextDatasetLoader:
    """Tests for Text Dataset Loader."""

    def test_init_defaults(self):
        """Verifies initialization with default values."""
        loader = HuggingFaceTextDatasetLoader("test/path")
        assert loader.num_of_clients == 5
        assert loader.batch_size == 16
        assert loader.chunk_size == 256
        assert loader.mlm_probability == 0.15

    def test_load_datasets_flow(
        self, mock_dataset_pkg, mock_tokenizer, mock_dataloader
    ):
        """Verifies the full load_datasets flow."""
        loader = HuggingFaceTextDatasetLoader(
            "test/path", num_of_clients=2, max_samples=20
        )

        trainloaders, valloaders = loader.load_datasets()

        assert len(trainloaders) == 2
        assert len(valloaders) == 2

        mock_tokenizer.from_pretrained.assert_called_with(loader.model_name)

        assert mock_dataset_pkg.map.call_count >= 2

    def test_load_datasets_iid(self, mock_dataset_pkg, mock_tokenizer, mock_dataloader):
        """Verifies IID loading when no labels present."""
        loader = HuggingFaceTextDatasetLoader("test/path", num_of_clients=2)

        mock_dataset_pkg.column_names = ["text"]

        loader.load_datasets()

        assert mock_dataloader.call_count > 0

    def test_dataset_size_limit(
        self, mock_dataset_pkg, mock_tokenizer, mock_dataloader
    ):
        """Verifies max_samples limit."""
        loader = HuggingFaceTextDatasetLoader("test/path", max_samples=10)
        mock_dataset_pkg.__len__.return_value = 100

        loader.load_datasets()

        mock_dataset_pkg.shuffle.assert_called()
        mock_dataset_pkg.select.assert_called()
