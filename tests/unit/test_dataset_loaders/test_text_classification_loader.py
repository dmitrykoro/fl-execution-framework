from unittest.mock import Mock, patch

import numpy as np

from src.dataset_loaders.text_classification_loader import TextClassificationLoader
from tests.common import pytest


class TestTextClassificationLoader:
    """Tests for TextClassificationLoader."""

    @pytest.fixture
    def loader_config(self):
        """Return default config for TextClassificationLoader tests."""
        return {
            "dataset_name": "stanfordnlp/sst2",
            "tokenizer_model": "distilbert-base-uncased",
            "num_of_clients": 3,
            "batch_size": 8,
            "training_subset_fraction": 1.0,
            "max_seq_length": 128,
            "text_column": "text",
            "text2_column": None,
            "label_column": "label",
            "partitioning_strategy": "iid",
            "partitioning_params": None,
        }

    @pytest.fixture
    def loader(self, loader_config):
        """Return a TextClassificationLoader instance."""
        return TextClassificationLoader(**loader_config)

    def test_init_sets_attributes_correctly(self, loader_config):
        """Verify __init__ sets all attributes as expected."""
        loader = TextClassificationLoader(**loader_config)

        assert loader.dataset_name == "stanfordnlp/sst2"
        assert loader.tokenizer_model == "distilbert-base-uncased"
        assert loader.num_of_clients == 3
        assert loader.batch_size == 8
        assert loader.training_subset_fraction == 1.0
        assert loader.max_seq_length == 128
        assert loader.text_column == "text"
        assert loader.text2_column is None
        assert loader.label_column == "label"
        assert loader.partitioning_strategy == "iid"
        assert loader.partitioning_params == {}

    def test_init_defaults_partitioning_params_to_empty_dict(self):
        """Verify partitioning_params defaults to empty dict when None."""
        loader = TextClassificationLoader(
            dataset_name="test",
            tokenizer_model="test-model",
            num_of_clients=2,
            batch_size=4,
            partitioning_params=None,
        )

        assert loader.partitioning_params == {}

    def test_init_preserves_partitioning_params(self):
        """Verify partitioning_params is preserved when provided."""
        params = {"alpha": 0.5}
        loader = TextClassificationLoader(
            dataset_name="test",
            tokenizer_model="test-model",
            num_of_clients=2,
            batch_size=4,
            partitioning_params=params,
        )

        assert loader.partitioning_params == {"alpha": 0.5}

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_uses_validation_split_when_available(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify load_datasets uses 'validation' split over 'test' when available."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=100)
        mock_train.select = Mock(return_value=mock_train)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_val = Mock()
        mock_val.map = Mock(return_value=mock_val)
        mock_val.rename_column = Mock(return_value=mock_val)
        mock_val.set_format = Mock()

        mock_dataset = {"train": mock_train, "validation": mock_val, "test": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                trainloaders, valloaders, num_labels = loader.load_datasets()

        assert mock_val.map.called
        assert num_labels == 2

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_uses_test_split_when_validation_unavailable(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify load_datasets falls back to 'test' split when no validation."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=100)
        mock_train.select = Mock(return_value=mock_train)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_test = Mock()
        mock_test.map = Mock(return_value=mock_test)
        mock_test.rename_column = Mock(return_value=mock_test)
        mock_test.set_format = Mock()

        mock_dataset = {"train": mock_train, "test": mock_test}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                trainloaders, valloaders, num_labels = loader.load_datasets()

        assert mock_test.map.called
        assert num_labels == 2

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_detects_num_labels_from_features(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify num_labels is detected from dataset features when available."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=5)}
        mock_train.__len__ = Mock(return_value=100)
        mock_train.select = Mock(return_value=mock_train)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 2, 3, 4])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                _, _, num_labels = loader.load_datasets()

        assert num_labels == 5

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_counts_unique_labels_as_fallback(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify num_labels falls back to counting unique labels."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(spec=[])}
        mock_train.__len__ = Mock(return_value=100)
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0, 1, 2])
        mock_train.shuffle = Mock(return_value=mock_train)
        mock_train.select = Mock(return_value=mock_train)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                _, _, num_labels = loader.load_datasets()

        assert num_labels == 3

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_applies_training_subset_fraction(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify training_subset_fraction reduces training dataset size."""
        loader.training_subset_fraction = 0.5

        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=100)
        mock_train.select = Mock(return_value=mock_train)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        assert mock_train.select.call_count >= 1
        first_call_args = mock_train.select.call_args_list[0][0][0]
        assert list(first_call_args) == list(range(50))

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_skips_subset_when_fraction_is_one(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify training dataset is not subset when fraction is 1.0."""
        loader.training_subset_fraction = 1.0

        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=100)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        assert mock_train.select.call_count == 3

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_tokenize_dataset_handles_single_text_column(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify tokenization works with single text column."""
        loader.text2_column = None

        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=10)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        assert mock_train.map.called
        tokenize_fn = mock_train.map.call_args[0][0]

        example = {"text": ["Hello world"]}
        tokenize_fn(example)

        mock_tokenizer.assert_called_with(
            ["Hello world"], truncation=True, max_length=128
        )

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_tokenize_dataset_handles_sentence_pairs(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify tokenization works with sentence pairs."""
        loader.text2_column = "text2"

        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=10)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        tokenize_fn = mock_train.map.call_args[0][0]

        example = {"text": ["Hello"], "text2": ["World"]}
        tokenize_fn(example)

        mock_tokenizer.assert_called_with(
            ["Hello"], ["World"], truncation=True, max_length=128
        )

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_tokenize_dataset_renames_label_column(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify label column is renamed to 'labels' for transformers."""
        loader.label_column = "sentiment"

        mock_train = Mock()
        mock_train.features = {"sentiment": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=10)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        mock_train.rename_column.assert_called_with("sentiment", "labels")

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_tokenize_dataset_skips_rename_when_already_labels(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify label column is not renamed when already 'labels'."""
        loader.label_column = "labels"

        mock_train = Mock()
        mock_train.features = {"labels": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=10)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        mock_train.rename_column.assert_not_called()

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_tokenize_dataset_sets_pytorch_format(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify dataset format is set to PyTorch with correct columns."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=10)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1])
        mock_train.shuffle = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            with patch("src.dataset_loaders.text_classification_loader.DataLoader"):
                loader.load_datasets()

        mock_train.set_format.assert_called_with(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    def test_partition_dataset_routes_to_iid(self, loader):
        """Verify _partition_dataset routes to _partition_iid."""
        loader.partitioning_strategy = "iid"
        mock_dataset = Mock()

        with patch.object(loader, "_partition_iid", return_value=[]) as mock_iid:
            loader._partition_dataset(mock_dataset)
            mock_iid.assert_called_once_with(mock_dataset)

    def test_partition_dataset_routes_to_dirichlet(self, loader):
        """Verify _partition_dataset routes to _partition_dirichlet."""
        loader.partitioning_strategy = "dirichlet"
        mock_dataset = Mock()

        with patch.object(loader, "_partition_dirichlet", return_value=[]) as mock_dir:
            loader._partition_dataset(mock_dataset)
            mock_dir.assert_called_once_with(mock_dataset)

    def test_partition_dataset_routes_to_pathological(self, loader):
        """Verify _partition_dataset routes to _partition_pathological."""
        loader.partitioning_strategy = "pathological"
        mock_dataset = Mock()

        with patch.object(
            loader, "_partition_pathological", return_value=[]
        ) as mock_path:
            loader._partition_dataset(mock_dataset)
            mock_path.assert_called_once_with(mock_dataset)

    def test_partition_dataset_raises_on_unknown_strategy(self, loader):
        """Verify _partition_dataset raises ValueError for unknown strategy."""
        loader.partitioning_strategy = "invalid"
        mock_dataset = Mock()

        with pytest.raises(ValueError, match="Unknown partitioning strategy: invalid"):
            loader._partition_dataset(mock_dataset)

    def test_partition_iid_shuffles_dataset(self, loader):
        """Verify _partition_iid shuffles dataset with seed 42."""
        mock_dataset = Mock()
        mock_shuffled = Mock()
        mock_shuffled.__len__ = Mock(return_value=90)
        mock_shuffled.select = Mock(return_value=Mock())
        mock_dataset.shuffle = Mock(return_value=mock_shuffled)

        loader._partition_iid(mock_dataset)

        mock_dataset.shuffle.assert_called_once_with(seed=42)

    def test_partition_iid_divides_evenly(self, loader):
        """Verify _partition_iid divides dataset evenly across clients."""
        loader.num_of_clients = 3

        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=90)
        mock_dataset.shuffle = Mock(return_value=mock_dataset)
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        client_datasets = loader._partition_iid(mock_dataset)

        assert len(client_datasets) == 3

        calls = mock_dataset.select.call_args_list
        assert list(calls[0][0][0]) == list(range(0, 30))
        assert list(calls[1][0][0]) == list(range(30, 60))
        assert list(calls[2][0][0]) == list(range(60, 90))

    def test_partition_iid_gives_remainder_to_last_client(self, loader):
        """Verify _partition_iid gives remaining samples to last client."""
        loader.num_of_clients = 3

        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=91)
        mock_dataset.shuffle = Mock(return_value=mock_dataset)
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        loader._partition_iid(mock_dataset)

        calls = mock_dataset.select.call_args_list
        assert list(calls[0][0][0]) == list(range(0, 30))
        assert list(calls[1][0][0]) == list(range(30, 60))
        assert list(calls[2][0][0]) == list(range(60, 91))

    def test_partition_dirichlet_uses_alpha_from_params(self, loader):
        """Verify _partition_dirichlet uses alpha from partitioning_params."""
        loader.partitioning_params = {"alpha": 0.3}

        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=np.array([0, 0, 1, 1]))
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        with patch("numpy.random.dirichlet") as mock_dirichlet:
            mock_dirichlet.return_value = np.array([0.5, 0.3, 0.2])
            loader._partition_dirichlet(mock_dataset)

            calls = mock_dirichlet.call_args_list
            for call in calls:
                assert call[0][0][0] == 0.3

    def test_partition_dirichlet_defaults_alpha_to_point_five(self, loader):
        """Verify _partition_dirichlet defaults alpha to 0.5."""
        loader.partitioning_params = {}

        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=np.array([0, 0, 1, 1]))
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        with patch("numpy.random.dirichlet") as mock_dirichlet:
            mock_dirichlet.return_value = np.array([0.5, 0.3, 0.2])
            loader._partition_dirichlet(mock_dataset)

            calls = mock_dirichlet.call_args_list
            for call in calls:
                assert call[0][0][0] == 0.5

    def test_partition_dirichlet_handles_empty_partition(self, loader):
        """Verify _partition_dirichlet creates minimal dataset for empty partitions."""
        loader.num_of_clients = 3

        mock_dataset = Mock()
        labels = np.array([0, 0])  # Only 2 samples, 1 class
        mock_dataset.__getitem__ = Mock(return_value=labels)
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        with patch("numpy.random.dirichlet") as mock_dirichlet:
            mock_dirichlet.return_value = np.array([1.0, 0.0, 0.0])
            with patch("numpy.random.shuffle"):
                client_datasets = loader._partition_dirichlet(mock_dataset)

        assert len(client_datasets) == 3

    def test_partition_pathological_uses_num_classes_from_params(self, loader):
        """Verify _partition_pathological uses num_classes_per_partition."""
        loader.partitioning_params = {"num_classes_per_partition": 3}
        loader.num_of_clients = 2

        mock_dataset = Mock()
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        mock_dataset.__getitem__ = Mock(return_value=labels)
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        with patch("numpy.random.shuffle"):
            client_datasets = loader._partition_pathological(mock_dataset)

        assert len(client_datasets) == 2

    def test_partition_pathological_defaults_to_two_classes(self, loader):
        """Verify _partition_pathological defaults to 2 classes per client."""
        loader.partitioning_params = {}
        loader.num_of_clients = 2

        mock_dataset = Mock()
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        mock_dataset.__getitem__ = Mock(return_value=labels)
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        with patch("numpy.random.shuffle"):
            client_datasets = loader._partition_pathological(mock_dataset)

        assert len(client_datasets) == 2

    def test_partition_pathological_handles_empty_partition(self, loader):
        """Verify _partition_pathological creates minimal dataset for empty partitions."""
        loader.num_of_clients = 5
        loader.partitioning_params = {"num_classes_per_partition": 2}

        mock_dataset = Mock()
        labels = np.array([0, 0])  # Only 2 samples, 1 class
        mock_dataset.__getitem__ = Mock(return_value=labels)
        mock_dataset.select = Mock(side_effect=lambda x: Mock())

        with patch("numpy.random.shuffle"):
            client_datasets = loader._partition_pathological(mock_dataset)

        assert len(client_datasets) == 5

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    @patch("src.dataset_loaders.text_classification_loader.DataLoader")
    def test_load_datasets_creates_correct_number_of_dataloaders(
        self, mock_dataloader_cls, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify load_datasets creates train and val loaders for each client."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=90)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0])
        mock_train.shuffle = Mock(return_value=mock_train)
        mock_train.select = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            trainloaders, valloaders, _ = loader.load_datasets()

        assert len(trainloaders) == 3
        assert len(valloaders) == 3

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    @patch("src.dataset_loaders.text_classification_loader.DataLoader")
    def test_load_datasets_shuffles_train_loaders(
        self, mock_dataloader_cls, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify train DataLoaders have shuffle=True."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=90)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0])
        mock_train.shuffle = Mock(return_value=mock_train)
        mock_train.select = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            loader.load_datasets()

        train_calls = [
            call
            for i, call in enumerate(mock_dataloader_cls.call_args_list)
            if i % 2 == 0
        ]
        for call in train_calls:
            _, kwargs = call
            assert kwargs.get("shuffle") is True

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    @patch("src.dataset_loaders.text_classification_loader.DataLoader")
    def test_load_datasets_does_not_shuffle_val_loaders(
        self, mock_dataloader_cls, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify validation DataLoaders have shuffle=False."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=90)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0])
        mock_train.shuffle = Mock(return_value=mock_train)
        mock_train.select = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            loader.load_datasets()

        val_calls = [
            call
            for i, call in enumerate(mock_dataloader_cls.call_args_list)
            if i % 2 == 1
        ]
        for call in val_calls:
            _, kwargs = call
            assert kwargs.get("shuffle") is False

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    @patch("src.dataset_loaders.text_classification_loader.DataLoader")
    def test_load_datasets_uses_correct_batch_size(
        self, mock_dataloader_cls, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify DataLoaders use configured batch_size."""
        loader.batch_size = 16

        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=90)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0])
        mock_train.shuffle = Mock(return_value=mock_train)
        mock_train.select = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ):
            loader.load_datasets()

        for call in mock_dataloader_cls.call_args_list:
            _, kwargs = call
            assert kwargs.get("batch_size") == 16

    @patch("src.dataset_loaders.text_classification_loader.load_dataset")
    @patch("src.dataset_loaders.text_classification_loader.AutoTokenizer")
    def test_load_datasets_uses_data_collator(
        self, mock_tokenizer_cls, mock_load_dataset, loader
    ):
        """Verify DataLoaders use DataCollatorWithPadding."""
        mock_train = Mock()
        mock_train.features = {"label": Mock(num_classes=2)}
        mock_train.__len__ = Mock(return_value=90)
        mock_train.map = Mock(return_value=mock_train)
        mock_train.rename_column = Mock(return_value=mock_train)
        mock_train.set_format = Mock()
        mock_train.__getitem__ = Mock(return_value=[0, 1, 0])
        mock_train.shuffle = Mock(return_value=mock_train)
        mock_train.select = Mock(return_value=mock_train)

        mock_dataset = {"train": mock_train, "validation": Mock()}
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        with patch(
            "src.dataset_loaders.text_classification_loader.DataCollatorWithPadding"
        ) as mock_collator_cls:
            mock_collator = Mock()
            mock_collator_cls.return_value = mock_collator

            with patch(
                "src.dataset_loaders.text_classification_loader.DataLoader"
            ) as mock_dl_cls:
                loader.load_datasets()

                mock_collator_cls.assert_called_once_with(tokenizer=mock_tokenizer)

                for call in mock_dl_cls.call_args_list:
                    _, kwargs = call
                    assert kwargs.get("collate_fn") == mock_collator
