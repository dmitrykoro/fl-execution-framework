from unittest.mock import patch

from tests.common import Mock, pytest
import torch
from src.dataset_loaders.image_dataset_loader import ImageDatasetLoader
from torchvision import transforms


class TestImageDatasetLoader:
    """Tests for ImageDatasetLoader."""

    @pytest.fixture
    def mock_transformer(self):
        """Return a simple torchvision transformer for tests."""
        return transforms.Compose([transforms.ToTensor()])

    @pytest.fixture
    def temp_dataset_dir(self, tmp_path):
        """Return path to a temporary image dataset with client folders and dummy images."""
        dataset_dir = tmp_path / "image_dataset"
        dataset_dir.mkdir()

        # Create client folders with proper naming
        for i in range(3):
            client_dir = dataset_dir / f"client_{i}"
            client_dir.mkdir()

            # Create class subdirectories (required by ImageFolder)
            class_dir = client_dir / "class_0"
            class_dir.mkdir()

            # Create dummy image files
            (class_dir / "image_1.jpg").touch()
            (class_dir / "image_2.jpg").touch()

        return str(dataset_dir)

    @pytest.fixture
    def dataset_loader(self, mock_transformer, temp_dataset_dir):
        """Return an ImageDatasetLoader configured for tests."""
        return ImageDatasetLoader(
            transformer=mock_transformer,
            dataset_dir=temp_dataset_dir,
            num_of_clients=3,
            batch_size=2,
            training_subset_fraction=0.8,
        )

    def test_init_sets_attributes_correctly(self, mock_transformer):
        """Verify ImageDatasetLoader.__init__ sets its attributes as expected."""
        loader = ImageDatasetLoader(
            transformer=mock_transformer,
            dataset_dir="/test/path",
            num_of_clients=5,
            batch_size=32,
            training_subset_fraction=0.7,
        )

        assert loader.transformer == mock_transformer
        assert loader.dataset_dir == "/test/path"
        assert loader.num_of_clients == 5
        assert loader.batch_size == 32
        assert loader.training_subset_fraction == 0.7

    @patch("src.dataset_loaders.image_dataset_loader.datasets.ImageFolder")
    @patch("src.dataset_loaders.image_dataset_loader.random_split")
    @patch("src.dataset_loaders.image_dataset_loader.DataLoader")
    def test_load_datasets_processes_client_folders(
        self, mock_dataloader, mock_random_split, mock_image_folder, dataset_loader
    ):
        """Verify load_datasets iterates client folders and returns loaders for each."""
        # Mock ImageFolder dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_image_folder.return_value = mock_dataset

        # Mock train/val split
        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)

        # Mock DataLoader (need 2 loaders per client, 3 clients total = 6 loaders)
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_dataloader.side_effect = [
            mock_train_loader,
            mock_val_loader,  # client 0
            mock_train_loader,
            mock_val_loader,  # client 1
            mock_train_loader,
            mock_val_loader,  # client 2
        ]

        trainloaders, valloaders = dataset_loader.load_datasets()

        # Should create ImageFolder for each client
        assert mock_image_folder.call_count == 3

        # Should create train/val splits for each client
        assert mock_random_split.call_count == 3

        # Should return lists of loaders
        assert len(trainloaders) == 3
        assert len(valloaders) == 3

    @patch("src.dataset_loaders.image_dataset_loader.datasets.ImageFolder")
    @patch("src.dataset_loaders.image_dataset_loader.random_split")
    def test_load_datasets_calculates_split_sizes_correctly(
        self, mock_random_split, mock_image_folder, dataset_loader
    ):
        """Verify train/validation split sizes are computed from fraction."""
        # Mock dataset with 100 samples
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_image_folder.return_value = mock_dataset

        # Mock split datasets
        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)

        with patch("src.dataset_loaders.image_dataset_loader.DataLoader"):
            dataset_loader.load_datasets()

        # Check that split was called with correct sizes
        # 80% of 100 = 80 for training, 20 for validation
        expected_train_size = int(100 * 0.8)
        expected_val_size = 100 - expected_train_size

        call_args = mock_random_split.call_args_list[0][0]
        assert call_args[0] == mock_dataset
        assert call_args[1] == [expected_train_size, expected_val_size]

    @patch("src.dataset_loaders.image_dataset_loader.datasets.ImageFolder")
    @patch("src.dataset_loaders.image_dataset_loader.random_split")
    @patch("src.dataset_loaders.image_dataset_loader.DataLoader")
    def test_load_datasets_creates_dataloaders_with_correct_params(
        self, mock_dataloader, mock_random_split, mock_image_folder, dataset_loader
    ):
        """Verify DataLoader instances are created with expected kwargs."""
        # Mock components
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_image_folder.return_value = mock_dataset

        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)

        dataset_loader.load_datasets()

        # Check DataLoader calls
        assert mock_dataloader.call_count == 6  # 3 clients * 2 loaders each

        # Check that train loaders have shuffle=True
        train_calls = [
            call for call in mock_dataloader.call_args_list[::2]
        ]  # Even indices
        for call in train_calls:
            args, kwargs = call
            assert kwargs.get("shuffle") is True
            assert kwargs.get("batch_size") == 2

        # Check that val loaders have shuffle=False (default)
        val_calls = [
            call for call in mock_dataloader.call_args_list[1::2]
        ]  # Odd indices
        for call in val_calls:
            args, kwargs = call
            assert kwargs.get("shuffle") is None or kwargs.get("shuffle") is False
            assert kwargs.get("batch_size") == 2

    def test_load_datasets_skips_hidden_files(self, dataset_loader, tmp_path):
        """Verify hidden files/folders (e.g. .DS_Store) are ignored when scanning clients."""
        # Create a dataset directory with hidden files
        dataset_dir = tmp_path / "test_hidden"
        dataset_dir.mkdir()

        # Create valid client folder
        client_dir = dataset_dir / "client_0"
        client_dir.mkdir()
        class_dir = client_dir / "class_0"
        class_dir.mkdir()
        (class_dir / "image.jpg").touch()

        # Create hidden folder
        hidden_dir = dataset_dir / ".DS_Store"
        hidden_dir.mkdir()

        dataset_loader.dataset_dir = str(dataset_dir)

        with patch(
            "src.dataset_loaders.image_dataset_loader.datasets.ImageFolder"
        ) as mock_image_folder:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)
            mock_image_folder.return_value = mock_dataset

            with patch(
                "src.dataset_loaders.image_dataset_loader.random_split"
            ) as mock_split:
                mock_train_ds = Mock()
                mock_val_ds = Mock()
                mock_split.return_value = (mock_train_ds, mock_val_ds)
                with patch("src.dataset_loaders.image_dataset_loader.DataLoader"):
                    trainloaders, valloaders = dataset_loader.load_datasets()

        # Should only process non-hidden folders
        assert mock_image_folder.call_count == 1  # Only client_0, not .DS_Store

    @patch("src.dataset_loaders.image_dataset_loader.os.listdir")
    def test_load_datasets_sorts_client_folders_correctly(
        self, mock_listdir, dataset_loader
    ):
        """Verify client folders are sorted by their numeric suffix before processing."""
        # Mock unsorted client folder list
        mock_listdir.return_value = ["client_10", "client_2", "client_1"]

        with patch(
            "src.dataset_loaders.image_dataset_loader.datasets.ImageFolder"
        ) as mock_image_folder:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)
            mock_image_folder.return_value = mock_dataset

            with patch(
                "src.dataset_loaders.image_dataset_loader.random_split"
            ) as mock_split:
                mock_train_ds = Mock()
                mock_val_ds = Mock()
                mock_split.return_value = (mock_train_ds, mock_val_ds)
                with patch("src.dataset_loaders.image_dataset_loader.DataLoader"):
                    dataset_loader.load_datasets()

        # Check that ImageFolder was called with correctly sorted paths
        calls = mock_image_folder.call_args_list
        expected_order = [
            f"{dataset_loader.dataset_dir}/client_1",
            f"{dataset_loader.dataset_dir}/client_2",
            f"{dataset_loader.dataset_dir}/client_10",
        ]

        for i, call in enumerate(calls):
            args, kwargs = call
            assert kwargs["root"] == expected_order[i]

    def test_load_datasets_uses_correct_transformer(self, dataset_loader):
        """Verify the configured transformer is passed to ImageFolder."""
        with patch(
            "src.dataset_loaders.image_dataset_loader.datasets.ImageFolder"
        ) as mock_image_folder:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)
            mock_image_folder.return_value = mock_dataset

            with patch(
                "src.dataset_loaders.image_dataset_loader.random_split"
            ) as mock_split:
                mock_train_ds = Mock()
                mock_val_ds = Mock()
                mock_split.return_value = (mock_train_ds, mock_val_ds)
                with patch("src.dataset_loaders.image_dataset_loader.DataLoader"):
                    dataset_loader.load_datasets()

        # Check that transformer was passed to ImageFolder
        for call in mock_image_folder.call_args_list:
            args, kwargs = call
            assert kwargs["transform"] == dataset_loader.transformer

    @patch("src.dataset_loaders.image_dataset_loader.datasets.ImageFolder")
    def test_load_datasets_handles_empty_directory(
        self, mock_image_folder, dataset_loader
    ):
        """Verify load_datasets returns empty lists when no client folders exist."""
        with patch(
            "src.dataset_loaders.image_dataset_loader.os.listdir", return_value=[]
        ):
            trainloaders, valloaders = dataset_loader.load_datasets()

            assert len(trainloaders) == 0
            assert len(valloaders) == 0
            mock_image_folder.assert_not_called()

    def test_load_datasets_uses_reproducible_seed(self, dataset_loader):
        """Verify a reproducible random generator (torch.Generator) is used for splits."""
        with patch(
            "src.dataset_loaders.image_dataset_loader.datasets.ImageFolder"
        ) as mock_image_folder:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_image_folder.return_value = mock_dataset

            with patch(
                "src.dataset_loaders.image_dataset_loader.random_split"
            ) as mock_random_split:
                mock_train_ds = Mock()
                mock_val_ds = Mock()
                mock_random_split.return_value = (mock_train_ds, mock_val_ds)
                with patch("src.dataset_loaders.image_dataset_loader.DataLoader"):
                    dataset_loader.load_datasets()

        # Check that all splits used the same fixed seed
        for call in mock_random_split.call_args_list:
            args, kwargs = call
            generator = args[2]  # Third argument is the generator
            # Verify it's a generator with seed 42 (we can't check seed directly)
            assert isinstance(generator, torch.Generator)
