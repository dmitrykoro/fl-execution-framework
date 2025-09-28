"""
Unit tests for DatasetHandler class.

Tests dataset setup and teardown operations, file operations mocking,
and dataset configuration handling and validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Set
from unittest.mock import call, patch

from tests.common import Mock, np, pytest
from src.data_models.simulation_strategy_config import StrategyConfig
from src.dataset_handlers.dataset_handler import DatasetHandler
from src.output_handlers.directory_handler import DirectoryHandler


class TestDatasetHandler:
    """Test DatasetHandler class."""

    @pytest.fixture
    def mock_strategy_config(self) -> StrategyConfig:
        """Create mock strategy configuration."""
        config = StrategyConfig()
        config.num_of_clients = 5
        config.dataset_keyword = "its"
        config.attack_type = "label_flipping"
        config.num_of_malicious_clients = 2
        config.preserve_dataset = False
        config.attack_ratio = 0.3
        config.gaussian_noise_mean = 0
        config.gaussian_noise_std = 25
        return config

    @pytest.fixture
    def mock_directory_handler(self, tmp_path: Path) -> Mock:
        """Create a mock directory handler with temporary paths."""
        handler = Mock(spec=DirectoryHandler)
        handler.dataset_dir = str(tmp_path / "dataset")
        return handler

    @pytest.fixture
    def mock_dataset_config_list(self, tmp_path: Path) -> Dict[str, str]:
        """Create mock dataset configuration mapping."""
        return {
            "its": str(tmp_path / "source_datasets" / "its"),
            "femnist_iid": str(tmp_path / "source_datasets" / "femnist_iid"),
            "pneumoniamnist": str(tmp_path / "source_datasets" / "pneumoniamnist"),
        }

    @pytest.fixture
    def dataset_handler(
        self,
        mock_strategy_config: StrategyConfig,
        mock_directory_handler: Mock,
        mock_dataset_config_list: Dict[str, str],
    ) -> DatasetHandler:
        """Create DatasetHandler instance for testing."""
        handler = DatasetHandler(
            strategy_config=mock_strategy_config,
            directory_handler=mock_directory_handler,
            dataset_config_list=mock_dataset_config_list,
        )
        # Ensure clean state for each test
        handler.all_poisoned_img_snrs.clear()
        handler.poisoned_client_ids.clear()
        return handler

    @pytest.fixture
    def temp_source_dataset(
        self, tmp_path: Path, mock_dataset_config_list: Dict[str, str]
    ) -> Path:
        """Create temporary source dataset structure for testing."""
        source_dir = Path(mock_dataset_config_list["its"])
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create client directories with proper naming
        for i in range(10):  # Create more clients than needed for testing
            client_dir = source_dir / f"client_{i}"
            client_dir.mkdir(exist_ok=True)

            # Create label directories with sample files
            for label in ["class_0", "class_1"]:
                label_dir = client_dir / label
                label_dir.mkdir(exist_ok=True)

                # Create sample image files
                for j in range(5):
                    sample_file = label_dir / f"sample_{j}.png"
                    sample_file.touch()

        return source_dir

    def test_init(
        self,
        mock_strategy_config: StrategyConfig,
        mock_directory_handler: Mock,
        mock_dataset_config_list: Dict[str, str],
    ) -> None:
        """Test initialization."""
        handler = DatasetHandler(
            strategy_config=mock_strategy_config,
            directory_handler=mock_directory_handler,
            dataset_config_list=mock_dataset_config_list,
        )

        assert handler._strategy_config == mock_strategy_config
        assert handler.dst_dataset == mock_directory_handler.dataset_dir
        assert handler.src_dataset == mock_dataset_config_list["its"]
        assert handler.poisoned_client_ids == set()
        assert handler.all_poisoned_img_snrs == []

    def test_setup_dataset_calls_copy_and_poison(
        self, dataset_handler: DatasetHandler
    ) -> None:
        """Test setup_dataset calls copy and poison methods."""
        with (
            patch.object(dataset_handler, "_copy_dataset") as mock_copy,
            patch.object(dataset_handler, "_poison_clients") as mock_poison,
        ):
            dataset_handler.setup_dataset()

            mock_copy.assert_called_once_with(5)  # num_of_clients
            mock_poison.assert_called_once_with(
                "label_flipping", 2
            )  # attack_type, num_malicious

    @patch("shutil.rmtree")
    def test_teardown_dataset_removes_directory_when_not_preserved(
        self, mock_rmtree: Mock, dataset_handler: DatasetHandler
    ) -> None:
        """Test teardown removes dataset when preserve_dataset is False."""
        dataset_handler._strategy_config.preserve_dataset = False

        dataset_handler.teardown_dataset()

        mock_rmtree.assert_called_once_with(dataset_handler.dst_dataset)

    @patch("shutil.rmtree")
    def test_teardown_dataset_preserves_directory_when_requested(
        self, mock_rmtree, dataset_handler
    ):
        """Test teardown preserves dataset when preserve_dataset is True."""
        dataset_handler._strategy_config.preserve_dataset = True

        dataset_handler.teardown_dataset()

        mock_rmtree.assert_not_called()

    @patch("shutil.rmtree", side_effect=Exception("Removal failed"))
    @patch("logging.error")
    def test_teardown_dataset_handles_removal_error(
        self, mock_log_error, mock_rmtree, dataset_handler
    ):
        """Test teardown handles removal errors gracefully."""
        dataset_handler._strategy_config.preserve_dataset = False

        dataset_handler.teardown_dataset()

        mock_log_error.assert_called_once()
        assert "Error while cleaning up the dataset" in mock_log_error.call_args[0][0]

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("shutil.copytree")
    def test_copy_dataset_copies_correct_number_of_clients(
        self, mock_copytree, mock_isdir, mock_listdir, dataset_handler
    ):
        """Test _copy_dataset copies the specified number of client directories."""
        # Mock directory listing
        mock_listdir.return_value = [
            "client_0",
            "client_1",
            "client_2",
            "client_3",
            "client_4",
            "client_5",
        ]
        mock_isdir.return_value = True

        dataset_handler._copy_dataset(3)

        # Should copy first 3 clients
        assert mock_copytree.call_count == 3
        expected_calls = [
            call(
                src=os.path.join(dataset_handler.src_dataset, f"client_{i}"),
                dst=os.path.join(dataset_handler.dst_dataset, f"client_{i}"),
            )
            for i in range(3)
        ]
        mock_copytree.assert_has_calls(expected_calls)

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("shutil.copytree", side_effect=Exception("Copy failed"))
    @patch("logging.error")
    def test_copy_dataset_handles_copy_error(
        self, mock_log_error, mock_copytree, mock_isdir, mock_listdir, dataset_handler
    ):
        """Test _copy_dataset handles copy errors gracefully."""
        mock_listdir.return_value = ["client_0"]
        mock_isdir.return_value = True

        dataset_handler._copy_dataset(1)

        mock_log_error.assert_called_once()
        assert "Error while preparing dataset" in mock_log_error.call_args[0][0]

    @patch("os.listdir")
    def test_poison_clients_assigns_poisoned_client_ids(
        self, mock_listdir: Mock, dataset_handler: DatasetHandler
    ) -> None:
        """Test _poison_clients correctly assigns poisoned client IDs."""
        mock_listdir.return_value = ["client_0", "client_1", "client_2"]

        with patch.object(dataset_handler, "_flip_labels"):
            dataset_handler._poison_clients("label_flipping", 2)

        expected_ids: Set[int] = {0, 1}  # First 2 clients should be poisoned
        assert dataset_handler.poisoned_client_ids == expected_ids

    def test_poison_clients_raises_error_for_unsupported_attack(
        self, dataset_handler: DatasetHandler
    ) -> None:
        """Test _poison_clients raises error for unsupported attack types."""
        with patch("os.listdir", return_value=["client_0"]):
            with pytest.raises(
                NotImplementedError, match="Not supported attack type: unsupported"
            ):
                dataset_handler._poison_clients("unsupported", 1)

    @patch("os.listdir")
    def test_poison_clients_calls_flip_labels_for_label_flipping(
        self, mock_listdir, dataset_handler
    ):
        """Test _poison_clients calls _flip_labels for label_flipping attack."""
        mock_listdir.return_value = ["client_0", "client_1"]

        with patch.object(dataset_handler, "_flip_labels") as mock_flip:
            dataset_handler._poison_clients("label_flipping", 2)

        expected_calls = [call("client_0"), call("client_1")]
        mock_flip.assert_has_calls(expected_calls)

    @patch("os.listdir")
    def test_poison_clients_calls_add_noise_for_gaussian_noise(
        self, mock_listdir, dataset_handler
    ):
        """Test _poison_clients calls _add_noise for gaussian_noise attack."""
        mock_listdir.return_value = ["client_0"]

        with (
            patch.object(dataset_handler, "_add_noise") as mock_add_noise,
            patch("logging.warning"),
        ):
            dataset_handler._poison_clients("gaussian_noise", 1)

        mock_add_noise.assert_called_once_with("client_0")

    @patch("os.listdir")
    @patch("os.rename")
    def test_flip_labels_renames_directories_correctly(
        self, mock_rename, mock_listdir, dataset_handler
    ):
        """Test _flip_labels renames label directories correctly."""
        # Mock the directory structure
        client_dir = "client_0"
        mock_listdir.side_effect = [
            ["class_0", "class_1"],  # Initial labels
            ["class_0_old", "class_1_old"],  # After first rename
        ]

        with patch("random.choice", side_effect=["class_1", "class_0"]):
            dataset_handler._flip_labels(client_dir)

        # Should rename directories and then rename client directory
        assert (
            mock_rename.call_count >= 3
        )  # At least initial renames + final client rename

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("src.dataset_handlers.dataset_handler.cv2.imread")
    @patch("src.dataset_handlers.dataset_handler.cv2.imwrite")
    @patch("src.dataset_handlers.dataset_handler.np.random.normal")
    def test_add_noise_processes_images_correctly(
        self,
        mock_normal,
        mock_imwrite,
        mock_imread,
        mock_isdir,
        mock_listdir,
        dataset_handler,
    ):
        """Test _add_noise adds Gaussian noise to images correctly."""
        # Setup mocks
        client_dir = "client_0"
        mock_listdir.side_effect = [
            ["class_0"],  # Label folders
            ["image1.png", "image2.jpg"],  # Images in class_0
        ]
        mock_isdir.return_value = True

        # Mock image data
        mock_image = np.ones((32, 32, 3), dtype=np.uint8) * 128
        mock_imread.return_value = mock_image
        mock_imwrite.return_value = True

        # Mock noise generation
        mock_noise = np.ones((32, 32, 3), dtype=np.float32) * 10
        mock_normal.return_value = mock_noise

        # Set attack ratio to poison first image only
        dataset_handler._strategy_config.attack_ratio = 0.5

        dataset_handler._add_noise(client_dir)

        # Should process one image (50% of 2 images = 1 image)
        assert mock_imread.call_count == 1
        assert mock_imwrite.call_count == 1
        assert len(dataset_handler.all_poisoned_img_snrs) == 1

    @patch("os.listdir", side_effect=FileNotFoundError("Directory not found"))
    @patch("logging.error")
    def test_add_noise_handles_missing_directory(
        self, mock_log_error, mock_listdir, dataset_handler
    ):
        """Test _add_noise handles missing client directory gracefully."""
        dataset_handler._add_noise("nonexistent_client")

        mock_log_error.assert_called_once()
        assert "Client directory not found" in mock_log_error.call_args[0][0]

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("src.dataset_handlers.dataset_handler.cv2.imread", return_value=None)
    @patch("logging.error")
    def test_add_noise_handles_failed_image_load(
        self, mock_log_error, mock_imread, mock_isdir, mock_listdir, dataset_handler
    ):
        """Test _add_noise handles failed image loading gracefully."""
        mock_listdir.side_effect = [["class_0"], ["image1.png"]]
        mock_isdir.return_value = True
        dataset_handler._strategy_config.attack_ratio = 1.0

        dataset_handler._add_noise("client_0")

        mock_log_error.assert_called()
        assert "Failed to load image" in mock_log_error.call_args[0][0]

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("src.dataset_handlers.dataset_handler.cv2.imread")
    @patch("src.dataset_handlers.dataset_handler.cv2.imwrite", return_value=False)
    @patch("logging.error")
    @patch("src.dataset_handlers.dataset_handler.np.random.normal")
    def test_add_noise_handles_failed_image_write(
        self,
        mock_normal,
        mock_log_error,
        mock_imwrite,
        mock_imread,
        mock_isdir,
        mock_listdir,
        dataset_handler,
    ):
        """Test _add_noise handles failed image writing gracefully."""
        mock_listdir.side_effect = [["class_0"], ["image1.png"]]
        mock_isdir.return_value = True
        mock_imread.return_value = np.ones((32, 32, 3), dtype=np.uint8) * 128
        mock_normal.return_value = np.ones((32, 32, 3), dtype=np.float32) * 10
        dataset_handler._strategy_config.attack_ratio = 1.0

        dataset_handler._add_noise("client_0")

        mock_log_error.assert_called()
        assert "Failed to write image" in mock_log_error.call_args[0][0]

    def test_assign_poisoned_client_ids_parses_correctly(
        self, dataset_handler: DatasetHandler
    ) -> None:
        """Test _assign_poisoned_client_ids parses client IDs correctly."""
        bad_client_dirs: List[str] = ["client_0", "client_3", "client_7"]

        dataset_handler._assign_poisoned_client_ids(bad_client_dirs)

        expected_ids: Set[int] = {0, 3, 7}
        assert dataset_handler.poisoned_client_ids == expected_ids

    @patch("logging.error")
    @patch("sys.exit")
    def test_assign_poisoned_client_ids_handles_parsing_error(
        self, mock_exit, mock_log_error, dataset_handler
    ):
        """Test _assign_poisoned_client_ids handles parsing errors."""
        bad_client_dirs = ["invalid_client_name"]

        dataset_handler._assign_poisoned_client_ids(bad_client_dirs)

        mock_log_error.assert_called_once()
        mock_exit.assert_called_once_with(-1)
        assert (
            "Error while parsing client dataset folder"
            in mock_log_error.call_args[0][0]
        )

    def test_snr_calculation_in_add_noise(self, dataset_handler):
        """Test SNR calculation in _add_noise method."""
        # Create a simple test case for SNR calculation
        with (
            patch("os.listdir") as mock_listdir,
            patch("os.path.isdir", return_value=True),
            patch("src.dataset_handlers.dataset_handler.cv2.imread") as mock_imread,
            patch(
                "src.dataset_handlers.dataset_handler.cv2.imwrite", return_value=True
            ),
            patch(
                "src.dataset_handlers.dataset_handler.np.random.normal"
            ) as mock_normal,
        ):
            mock_listdir.side_effect = [["class_0"], ["image1.png"]]

            # Create test image and noise
            test_image = np.ones((10, 10, 3), dtype=np.uint8) * 100
            test_noise = np.ones((10, 10, 3), dtype=np.float32) * 10

            mock_imread.return_value = test_image
            mock_normal.return_value = test_noise

            dataset_handler._strategy_config.attack_ratio = 1.0
            dataset_handler._add_noise("client_0")

            # Check that SNR was calculated and stored
            assert len(dataset_handler.all_poisoned_img_snrs) == 1
            snr_value = dataset_handler.all_poisoned_img_snrs[0]
            assert isinstance(snr_value, (int, float))

    @pytest.mark.parametrize(
        "attack_type,expected_method",
        [
            ("label_flipping", "_flip_labels"),
            ("gaussian_noise", "_add_noise"),
        ],
    )
    def test_poison_clients_attack_type_routing(
        self, attack_type, expected_method, dataset_handler
    ):
        """Test _poison_clients routes to correct method based on attack type."""
        with (
            patch("os.listdir", return_value=["client_0"]),
            patch.object(dataset_handler, expected_method) as mock_method,
            patch("logging.warning"),
        ):  # For gaussian_noise logging
            dataset_handler._poison_clients(attack_type, 1)

            mock_method.assert_called_once_with("client_0")

    def test_dataset_handler_with_real_temp_directories(self, tmp_path: Path) -> None:
        """Integration test with real temporary directories."""
        # Create real temporary directory structure
        source_dir = tmp_path / "source"
        dest_dir = tmp_path / "dest"

        source_dir.mkdir()
        dest_dir.mkdir()

        # Create client directories
        for i in range(3):
            client_dir = source_dir / f"client_{i}"
            client_dir.mkdir()
            (client_dir / "data.txt").write_text(f"client {i} data")

        # Create configuration
        config = StrategyConfig()
        config.num_of_clients = 2
        config.dataset_keyword = "test"
        config.attack_type = "label_flipping"
        config.num_of_malicious_clients = 0
        config.preserve_dataset = False

        # Create directory handler mock
        dir_handler = Mock()
        dir_handler.dataset_dir = str(dest_dir)

        # Create dataset config
        dataset_config: Dict[str, str] = {"test": str(source_dir)}

        # Create handler and test copy functionality
        handler = DatasetHandler(config, dir_handler, dataset_config)

        with patch.object(handler, "_poison_clients"):  # Skip poisoning for this test
            handler.setup_dataset()

        # Verify directories were copied
        assert (dest_dir / "client_0").exists()
        assert (dest_dir / "client_1").exists()
        assert not (dest_dir / "client_2").exists()  # Should only copy first 2

        # Verify file contents
        assert (dest_dir / "client_0" / "data.txt").read_text() == "client 0 data"
        assert (dest_dir / "client_1" / "data.txt").read_text() == "client 1 data"
