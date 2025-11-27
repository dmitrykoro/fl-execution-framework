"""
Unit tests for DatasetHandler class.

Tests dataset setup and teardown operations, file operations mocking,
and dataset configuration handling and validation.
"""

import os
from pathlib import Path
from unittest.mock import patch

from tests.common import Mock, pytest
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
        config.preserve_dataset = False
        return config

    @pytest.fixture
    def mock_directory_handler(self, tmp_path: Path) -> Mock:
        """Create a mock directory handler with temporary paths."""
        handler = Mock(spec=DirectoryHandler)
        handler.dataset_dir = str(tmp_path / "dataset")
        return handler

    @pytest.fixture
    def mock_dataset_config_list(self, tmp_path: Path) -> dict[str, str]:
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
        mock_dataset_config_list: dict[str, str],
    ) -> DatasetHandler:
        """Create DatasetHandler instance for testing."""
        handler = DatasetHandler(
            strategy_config=mock_strategy_config,
            directory_handler=mock_directory_handler,
            dataset_config_list=mock_dataset_config_list,
        )
        return handler

    @pytest.fixture
    def temp_source_dataset(
        self, tmp_path: Path, mock_dataset_config_list: dict[str, str]
    ) -> Path:
        """Create temporary source dataset structure for testing."""
        source_dir = Path(mock_dataset_config_list["its"])
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create client directories with proper naming
        for i in range(10):
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
        mock_dataset_config_list: dict[str, str],
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

    def test_setup_dataset_calls_copy(self, dataset_handler: DatasetHandler) -> None:
        """Test setup_dataset calls copy method."""
        with patch.object(dataset_handler, "_copy_dataset") as mock_copy:
            dataset_handler.setup_dataset()
            mock_copy.assert_called_once_with(
                dataset_handler._strategy_config.num_of_clients
            )

    def test_teardown_dataset_removes_directory_when_not_preserved(
        self, dataset_handler: DatasetHandler
    ) -> None:
        """Test teardown removes dataset when preserve_dataset is False."""
        with patch("shutil.rmtree") as mock_rmtree:
            dataset_handler._strategy_config.preserve_dataset = False
            dataset_handler.teardown_dataset()
            mock_rmtree.assert_called_once_with(dataset_handler.dst_dataset)

    def test_teardown_dataset_preserves_directory_when_requested(
        self, dataset_handler: DatasetHandler
    ) -> None:
        """Test teardown preserves dataset when preserve_dataset is True."""
        with patch("shutil.rmtree") as mock_rmtree:
            dataset_handler._strategy_config.preserve_dataset = True
            dataset_handler.teardown_dataset()
            mock_rmtree.assert_not_called()

    def test_teardown_dataset_handles_removal_error(
        self, dataset_handler: DatasetHandler
    ) -> None:
        """Test teardown handles errors during directory removal."""
        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = OSError("Permission denied")
            dataset_handler._strategy_config.preserve_dataset = False
            dataset_handler.teardown_dataset()

    def test_copy_dataset_copies_correct_number_of_clients(
        self,
        dataset_handler: DatasetHandler,
        temp_source_dataset: Path,
        mock_directory_handler: Mock,
    ) -> None:
        """Test _copy_dataset copies the correct number of client directories."""
        Path(mock_directory_handler.dataset_dir).mkdir(parents=True, exist_ok=True)

        num_clients_to_copy = 5
        dataset_handler._copy_dataset(num_clients_to_copy)

        copied_clients = [
            d
            for d in os.listdir(dataset_handler.dst_dataset)
            if os.path.isdir(os.path.join(dataset_handler.dst_dataset, d))
        ]

        assert len(copied_clients) == num_clients_to_copy
        for i in range(num_clients_to_copy):
            assert f"client_{i}" in copied_clients

    def test_copy_dataset_handles_copy_error(
        self, dataset_handler: DatasetHandler, temp_source_dataset: Path
    ) -> None:
        """Test _copy_dataset handles errors during copying."""
        with patch("shutil.copytree", side_effect=OSError("Copy failed")):
            dataset_handler._copy_dataset(3)

    def test_dataset_handler_with_real_temp_directories(self, tmp_path: Path) -> None:
        """Test DatasetHandler with real temporary directory structure."""
        source_dir = tmp_path / "source" / "its"
        dest_dir = tmp_path / "dest"

        source_dir.mkdir(parents=True)

        for i in range(5):
            client_dir = source_dir / f"client_{i}"
            client_dir.mkdir()

            for label in ["class_0", "class_1"]:
                label_dir = client_dir / label
                label_dir.mkdir()
                (label_dir / f"sample_{i}.png").touch()

        config = StrategyConfig()
        config.num_of_clients = 3
        config.dataset_keyword = "its"
        config.preserve_dataset = False

        dir_handler = Mock(spec=DirectoryHandler)
        dir_handler.dataset_dir = str(dest_dir)

        dataset_config_list = {"its": str(source_dir)}

        handler = DatasetHandler(
            strategy_config=config,
            directory_handler=dir_handler,
            dataset_config_list=dataset_config_list,
        )

        handler.setup_dataset()

        assert dest_dir.exists()
        copied_clients = list(dest_dir.iterdir())
        assert len(copied_clients) == 3
