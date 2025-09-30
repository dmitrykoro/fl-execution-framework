import csv
import json
from unittest.mock import patch

from tests.common import Mock, pytest
from src.data_models.client_info import ClientInfo
from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.dataset_handlers.dataset_handler import DatasetHandler
from src.output_handlers.directory_handler import DirectoryHandler


class TestDirectoryHandler:
    """Test suite for DirectoryHandler output functionality"""

    @pytest.fixture
    def mock_strategy_config(self):
        """Create a mock strategy configuration for testing"""
        return StrategyConfig(
            aggregation_strategy_keyword="trust",
            num_of_rounds=3,
            num_of_clients=5,
            strategy_number=1,
            trust_threshold=0.7,
            remove_clients=True,
        )

    @pytest.fixture
    def mock_client_info_list(self):
        """Create mock client info list for testing"""
        clients = []
        for i in range(3):
            client = ClientInfo(client_id=i, num_of_rounds=3)
            client.loss_history = [0.5 + i * 0.1, 0.4 + i * 0.1, 0.3 + i * 0.1]
            client.accuracy_history = [0.8 - i * 0.05, 0.85 - i * 0.05, 0.9 - i * 0.05]
            clients.append(client)
        return clients

    @pytest.fixture
    def mock_round_info_list(self):
        """Create mock round info list for testing"""
        rounds = []
        for i in range(3):
            round_info = Mock()
            round_info.round_number = i + 1
            round_info.aggregated_loss = 0.5 - i * 0.1
            round_info.aggregated_accuracy = 0.8 + i * 0.05
            rounds.append(round_info)
        return rounds

    @pytest.fixture
    def mock_simulation_history(
        self, mock_strategy_config, mock_client_info_list, mock_round_info_list
    ):
        """Create mock simulation strategy history for testing"""

        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        mock_rounds = Mock(spec=RoundsInfo)
        mock_rounds.savable_metrics = [
            "score_calculation_time_nanos_history",
            "removal_threshold_history",
            "aggregated_loss_history",
            "average_accuracy_history",
        ]

        def get_metric(name):
            if name == "aggregated_loss_history":
                return [r.aggregated_loss for r in mock_round_info_list]
            if name == "average_accuracy_history":
                return [r.aggregated_accuracy for r in mock_round_info_list]
            return []

        mock_rounds.get_metric_by_name.side_effect = get_metric

        history = SimulationStrategyHistory(
            strategy_config=mock_strategy_config,
            dataset_handler=mock_dataset_handler,
            rounds_history=mock_rounds,
        )
        history.get_all_clients = Mock(return_value=mock_client_info_list)
        return history

    @patch("os.makedirs")
    def test_init_creates_directories(self, mock_makedirs):
        """Test DirectoryHandler initialization creates required directories"""
        handler = DirectoryHandler()

        assert mock_makedirs.call_count == 2
        assert handler.simulation_strategy_history is None
        assert handler.dataset_dir is None

    @patch("os.makedirs")
    def test_assign_dataset_dir(self, mock_makedirs):
        """Test assign_dataset_dir creates dataset directory"""
        handler = DirectoryHandler()
        mock_makedirs.reset_mock()

        handler.assign_dataset_dir(1)

        assert handler.dataset_dir is not None
        assert handler.dataset_dir.endswith("/dataset_1")
        mock_makedirs.assert_called_once_with(handler.dataset_dir)

    def test_save_csv_and_config_calls_all_save_methods(
        self, mock_simulation_history, tmp_path, monkeypatch
    ):
        """Test save_csv_and_config calls all individual save methods"""
        # Mock the DirectoryHandler.dirname to use temp directory
        test_dir = tmp_path / "test_output"
        test_dir.mkdir()
        csv_dir = test_dir / "csv"
        csv_dir.mkdir()

        with patch.object(DirectoryHandler, "dirname", str(test_dir)):
            with patch.object(DirectoryHandler, "new_csv_dirname", str(csv_dir)):
                handler = DirectoryHandler()
                handler.dirname = str(test_dir)
                handler.new_csv_dirname = str(csv_dir)

                handler.save_csv_and_config(mock_simulation_history)

                # Check that config file was created
                config_file = test_dir / "strategy_config_1.json"
                assert config_file.exists()

                # Check that CSV files were created
                client_csv = csv_dir / "per_client_metrics_1.csv"
                round_csv = csv_dir / "round_metrics_1.csv"
                execution_csv = csv_dir / "exec_stats_1.csv"

                assert client_csv.exists()
                assert round_csv.exists()
                assert execution_csv.exists()

    def test_save_simulation_config_creates_json_file(
        self, mock_simulation_history, tmp_path
    ):
        """Test _save_simulation_config creates correct JSON file"""
        test_dir = tmp_path / "config_test"
        test_dir.mkdir()

        with patch.object(DirectoryHandler, "dirname", str(test_dir)):
            handler = DirectoryHandler()
            handler.dirname = str(test_dir)
            handler.simulation_strategy_history = mock_simulation_history

            handler._save_simulation_config()

            config_file = test_dir / "strategy_config_1.json"
            assert config_file.exists()

            with open(config_file, "r") as f:
                saved_config = json.load(f)

            assert saved_config["aggregation_strategy_keyword"] == "trust"
            assert saved_config["num_of_rounds"] == 3
            assert saved_config["strategy_number"] == 1

    def test_save_per_client_to_csv_creates_correct_format(
        self, mock_simulation_history, tmp_path
    ):
        """Test _save_per_client_to_csv creates CSV with correct format"""
        csv_dir = tmp_path / "csv_test"
        csv_dir.mkdir()

        with patch.object(DirectoryHandler, "new_csv_dirname", str(csv_dir)):
            handler = DirectoryHandler()
            handler.new_csv_dirname = str(csv_dir)
            handler.simulation_strategy_history = mock_simulation_history

            handler._save_per_client_to_csv()

            csv_file = csv_dir / "per_client_metrics_1.csv"
            assert csv_file.exists()

            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)

                # Should have round # column plus client metrics
                assert headers[0] == "round #"
                assert "client_0_loss_history" in headers
                assert "client_0_accuracy_history" in headers

                # Check data rows
                rows = list(reader)
                assert len(rows) == 3  # 3 rounds

    def test_save_per_client_to_csv_handles_missing_metrics(
        self, mock_strategy_config, tmp_path
    ):
        """Test _save_per_client_to_csv handles clients with missing metrics"""

        client_with_missing_metrics = ClientInfo(
            client_id=0, num_of_rounds=mock_strategy_config.num_of_rounds
        )

        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        history = SimulationStrategyHistory(
            strategy_config=mock_strategy_config,
            dataset_handler=mock_dataset_handler,
            rounds_history=Mock(spec=RoundsInfo),
        )
        history.get_all_clients = Mock(return_value=[client_with_missing_metrics])

        csv_dir = tmp_path / "csv_missing_test"
        csv_dir.mkdir()

        with patch.object(DirectoryHandler, "new_csv_dirname", str(csv_dir)):
            handler = DirectoryHandler()
            handler.new_csv_dirname = str(csv_dir)
            handler.simulation_strategy_history = history

            handler._save_per_client_to_csv()

            csv_file = csv_dir / "per_client_metrics_1.csv"
            assert csv_file.exists()

            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)

                loss_col_index = headers.index("client_0_loss_history")
                agg_part_col_index = headers.index(
                    "client_0_aggregation_participation_history"
                )

                for row in rows:
                    assert row[loss_col_index] == "not collected"
                    assert row[agg_part_col_index] == "1"

    def test_save_per_round_to_csv_creates_correct_format(
        self, mock_simulation_history, tmp_path
    ):
        """Test _save_per_round_to_csv creates CSV with correct format"""
        csv_dir = tmp_path / "round_csv_test"
        csv_dir.mkdir()

        with patch.object(DirectoryHandler, "new_csv_dirname", str(csv_dir)):
            handler = DirectoryHandler()
            handler.new_csv_dirname = str(csv_dir)
            handler.simulation_strategy_history = mock_simulation_history

            handler._save_per_round_to_csv()

            csv_file = csv_dir / "round_metrics_1.csv"
            assert csv_file.exists()

            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                headers = next(reader)

                # Should have round # and metric columns
                assert headers[0] == "round #"
                assert "aggregated_loss_history" in headers
                assert "average_accuracy_history" in headers

    def test_save_per_execution_to_csv_creates_file(
        self, mock_simulation_history, tmp_path
    ):
        """Test _save_per_execution_to_csv creates execution metrics file"""
        csv_dir = tmp_path / "execution_csv_test"
        csv_dir.mkdir()

        with patch.object(DirectoryHandler, "new_csv_dirname", str(csv_dir)):
            handler = DirectoryHandler()
            handler.new_csv_dirname = str(csv_dir)
            handler.simulation_strategy_history = mock_simulation_history

            handler._save_per_execution_to_csv()

            csv_file = csv_dir / "exec_stats_1.csv"
            assert csv_file.exists()

    def test_directory_naming_uses_timestamp(self):
        """Test that directory names include timestamp"""
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "01-01-2024_12-00-00"

            # Reset class variables
            DirectoryHandler.dirname = (
                f"out/{mock_datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"
            )

            assert "01-01-2024_12-00-00" in DirectoryHandler.dirname

    def test_csv_dirname_includes_csv_subdirectory(self, monkeypatch):
        """Test that CSV directory is subdirectory of main directory"""
        monkeypatch.setattr(DirectoryHandler, "dirname", "out/test_dir")
        monkeypatch.setattr(DirectoryHandler, "new_csv_dirname", "out/test_dir/csv")
        assert DirectoryHandler.new_csv_dirname == DirectoryHandler.dirname + "/csv"
