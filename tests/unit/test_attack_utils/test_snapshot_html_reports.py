"""
Tests for HTML and JSON reporting utilities in attack snapshots.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.attack_utils.snapshot_html_reports import (
    _extract_attack_params_for_display,
    _generate_index_html,
    _generate_main_dashboard_html,
    _get_snapshots_dir_checked,
    _split_composite_attack_info,
    generate_main_dashboard,
    generate_snapshot_index,
    generate_summary_json,
)


class TestGetSnapshotsDirChecked:
    """Test suite for _get_snapshots_dir_checked function."""

    def test_should_return_path_when_directory_exists(self, tmp_path: Path) -> None:
        """Test that path is returned when snapshots directory exists."""
        output_dir = tmp_path / "output"
        snapshots_dir = output_dir / "attack_snapshots_0"
        snapshots_dir.mkdir(parents=True)

        result = _get_snapshots_dir_checked(str(output_dir), strategy_number=0)

        assert result is not None
        assert result == snapshots_dir
        assert result.exists()

    def test_should_return_none_when_directory_missing(self, tmp_path: Path) -> None:
        """Test that None is returned when snapshots directory missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("src.attack_utils.snapshot_html_reports.logging") as mock_logging:
            result = _get_snapshots_dir_checked(str(output_dir), strategy_number=0)

        assert result is None
        mock_logging.warning.assert_called_once()

    @pytest.mark.parametrize("strategy_number", [0, 1, 2])
    def test_should_handle_different_strategy_numbers(
        self, tmp_path: Path, strategy_number: int
    ) -> None:
        """Test that different strategy numbers are handled correctly."""
        output_dir = tmp_path / "output"
        snapshots_dir = output_dir / f"attack_snapshots_{strategy_number}"
        snapshots_dir.mkdir(parents=True)

        result = _get_snapshots_dir_checked(str(output_dir), strategy_number)

        assert result is not None
        assert result == snapshots_dir
        assert result.name == f"attack_snapshots_{strategy_number}"


class TestExtractAttackParamsForDisplay:
    """Test suite for _extract_attack_params_for_display function."""

    def test_should_return_empty_list_for_label_flipping(self) -> None:
        """Test that label flipping returns empty params list."""
        config = {"attack_type": "label_flipping"}
        result = _extract_attack_params_for_display("label_flipping", config)
        assert result == []

    def test_should_extract_gaussian_noise_params(self) -> None:
        """Test that gaussian noise params are extracted."""
        config = {
            "attack_type": "gaussian_noise",
            "target_noise_snr": 10,
            "attack_ratio": 0.3,
        }
        result = _extract_attack_params_for_display("gaussian_noise", config)

        assert "SNR=10dB" in result
        assert "ratio=0.3" in result

    def test_should_extract_token_replacement_params(self) -> None:
        """Test that token replacement params are extracted."""
        config = {
            "attack_type": "token_replacement",
            "target_vocabulary": "negative",
            "replacement_strategy": "random",
            "replacement_probability": 0.5,
        }
        result = _extract_attack_params_for_display("token_replacement", config)

        assert "vocab=negative" in result
        assert "strategy=random" in result
        assert "prob=0.5" in result

    def test_should_handle_missing_params(self) -> None:
        """Test that missing params are handled gracefully."""
        config = {"attack_type": "gaussian_noise"}
        result = _extract_attack_params_for_display("gaussian_noise", config)

        assert any("SNR=?" in p for p in result)
        assert any("ratio=?" in p for p in result)

    @pytest.mark.parametrize(
        "attack_type,config,expected_count",
        [
            ("label_flipping", {}, 0),
            ("gaussian_noise", {"target_noise_snr": 10, "attack_ratio": 0.3}, 2),
            (
                "token_replacement",
                {
                    "target_vocabulary": "neg",
                    "replacement_strategy": "rand",
                    "replacement_probability": 0.5,
                },
                3,
            ),
            ("unknown_type", {}, 0),
        ],
    )
    def test_extract_params_variations(
        self, attack_type: str, config: dict, expected_count: int
    ) -> None:
        """Test various parameter extraction scenarios."""
        result = _extract_attack_params_for_display(attack_type, config)
        assert len(result) == expected_count


class TestSplitCompositeAttackInfo:
    """Test suite for _split_composite_attack_info function."""

    def test_should_split_single_attack(self) -> None:
        """Test that single attack is handled correctly."""
        attack_type = "label_flipping"
        configs = [{"attack_type": "label_flipping"}]

        result = _split_composite_attack_info(attack_type, configs)

        assert len(result) == 1
        assert result[0]["type"] == "label_flipping"
        assert result[0]["params"] == []

    def test_should_split_composite_attack(self) -> None:
        """Test that composite attack is split correctly."""
        attack_type = "label_flipping_gaussian_noise"
        configs = [
            {"attack_type": "label_flipping"},
            {"attack_type": "gaussian_noise", "target_noise_snr": 10},
        ]

        result = _split_composite_attack_info(attack_type, configs)

        assert len(result) == 2
        assert result[0]["type"] == "label_flipping"
        assert result[1]["type"] == "gaussian_noise"
        assert len(result[1]["params"]) > 0

    def test_should_include_params_for_each_attack(self) -> None:
        """Test that params are included for each attack type."""
        attack_type = "gaussian_noise_token_replacement"
        configs = [
            {"attack_type": "gaussian_noise", "target_noise_snr": 15},
            {
                "attack_type": "token_replacement",
                "target_vocabulary": "negative",
            },
        ]

        result = _split_composite_attack_info(attack_type, configs)

        assert len(result) == 2
        assert len(result[0]["params"]) > 0
        assert len(result[1]["params"]) > 0


class TestGenerateSummaryJson:
    """Test suite for generate_summary_json function."""

    @pytest.fixture
    def mock_snapshot_data(self, tmp_path: Path) -> tuple:
        """Create mock snapshot data for testing."""
        output_dir = tmp_path / "output"
        snapshots_dir = output_dir / "attack_snapshots_0"
        snapshots_dir.mkdir(parents=True)

        snapshot1 = snapshots_dir / "client_0" / "round_1"
        snapshot1.mkdir(parents=True)
        snapshot1_file = snapshot1 / "label_flipping.pkl"
        snapshot1_file.touch()

        return output_dir, snapshots_dir

    def test_should_return_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Test that function returns early when snapshots dir missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("src.attack_utils.snapshot_html_reports.logging"):
            generate_summary_json(str(output_dir))

        summary_file = output_dir / "attack_snapshots_0" / "summary.json"
        assert not summary_file.exists()

    @patch("src.attack_utils.snapshot_html_reports.list_attack_snapshots")
    @patch("src.attack_utils.snapshot_html_reports.load_attack_snapshot")
    @patch("src.attack_utils.snapshot_html_reports.get_snapshot_summary")
    def test_should_generate_summary_json(
        self,
        mock_get_summary: Mock,
        mock_load_snapshot: Mock,
        mock_list_snapshots: Mock,
        mock_snapshot_data: tuple,
    ) -> None:
        """Test that summary.json is generated correctly."""
        output_dir, snapshots_dir = mock_snapshot_data

        mock_get_summary.return_value = {"total_snapshots": 2}
        mock_list_snapshots.return_value = [
            snapshots_dir / "client_0" / "round_1" / "label_flipping.pkl"
        ]
        mock_load_snapshot.return_value = {
            "metadata": {
                "client_id": 0,
                "round_num": 1,
                "attack_type": "label_flipping",
            }
        }

        generate_summary_json(str(output_dir))

        summary_file = snapshots_dir / "summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            data = json.load(f)

        assert "experiment" in data
        assert "attack_summary" in data
        assert "attack_timeline" in data
        assert data["experiment"]["run_id"] == output_dir.name

    @patch("src.attack_utils.snapshot_html_reports.list_attack_snapshots")
    @patch("src.attack_utils.snapshot_html_reports.load_attack_snapshot")
    @patch("src.attack_utils.snapshot_html_reports.get_snapshot_summary")
    def test_should_include_run_config_when_provided(
        self,
        mock_get_summary: Mock,
        mock_load_snapshot: Mock,
        mock_list_snapshots: Mock,
        mock_snapshot_data: tuple,
    ) -> None:
        """Test that run config is included when provided."""
        output_dir, snapshots_dir = mock_snapshot_data

        mock_get_summary.return_value = {"total_snapshots": 1}
        mock_list_snapshots.return_value = []

        run_config = {"num_of_clients": 10, "num_of_rounds": 50}

        generate_summary_json(str(output_dir), run_config=run_config)

        summary_file = snapshots_dir / "summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            data = json.load(f)

        assert data["experiment"]["total_clients"] == 10
        assert data["experiment"]["total_rounds"] == 50

    @patch("src.attack_utils.snapshot_html_reports.list_attack_snapshots")
    @patch("src.attack_utils.snapshot_html_reports.load_attack_snapshot")
    @patch("src.attack_utils.snapshot_html_reports.get_snapshot_summary")
    def test_should_build_attack_timeline(
        self,
        mock_get_summary: Mock,
        mock_load_snapshot: Mock,
        mock_list_snapshots: Mock,
        mock_snapshot_data: tuple,
    ) -> None:
        """Test that attack timeline is built correctly."""
        output_dir, snapshots_dir = mock_snapshot_data

        mock_get_summary.return_value = {"total_snapshots": 2}
        mock_list_snapshots.return_value = [
            snapshots_dir / "client_0" / "round_1" / "attack1.pkl",
            snapshots_dir / "client_0" / "round_2" / "attack2.pkl",
        ]
        mock_load_snapshot.side_effect = [
            {
                "metadata": {
                    "client_id": 0,
                    "round_num": 1,
                    "attack_type": "label_flipping",
                }
            },
            {
                "metadata": {
                    "client_id": 0,
                    "round_num": 2,
                    "attack_type": "gaussian_noise",
                }
            },
        ]

        generate_summary_json(str(output_dir))

        summary_file = snapshots_dir / "summary.json"
        with open(summary_file) as f:
            data = json.load(f)

        assert "0" in data["attack_timeline"]
        assert "1" in data["attack_timeline"]["0"]
        assert "2" in data["attack_timeline"]["0"]


class TestGenerateSnapshotIndex:
    """Test suite for generate_snapshot_index function."""

    @pytest.fixture
    def mock_snapshot_dir(self, tmp_path: Path) -> tuple:
        """Create mock snapshot directory."""
        output_dir = tmp_path / "output"
        snapshots_dir = output_dir / "attack_snapshots_0"
        snapshots_dir.mkdir(parents=True)
        return output_dir, snapshots_dir

    def test_should_return_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Test that function returns early when directory missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("src.attack_utils.snapshot_html_reports.logging"):
            generate_snapshot_index(str(output_dir))

        index_file = output_dir / "attack_snapshots_0" / "index.html"
        assert not index_file.exists()

    @patch("src.attack_utils.snapshot_html_reports.list_attack_snapshots")
    @patch("src.attack_utils.snapshot_html_reports.load_attack_snapshot")
    def test_should_generate_index_html(
        self,
        mock_load_snapshot: Mock,
        mock_list_snapshots: Mock,
        mock_snapshot_dir: tuple,
    ) -> None:
        """Test that index.html is generated."""
        output_dir, snapshots_dir = mock_snapshot_dir

        mock_list_snapshots.return_value = [
            snapshots_dir / "client_0" / "round_1" / "label_flipping.pkl"
        ]
        mock_load_snapshot.return_value = {
            "metadata": {
                "client_id": 0,
                "round_num": 1,
                "attack_type": "label_flipping",
                "attack_config": {"attack_type": "label_flipping"},
                "num_samples": 5,
            }
        }

        with patch(
            "src.attack_utils.snapshot_html_reports._generate_index_html"
        ) as mock_gen:
            mock_gen.return_value = "<html>Test</html>"
            generate_snapshot_index(str(output_dir))

        index_file = snapshots_dir / "index.html"
        assert index_file.exists()

    @patch("src.attack_utils.snapshot_html_reports.list_attack_snapshots")
    @patch("src.attack_utils.snapshot_html_reports.load_attack_snapshot")
    def test_should_handle_composite_attacks(
        self,
        mock_load_snapshot: Mock,
        mock_list_snapshots: Mock,
        mock_snapshot_dir: tuple,
    ) -> None:
        """Test that composite attacks are handled in index."""
        output_dir, snapshots_dir = mock_snapshot_dir

        mock_list_snapshots.return_value = [
            snapshots_dir / "client_0" / "round_1" / "composite.pkl"
        ]
        mock_load_snapshot.return_value = {
            "metadata": {
                "client_id": 0,
                "round_num": 1,
                "attack_type": "label_flipping_gaussian_noise",
                "attack_config": [
                    {"attack_type": "label_flipping"},
                    {"attack_type": "gaussian_noise", "target_noise_snr": 10},
                ],
                "num_samples": 5,
            }
        }

        with patch(
            "src.attack_utils.snapshot_html_reports._generate_index_html"
        ) as mock_gen:
            mock_gen.return_value = "<html>Test</html>"
            generate_snapshot_index(str(output_dir))

        index_file = snapshots_dir / "index.html"
        assert index_file.exists()

    @patch("src.attack_utils.snapshot_html_reports.list_attack_snapshots")
    @patch("src.attack_utils.snapshot_html_reports.load_attack_snapshot")
    def test_should_handle_token_replacement(
        self,
        mock_load_snapshot: Mock,
        mock_list_snapshots: Mock,
        mock_snapshot_dir: tuple,
    ) -> None:
        """Test that token replacement attack uses correct visual type."""
        output_dir, snapshots_dir = mock_snapshot_dir

        mock_list_snapshots.return_value = [
            snapshots_dir / "client_0" / "round_1" / "token_replacement.pkl"
        ]
        mock_load_snapshot.return_value = {
            "metadata": {
                "client_id": 0,
                "round_num": 1,
                "attack_type": "token_replacement",
                "attack_config": {"attack_type": "token_replacement"},
                "num_samples": 5,
            }
        }

        with patch(
            "src.attack_utils.snapshot_html_reports._generate_index_html"
        ) as mock_gen:
            mock_gen.return_value = "<html>Test</html>"
            generate_snapshot_index(str(output_dir))

        mock_gen.assert_called_once()
        snapshot_data = mock_gen.call_args[0][0]
        assert snapshot_data[0]["visual_type"] == "text"


class TestGenerateIndexHtml:
    """Test suite for _generate_index_html function."""

    @pytest.fixture
    def sample_snapshot_data(self) -> list:
        """Create sample snapshot data."""
        return [
            {
                "client": 0,
                "round": 1,
                "attack_types": ["label_flipping"],
                "is_stacked": False,
                "samples": 5,
                "parameters": "N/A",
                "pickle_path": "client_0/round_1/label_flipping.pkl",
                "visual_path": "client_0/round_1/label_flipping_visual.png",
                "visual_type": "image",
                "metadata_path": "client_0/round_1/label_flipping_metadata.json",
            }
        ]

    @patch("src.attack_utils.snapshot_html_reports.Environment")
    def test_should_generate_html_content(
        self, mock_env_class: Mock, sample_snapshot_data: list, tmp_path: Path
    ) -> None:
        """Test that HTML content is generated."""
        mock_env = Mock()
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test</html>"
        mock_env.get_template.return_value = mock_template
        mock_env_class.return_value = mock_env

        result = _generate_index_html(
            sample_snapshot_data, str(tmp_path), run_config=None
        )

        assert result == "<html>Test</html>"
        mock_template.render.assert_called_once()

    @patch("src.attack_utils.snapshot_html_reports.Environment")
    def test_should_include_run_config_data(
        self, mock_env_class: Mock, sample_snapshot_data: list, tmp_path: Path
    ) -> None:
        """Test that run config data is included in context."""
        mock_env = Mock()
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test</html>"
        mock_env.get_template.return_value = mock_template
        mock_env_class.return_value = mock_env

        run_config = {"num_of_clients": 10, "num_of_rounds": 50}

        _generate_index_html(sample_snapshot_data, str(tmp_path), run_config)

        call_args = mock_template.render.call_args[0][0]
        assert call_args["total_clients"] == 10
        assert call_args["total_rounds"] == 50

    @patch("src.attack_utils.snapshot_html_reports.Environment")
    def test_should_extract_unique_values(
        self, mock_env_class: Mock, tmp_path: Path
    ) -> None:
        """Test that unique clients/rounds/attack types are extracted."""
        mock_env = Mock()
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test</html>"
        mock_env.get_template.return_value = mock_template
        mock_env_class.return_value = mock_env

        snapshot_data = [
            {
                "client": 0,
                "round": 1,
                "attack_types": ["label_flipping"],
                "is_stacked": False,
                "samples": 5,
                "parameters": "N/A",
                "pickle_path": "test.pkl",
                "visual_path": "test.png",
                "visual_type": "image",
                "metadata_path": "test.json",
            },
            {
                "client": 1,
                "round": 2,
                "attack_types": ["gaussian_noise"],
                "is_stacked": False,
                "samples": 5,
                "parameters": "N/A",
                "pickle_path": "test2.pkl",
                "visual_path": "test2.png",
                "visual_type": "image",
                "metadata_path": "test2.json",
            },
        ]

        _generate_index_html(snapshot_data, str(tmp_path), run_config=None)

        call_args = mock_template.render.call_args[0][0]
        assert 0 in call_args["unique_clients"]
        assert 1 in call_args["unique_clients"]
        assert 1 in call_args["unique_rounds"]
        assert 2 in call_args["unique_rounds"]
        assert "label_flipping" in call_args["unique_attack_types"]
        assert "gaussian_noise" in call_args["unique_attack_types"]


class TestGenerateMainDashboard:
    """Test suite for generate_main_dashboard function."""

    def test_should_return_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Test that function returns early when output dir missing."""
        output_dir = tmp_path / "missing"

        with patch("src.attack_utils.snapshot_html_reports.logging") as mock_logging:
            generate_main_dashboard(str(output_dir))

        mock_logging.warning.assert_called_once()

    def test_should_generate_dashboard_html(self, tmp_path: Path) -> None:
        """Test that dashboard HTML is generated."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        snapshots_dir = output_dir / "attack_snapshots_0"
        snapshots_dir.mkdir()

        index_file = output_dir / "index.html"

        with patch(
            "src.attack_utils.snapshot_html_reports._generate_main_dashboard_html"
        ) as mock_gen:
            mock_gen.return_value = "<html>Dashboard</html>"
            generate_main_dashboard(str(output_dir))

        assert index_file.exists()
        content = index_file.read_text(encoding="utf-8")
        assert content == "<html>Dashboard</html>"

    def test_should_scan_for_snapshot_directories(self, tmp_path: Path) -> None:
        """Test that snapshot directories are scanned."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for i in range(3):
            snapshots_dir = output_dir / f"attack_snapshots_{i}"
            snapshots_dir.mkdir()

        with patch(
            "src.attack_utils.snapshot_html_reports._generate_main_dashboard_html"
        ) as mock_gen:
            mock_gen.return_value = "<html>Dashboard</html>"
            generate_main_dashboard(str(output_dir))

        call_args = mock_gen.call_args[1]
        snapshot_info = call_args["snapshot_info"]
        assert len(snapshot_info) == 3

    def test_should_categorize_plots(self, tmp_path: Path) -> None:
        """Test that plots are categorized correctly."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (output_dir / "accuracy_plot.pdf").touch()
        (output_dir / "removal_rate.pdf").touch()
        (output_dir / "distance_plot.pdf").touch()
        (output_dir / "time_metrics.pdf").touch()

        with patch(
            "src.attack_utils.snapshot_html_reports._generate_main_dashboard_html"
        ) as mock_gen:
            mock_gen.return_value = "<html>Dashboard</html>"
            generate_main_dashboard(str(output_dir))

        call_args = mock_gen.call_args[1]
        plot_categories = call_args["plot_categories"]

        assert len(plot_categories["Performance Metrics"]) > 0
        assert len(plot_categories["Attack Detection"]) > 0
        assert len(plot_categories["Client Analysis"]) > 0
        assert len(plot_categories["System Metrics"]) > 0


class TestGenerateMainDashboardHtml:
    """Test suite for _generate_main_dashboard_html function."""

    @pytest.fixture
    def sample_dashboard_data(self) -> dict:
        """Create sample dashboard data."""
        return {
            "run_id": "test_run_123",
            "timestamp": "2025-01-01 12:00:00",
            "snapshot_info": [
                {
                    "strategy_num": "0",
                    "dir_name": "attack_snapshots_0",
                    "has_index": True,
                    "num_clients": 5,
                    "num_rounds": 10,
                    "total_snapshots": 50,
                }
            ],
            "plot_categories": {
                "Performance Metrics": ["accuracy.pdf", "loss.pdf"],
                "Attack Detection": ["removal_rate.pdf"],
                "Client Analysis": ["distance.pdf"],
                "System Metrics": ["time.pdf"],
            },
            "csv_files": ["results.csv", "metrics.csv"],
            "config_files": ["strategy_config_0.json"],
        }

    def test_should_generate_html_string(self, sample_dashboard_data: dict) -> None:
        """Test that HTML string is generated."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result

    def test_should_include_run_id(self, sample_dashboard_data: dict) -> None:
        """Test that run ID is included in HTML."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert "test_run_123" in result

    def test_should_include_statistics(self, sample_dashboard_data: dict) -> None:
        """Test that statistics are included."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert "50" in result

    def test_should_include_snapshot_cards(self, sample_dashboard_data: dict) -> None:
        """Test that snapshot cards are included."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert "attack_snapshots_0" in result
        assert "50 snapshots" in result

    def test_should_include_plot_categories(self, sample_dashboard_data: dict) -> None:
        """Test that plot categories are included."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert "Performance Metrics" in result
        assert "Attack Detection" in result
        assert "accuracy.pdf" in result

    def test_should_include_csv_files(self, sample_dashboard_data: dict) -> None:
        """Test that CSV files are included."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert "results.csv" in result
        assert "metrics.csv" in result

    def test_should_include_config_files(self, sample_dashboard_data: dict) -> None:
        """Test that config files are included."""
        result = _generate_main_dashboard_html(**sample_dashboard_data)

        assert "strategy_config_0.json" in result

    def test_should_handle_empty_sections(self) -> None:
        """Test that empty sections are handled gracefully."""
        data = {
            "run_id": "test",
            "timestamp": "2025-01-01",
            "snapshot_info": [],
            "plot_categories": {
                "Performance Metrics": [],
                "Attack Detection": [],
                "Client Analysis": [],
                "System Metrics": [],
            },
            "csv_files": [],
            "config_files": [],
        }

        result = _generate_main_dashboard_html(**data)

        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    @pytest.mark.parametrize("num_strategies", [1, 2, 5])
    def test_should_handle_multiple_strategies(self, num_strategies: int) -> None:
        """Test that multiple strategies are handled."""
        data = {
            "run_id": "test",
            "timestamp": "2025-01-01",
            "snapshot_info": [
                {
                    "strategy_num": str(i),
                    "dir_name": f"attack_snapshots_{i}",
                    "has_index": True,
                    "num_clients": 5,
                    "num_rounds": 10,
                    "total_snapshots": 50,
                }
                for i in range(num_strategies)
            ],
            "plot_categories": {
                "Performance Metrics": [],
                "Attack Detection": [],
                "Client Analysis": [],
                "System Metrics": [],
            },
            "csv_files": [],
            "config_files": [],
        }

        result = _generate_main_dashboard_html(**data)

        assert isinstance(result, str)
        for i in range(num_strategies):
            assert f"attack_snapshots_{i}" in result
