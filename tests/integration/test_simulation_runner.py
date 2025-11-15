"""
Integration tests for SimulationRunner class.

Tests multi-strategy execution workflows, configuration loading, strategy processing,
and output generation with mocked dependencies.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from src.data_models.simulation_strategy_config import StrategyConfig
from src.simulation_runner import SimulationRunner
from tests.common import Mock, pytest


def _create_mock_strategy_config() -> Dict[str, Any]:
    """Return mock strategy configuration for testing."""
    return {
        "shared_settings": {
            "aggregation_strategy_keyword": "trust",
            "dataset_keyword": "its",
            "num_of_rounds": 3,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "begin_removing_from_round": 1,
            "remove_clients": True,
            "min_fit_clients": 3,
            "min_evaluate_clients": 3,
            "min_available_clients": 5,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "batch_size": 32,
            "num_of_client_epochs": 1,
            "training_subset_fraction": 1.0,
            "model_type": "cnn",
            "use_llm": False,
            "show_plots": False,
            "save_plots": True,
            "save_csv": True,
        },
        "simulation_strategies": [
            {
                "attack_schedule": [
                    {
                        "start_round": 1,
                        "end_round": 3,
                        "attack_type": "gaussian_noise",
                        "target_noise_snr": 10.0,
                        "attack_ratio": 1.0,
                        "selection_strategy": "percentage",
                        "malicious_percentage": 0.2,
                    }
                ]
            },
            {
                "attack_schedule": [
                    {
                        "start_round": 1,
                        "end_round": 3,
                        "attack_type": "label_flipping",
                        "flip_fraction": 1.0,
                        "selection_strategy": "percentage",
                        "malicious_percentage": 0.2,
                    }
                ]
            },
        ],
    }


def __create_multi_strategy_config() -> Dict[str, Any]:
    """Return configuration with multiple strategies."""
    return {
        "shared_settings": {
            "dataset_keyword": "its",
            "num_of_rounds": 2,
            "num_of_clients": 4,
            "num_of_malicious_clients": 1,
            "begin_removing_from_round": 1,
            "remove_clients": True,
            "min_fit_clients": 3,
            "min_evaluate_clients": 3,
            "min_available_clients": 4,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "batch_size": 16,
            "num_of_client_epochs": 1,
            "training_subset_fraction": 1.0,
            "model_type": "cnn",
            "use_llm": False,
            "show_plots": False,
            "save_plots": True,
            "save_csv": True,
        },
        "simulation_strategies": [
            {
                "aggregation_strategy_keyword": "trust",
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            },
            {
                "aggregation_strategy_keyword": "pid",
                "Kp": 1.0,
                "Ki": 0.1,
                "Kd": 0.01,
                "num_std_dev": 2.0,
            },
            {
                "aggregation_strategy_keyword": "krum",
                "num_krum_selections": 3,
            },
        ],
    }


class TestSimulationRunnerInitialization:
    """Test SimulationRunner initialization."""

    @pytest.fixture
    def temp_config_files(self, tmp_path: Path) -> Dict[str, Path]:
        """Create temporary configuration files."""
        # Create strategy config
        strategy_config = _create_mock_strategy_config()
        strategy_file = tmp_path / "test_strategy.json"
        with open(strategy_file, "w") as f:
            json.dump(strategy_config, f, indent=2)

        # Create dataset config
        dataset_config = {
            "its": "datasets/its",
            "femnist_iid": "datasets/femnist_iid",
            "pneumoniamnist": "datasets/pneumoniamnist",
        }
        dataset_file = tmp_path / "dataset_config.json"
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f, indent=2)

        return {"strategy": strategy_file, "dataset": dataset_file}

    def test_simulation_runner_initialization_with_valid_config(
        self, temp_config_files: Dict[str, Path]
    ) -> None:
        """Test initialization with valid configuration files."""
        # Arrange & Act
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 3,
                    "num_of_clients": 5,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                }
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance
            mock_directory_handler.return_value = Mock()

            runner = SimulationRunner("test_strategy.json")

        # Assert
        assert runner._config_loader is not None
        assert runner._simulation_strategy_config_dicts is not None
        assert runner._dataset_config_list is not None
        assert runner._directory_handler is not None
        assert len(runner._simulation_strategy_config_dicts) == 1

    def test_simulation_runner_initialization_with_multi_strategy_config(
        self, temp_config_files: Dict[str, Path]
    ) -> None:
        """Test initialization with multiple strategies."""
        # Arrange

        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 4,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                },
                {
                    "aggregation_strategy_keyword": "pid",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 4,
                    "Kp": 1.0,
                    "Ki": 0.1,
                    "Kd": 0.01,
                },
                {
                    "aggregation_strategy_keyword": "krum",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 4,
                    "num_krum_selections": 3,
                },
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance
            mock_directory_handler.return_value = Mock()

            # Act
            runner = SimulationRunner("multi_strategy.json")

        # Assert
        assert len(runner._simulation_strategy_config_dicts) == 3
        strategies = [
            config["aggregation_strategy_keyword"]
            for config in runner._simulation_strategy_config_dicts
        ]
        assert "trust" in strategies
        assert "pid" in strategies
        assert "krum" in strategies

    def test_simulation_runner_logging_configuration(self) -> None:
        """Test logging configuration."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.logging.basicConfig") as mock_logging_config,
            patch("src.simulation_runner.logging.getLogger") as mock_get_logger,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
        ):
            mock_logger = Mock()
            mock_logger.hasHandlers.return_value = False
            mock_get_logger.return_value = mock_logger

            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = []
            mock_loader_instance.get_dataset_config_list.return_value = []
            mock_config_loader.return_value = mock_loader_instance
            mock_directory_handler.return_value = Mock()

            # Act
            SimulationRunner("test_config.json")

        # Assert
        mock_logging_config.assert_called_once_with(
            level=logging.INFO, format="%(levelname)s: %(message)s"
        )


class TestSimulationRunnerExecution:
    """Test SimulationRunner execution."""

    @pytest.fixture
    def mock_runner_components(self) -> Any:
        """Create mocked components for testing."""
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
            patch("src.simulation_runner.DatasetHandler") as mock_dataset_handler,
            patch(
                "src.simulation_runner.FederatedSimulation"
            ) as mock_federated_simulation,
            patch("src.simulation_runner.new_plot_handler") as mock_plot_handler,
        ):
            # Configure ConfigLoader mock
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 3,
                    "num_of_clients": 5,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                }
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            # Configure DirectoryHandler mock
            mock_dir_instance = Mock()
            mock_dir_instance.dataset_dir = "/tmp/test_dataset"
            mock_directory_handler.return_value = mock_dir_instance

            # Configure DatasetHandler mock
            mock_dataset_instance = Mock()
            mock_dataset_handler.return_value = mock_dataset_instance

            # Configure FederatedSimulation mock
            mock_simulation_instance = Mock()
            mock_simulation_instance.strategy_history = Mock()
            mock_simulation_instance.strategy_history.calculate_additional_rounds_data = Mock()
            mock_federated_simulation.return_value = mock_simulation_instance

            yield {
                "config_loader": mock_config_loader,
                "directory_handler": mock_directory_handler,
                "dataset_handler": mock_dataset_handler,
                "federated_simulation": mock_federated_simulation,
                "plot_handler": mock_plot_handler,
                "loader_instance": mock_loader_instance,
                "dir_instance": mock_dir_instance,
                "dataset_instance": mock_dataset_instance,
                "simulation_instance": mock_simulation_instance,
            }

    def test_single_strategy_execution_workflow(self, mock_runner_components):
        """Test single strategy execution workflow."""
        # Arrange
        mocks = mock_runner_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert - Verify component initialization and method calls
        mocks["config_loader"].assert_called_once()
        mocks["directory_handler"].assert_called_once()

        # Verify dataset handler setup and teardown
        mocks["dataset_handler"].assert_called_once()
        mocks["dataset_instance"].setup_dataset.assert_called_once()
        mocks["dataset_instance"].teardown_dataset.assert_called_once()

        # Verify simulation execution
        mocks["federated_simulation"].assert_called_once()
        mocks["simulation_instance"].run_simulation.assert_called_once()

        # Verify directory operations
        mocks["dir_instance"].assign_dataset_dir.assert_called_once_with(0)
        mocks["dir_instance"].save_csv_and_config.assert_called_once()

        # Verify plot generation
        mocks["plot_handler"].show_plots_within_strategy.assert_called_once()
        mocks["plot_handler"].show_inter_strategy_plots.assert_called_once()

        # Verify strategy history calculations
        mocks[
            "simulation_instance"
        ].strategy_history.calculate_additional_rounds_data.assert_called_once()

    def test_multi_strategy_execution_workflow(self, mock_runner_components):
        """Test multiple strategy execution workflow."""
        # Arrange
        mocks = mock_runner_components
        mocks["loader_instance"].get_usecase_config_list.return_value = [
            {
                "aggregation_strategy_keyword": "trust",
                "dataset_keyword": "its",
                "num_of_rounds": 2,
                "num_of_clients": 4,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            },
            {
                "aggregation_strategy_keyword": "pid",
                "dataset_keyword": "its",
                "num_of_rounds": 2,
                "num_of_clients": 4,
                "Kp": 1.0,
                "Ki": 0.1,
                "Kd": 0.01,
            },
        ]

        runner = SimulationRunner("multi_config.json")

        # Act
        runner.run()

        # Assert - Verify multiple strategy execution
        assert mocks["dataset_handler"].call_count == 2
        assert mocks["federated_simulation"].call_count == 2
        assert mocks["dataset_instance"].setup_dataset.call_count == 2
        assert mocks["dataset_instance"].teardown_dataset.call_count == 2
        assert mocks["simulation_instance"].run_simulation.call_count == 2

        # Verify directory assignment for each strategy
        expected_calls = [((0,),), ((1,),)]
        actual_calls = mocks["dir_instance"].assign_dataset_dir.call_args_list
        assert len(actual_calls) == 2
        assert actual_calls[0] == expected_calls[0]
        assert actual_calls[1] == expected_calls[1]

        # Verify inter-strategy plots are generated once at the end
        mocks["plot_handler"].show_inter_strategy_plots.assert_called_once()

    def test_strategy_config_creation_and_assignment(self, mock_runner_components):
        """Test that StrategyConfig objects are properly created and strategy numbers assigned."""
        # Arrange
        mocks = mock_runner_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert - Verify StrategyConfig creation
        call_args = mocks["federated_simulation"].call_args
        strategy_config = call_args.kwargs["strategy_config"]

        assert isinstance(strategy_config, StrategyConfig)
        assert hasattr(strategy_config, "strategy_number")
        assert strategy_config.strategy_number == 0
        assert strategy_config.aggregation_strategy_keyword == "trust"

    def test_dataset_handler_initialization_with_correct_parameters(
        self, mock_runner_components
    ):
        """Test that DatasetHandler is initialized with correct parameters."""
        # Arrange
        mocks = mock_runner_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert - Verify DatasetHandler initialization
        call_args = mocks["dataset_handler"].call_args
        assert "strategy_config" in call_args.kwargs
        assert "directory_handler" in call_args.kwargs
        assert "dataset_config_list" in call_args.kwargs

        strategy_config = call_args.kwargs["strategy_config"]
        assert isinstance(strategy_config, StrategyConfig)
        assert call_args.kwargs["directory_handler"] == mocks["dir_instance"]

    def test_federated_simulation_initialization_with_correct_parameters(
        self, mock_runner_components
    ):
        """Test that FederatedSimulation is initialized with correct parameters."""
        # Arrange
        mocks = mock_runner_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert - Verify FederatedSimulation initialization
        call_args = mocks["federated_simulation"].call_args
        assert "strategy_config" in call_args.kwargs
        assert "dataset_dir" in call_args.kwargs
        assert "dataset_handler" in call_args.kwargs

        assert call_args.kwargs["dataset_dir"] == "/tmp/test_dataset"
        assert call_args.kwargs["dataset_handler"] == mocks["dataset_instance"]

    def test_execution_order_and_cleanup(self, mock_runner_components):
        """Test that execution follows correct order and cleanup is performed."""
        # Arrange
        mocks = mock_runner_components
        runner = SimulationRunner("test_config.json")

        # Create a call tracker to verify order
        call_order = []

        def track_setup_call():
            call_order.append("setup_dataset")

        def track_simulation_call():
            call_order.append("run_simulation")

        def track_teardown_call():
            call_order.append("teardown_dataset")

        mocks["dataset_instance"].setup_dataset.side_effect = track_setup_call
        mocks["simulation_instance"].run_simulation.side_effect = track_simulation_call
        mocks["dataset_instance"].teardown_dataset.side_effect = track_teardown_call

        # Act
        runner.run()

        # Assert - Verify correct execution order
        expected_order = ["setup_dataset", "run_simulation", "teardown_dataset"]
        assert call_order == expected_order


class TestSimulationRunnerConfigurationProcessing:
    """Test configuration loading and strategy processing."""

    def test_configuration_loading_with_shared_settings(self):
        """Test that shared settings are properly applied to all strategies."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 3,
                    "num_of_clients": 5,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                    "shared_setting": "shared_value",
                },
                {
                    "aggregation_strategy_keyword": "pid",
                    "dataset_keyword": "its",
                    "num_of_rounds": 3,
                    "num_of_clients": 5,
                    "Kp": 1.0,
                    "Ki": 0.1,
                    "Kd": 0.01,
                    "shared_setting": "shared_value",
                },
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance
            mock_directory_handler.return_value = Mock()

            # Act
            runner = SimulationRunner("test_config.json")

        # Assert
        configs = runner._simulation_strategy_config_dicts
        assert len(configs) == 2

        # Verify shared settings are applied to all strategies
        for config in configs:
            assert config["shared_setting"] == "shared_value"
            assert config["dataset_keyword"] == "its"
            assert config["num_of_rounds"] == 3
            assert config["num_of_clients"] == 5

        # Verify strategy-specific settings are preserved
        trust_config = next(
            c for c in configs if c["aggregation_strategy_keyword"] == "trust"
        )
        pid_config = next(
            c for c in configs if c["aggregation_strategy_keyword"] == "pid"
        )

        assert trust_config["trust_threshold"] == 0.7
        assert trust_config["beta_value"] == 0.5
        assert pid_config["Kp"] == 1.0
        assert pid_config["Ki"] == 0.1

    def test_dataset_configuration_processing(self):
        """Test that dataset configuration is properly loaded and processed."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {"aggregation_strategy_keyword": "trust", "dataset_keyword": "its"}
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {
                    "its": "datasets/its",
                    "femnist_iid": "datasets/femnist_iid",
                    "pneumoniamnist": "datasets/pneumoniamnist",
                }
            ]
            mock_config_loader.return_value = mock_loader_instance
            mock_directory_handler.return_value = Mock()

            # Act
            runner = SimulationRunner("test_config.json")

        # Assert
        dataset_config = runner._dataset_config_list
        assert len(dataset_config) == 1
        assert "its" in dataset_config[0]
        assert "femnist_iid" in dataset_config[0]
        assert "pneumoniamnist" in dataset_config[0]

    @pytest.mark.parametrize(
        "strategy_keyword,expected_params",
        [
            ("trust", ["trust_threshold", "beta_value"]),
            ("pid", ["Kp", "Ki", "Kd"]),
            ("krum", ["num_krum_selections"]),
            ("multi-krum", ["num_krum_selections"]),
            ("trimmed_mean", []),
            ("rfa", []),
            ("bulyan", []),
        ],
    )
    def test_strategy_specific_parameter_processing(
        self, strategy_keyword: str, expected_params: List[str]
    ):
        """Test that strategy-specific parameters are properly processed."""
        # Arrange
        base_config = {
            "aggregation_strategy_keyword": strategy_keyword,
            "dataset_keyword": "its",
            "num_of_rounds": 3,
            "num_of_clients": 5,
        }

        # Add strategy-specific parameters
        if strategy_keyword == "trust":
            base_config.update({"trust_threshold": 0.7, "beta_value": 0.5})
        elif strategy_keyword == "pid":
            base_config.update({"Kp": 1.0, "Ki": 0.1, "Kd": 0.01})
        elif strategy_keyword in ["krum", "multi-krum"]:
            base_config.update({"num_krum_selections": 3})

        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [base_config]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance
            mock_directory_handler.return_value = Mock()

            # Act
            runner = SimulationRunner("test_config.json")

        # Assert
        config = runner._simulation_strategy_config_dicts[0]
        assert config["aggregation_strategy_keyword"] == strategy_keyword

        for param in expected_params:
            assert param in config, f"Parameter {param} missing for {strategy_keyword}"


class TestSimulationRunnerOutputGeneration:
    """Test output generation and cleanup operations."""

    @pytest.fixture
    def mock_output_components(self):
        """Create mocked components for output testing."""
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
            patch("src.simulation_runner.DatasetHandler") as mock_dataset_handler,
            patch(
                "src.simulation_runner.FederatedSimulation"
            ) as mock_federated_simulation,
            patch("src.simulation_runner.new_plot_handler") as mock_plot_handler,
        ):
            # Configure mocks
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 3,
                }
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            mock_dir_instance = Mock()
            mock_directory_handler.return_value = mock_dir_instance

            mock_dataset_instance = Mock()
            mock_dataset_handler.return_value = mock_dataset_instance

            mock_simulation_instance = Mock()
            mock_simulation_instance.strategy_history = Mock()
            mock_federated_simulation.return_value = mock_simulation_instance

            yield {
                "plot_handler": mock_plot_handler,
                "dir_instance": mock_dir_instance,
                "simulation_instance": mock_simulation_instance,
            }

    def test_per_strategy_plot_generation(self, mock_output_components):
        """Test that per-strategy plots are generated correctly."""
        # Arrange
        mocks = mock_output_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert
        mocks["plot_handler"].show_plots_within_strategy.assert_called_once_with(
            mocks["simulation_instance"], mocks["dir_instance"]
        )

    def test_inter_strategy_plot_generation(self, mock_output_components):
        """Test that inter-strategy comparison plots are generated."""
        # Arrange
        mocks = mock_output_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert
        mocks["plot_handler"].show_inter_strategy_plots.assert_called_once()
        call_args = mocks["plot_handler"].show_inter_strategy_plots.call_args
        executed_strategies = call_args[0][0]
        directory_handler = call_args[0][1]

        assert len(executed_strategies) == 1
        assert executed_strategies[0] == mocks["simulation_instance"]
        assert directory_handler == mocks["dir_instance"]

    def test_csv_and_config_output_generation(self, mock_output_components):
        """Test that CSV files and configuration are saved correctly."""
        # Arrange
        mocks = mock_output_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert
        mocks["dir_instance"].save_csv_and_config.assert_called_once_with(
            mocks["simulation_instance"].strategy_history
        )

    def test_strategy_history_calculations(self, mock_output_components):
        """Test that additional strategy history calculations are performed."""
        # Arrange
        mocks = mock_output_components
        runner = SimulationRunner("test_config.json")

        # Act
        runner.run()

        # Assert
        mocks[
            "simulation_instance"
        ].strategy_history.calculate_additional_rounds_data.assert_called_once()

    def test_multi_strategy_output_aggregation(self, mock_output_components):
        """Test output generation for multiple strategies."""
        # Arrange
        mocks = mock_output_components

        # Configure multiple strategies
        with patch("src.simulation_runner.ConfigLoader") as mock_config_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 3,
                },
                {
                    "aggregation_strategy_keyword": "pid",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 3,
                },
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            runner = SimulationRunner("multi_config.json")

        # Act
        runner.run()

        # Assert - Verify per-strategy outputs
        assert mocks["plot_handler"].show_plots_within_strategy.call_count == 2
        assert mocks["dir_instance"].save_csv_and_config.call_count == 2

        # Verify inter-strategy comparison is generated once
        mocks["plot_handler"].show_inter_strategy_plots.assert_called_once()
        call_args = mocks["plot_handler"].show_inter_strategy_plots.call_args
        executed_strategies = call_args[0][0]
        assert len(executed_strategies) == 2


class TestSimulationRunnerErrorHandling:
    """Test error handling in SimulationRunner."""

    def test_configuration_loading_error_handling(self):
        """Test handling of configuration loading errors."""
        # Arrange
        with patch("src.simulation_runner.ConfigLoader") as mock_config_loader:
            mock_config_loader.side_effect = FileNotFoundError("Config file not found")

            # Act & Assert
            with pytest.raises(FileNotFoundError, match="Config file not found"):
                SimulationRunner("nonexistent_config.json")

    def test_simulation_execution_error_handling(self):
        """Test handling of simulation execution errors."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
            patch("src.simulation_runner.DatasetHandler") as mock_dataset_handler,
            patch(
                "src.simulation_runner.FederatedSimulation"
            ) as mock_federated_simulation,
        ):
            # Configure mocks
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {"aggregation_strategy_keyword": "trust", "dataset_keyword": "its"}
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            mock_directory_handler.return_value = Mock()
            mock_dataset_handler.return_value = Mock()

            # Make simulation raise an error
            mock_simulation_instance = Mock()
            mock_simulation_instance.run_simulation.side_effect = RuntimeError(
                "Simulation failed"
            )
            mock_federated_simulation.return_value = mock_simulation_instance

            runner = SimulationRunner("test_config.json")

            # Act & Assert
            with pytest.raises(RuntimeError, match="Simulation failed"):
                runner.run()

    def test_dataset_setup_error_handling(self):
        """Test handling of dataset setup errors."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
            patch("src.simulation_runner.DatasetHandler") as mock_dataset_handler,
        ):
            # Configure mocks
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {"aggregation_strategy_keyword": "trust", "dataset_keyword": "its"}
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            mock_directory_handler.return_value = Mock()

            # Make dataset setup raise an error
            mock_dataset_instance = Mock()
            mock_dataset_instance.setup_dataset.side_effect = IOError(
                "Dataset setup failed"
            )
            mock_dataset_handler.return_value = mock_dataset_instance

            runner = SimulationRunner("test_config.json")

            # Act & Assert
            with pytest.raises(IOError, match="Dataset setup failed"):
                runner.run()

    def test_cleanup_on_error(self):
        """Test that cleanup is NOT performed when errors occur during simulation (current behavior)."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler") as mock_directory_handler,
            patch("src.simulation_runner.DatasetHandler") as mock_dataset_handler,
            patch(
                "src.simulation_runner.FederatedSimulation"
            ) as mock_federated_simulation,
        ):
            # Configure mocks
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {"aggregation_strategy_keyword": "trust", "dataset_keyword": "its"}
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            mock_directory_handler.return_value = Mock()

            mock_dataset_instance = Mock()
            mock_dataset_handler.return_value = mock_dataset_instance

            # Make simulation raise an error after dataset setup
            mock_simulation_instance = Mock()
            mock_simulation_instance.run_simulation.side_effect = RuntimeError(
                "Simulation failed"
            )
            mock_federated_simulation.return_value = mock_simulation_instance

            runner = SimulationRunner("test_config.json")

            # Act
            with pytest.raises(RuntimeError):
                runner.run()

            # Assert - Verify both setup and teardown were called
            mock_dataset_instance.setup_dataset.assert_called_once()
            mock_dataset_instance.teardown_dataset.assert_called_once()


class TestSimulationRunnerLogging:
    """Test logging functionality in SimulationRunner."""

    def test_strategy_execution_logging(self, caplog):
        """Test that strategy execution is properly logged."""
        # Arrange
        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler"),
            patch("src.simulation_runner.DatasetHandler"),
            patch("src.simulation_runner.FederatedSimulation"),
            patch("src.simulation_runner.new_plot_handler"),
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [
                {
                    "aggregation_strategy_keyword": "trust",
                    "dataset_keyword": "its",
                    "num_of_rounds": 2,
                    "num_of_clients": 3,
                    "trust_threshold": 0.7,
                }
            ]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"its": "datasets/its"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            runner = SimulationRunner("test_config.json")

            # Act
            with caplog.at_level(logging.INFO):
                runner.run()

        # Assert
        assert "Executing new strategy" in caplog.text
        assert "Strategy config:" in caplog.text
        assert "trust" in caplog.text

    def test_configuration_logging_format(self, caplog):
        """Test that configuration is logged in proper JSON format."""
        # Arrange
        test_config = {
            "aggregation_strategy_keyword": "pid",
            "dataset_keyword": "femnist_iid",
            "num_of_rounds": 3,
            "Kp": 1.0,
            "Ki": 0.1,
        }

        with (
            patch("src.simulation_runner.ConfigLoader") as mock_config_loader,
            patch("src.simulation_runner.DirectoryHandler"),
            patch("src.simulation_runner.DatasetHandler"),
            patch("src.simulation_runner.FederatedSimulation"),
            patch("src.simulation_runner.new_plot_handler"),
        ):
            mock_loader_instance = Mock()
            mock_loader_instance.get_usecase_config_list.return_value = [test_config]
            mock_loader_instance.get_dataset_config_list.return_value = [
                {"femnist_iid": "datasets/femnist_iid"}
            ]
            mock_config_loader.return_value = mock_loader_instance

            runner = SimulationRunner("test_config.json")

            # Act
            with caplog.at_level(logging.INFO):
                runner.run()

        # Assert - Verify JSON formatting in logs
        log_text = caplog.text
        assert '"aggregation_strategy_keyword": "pid"' in log_text
        assert '"Kp": 1.0' in log_text
        assert '"Ki": 0.1' in log_text
