"""
Unit tests for ConfigLoader class.

Tests JSON parsing, configuration merging, dataset configuration mapping,
and error handling with mocked file operations.
"""

import json
from unittest.mock import mock_open, patch

from tests.common import pytest
from src.config_loaders.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader functionality."""

    def test_init_with_valid_paths(self, tmp_path):
        """Test ConfigLoader initialization with valid file paths."""
        # Create mock usecase config
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 5,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "true",
                "num_of_clients": 10,
                "num_of_malicious_clients": 2,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "true",
                "preserve_dataset": "false",
                "training_subset_fraction": 0.8,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 10,
                "min_evaluate_clients": 10,
                "min_available_clients": 10,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "trust",
                    "begin_removing_from_round": 2,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                    "num_of_clusters": 1,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 5,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                }
            ],
        }

        dataset_config = {"its": "datasets/its", "femnist_iid": "datasets/femnist_iid"}

        # Create temporary files
        usecase_file = tmp_path / "usecase_config.json"
        dataset_file = tmp_path / "dataset_config.json"

        with open(usecase_file, "w") as f:
            json.dump(usecase_config, f)
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f)

        # Test initialization
        config_loader = ConfigLoader(str(usecase_file), str(dataset_file))

        assert config_loader.usecase_config_path == str(usecase_file)
        assert config_loader.dataset_config_path == str(dataset_file)
        assert len(config_loader.usecase_config_list) == 1
        assert config_loader.dataset_config_list == dataset_config

    def test_merge_usecase_configs_success(self, tmp_path):
        """Test successful merging of usecase configurations with shared settings."""
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 5,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "true",
                "num_of_clients": 10,
                "num_of_malicious_clients": 2,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "true",
                "preserve_dataset": "false",
                "training_subset_fraction": 0.8,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 10,
                "min_evaluate_clients": 10,
                "min_available_clients": 10,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "trust",
                    "begin_removing_from_round": 2,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                    "num_of_clusters": 1,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 5,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                },
                {
                    "aggregation_strategy_keyword": "pid",
                    "num_std_dev": 2.0,
                    "Kp": 1.0,
                    "Ki": 0.1,
                    "Kd": 0.01,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 5,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                },
            ],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(usecase_config, f)

        result = ConfigLoader._merge_usecase_configs(str(config_file))

        # Verify shared settings were merged into each strategy
        assert len(result) == 2
        for strategy in result:
            assert strategy["num_of_rounds"] == 5
            assert strategy["dataset_keyword"] == "its"
            assert strategy["num_of_clients"] == 10

        # Verify strategy-specific parameters are preserved
        trust_strategy = next(
            s for s in result if s["aggregation_strategy_keyword"] == "trust"
        )
        assert trust_strategy["trust_threshold"] == 0.7
        assert trust_strategy["beta_value"] == 0.5

        pid_strategy = next(
            s for s in result if s["aggregation_strategy_keyword"] == "pid"
        )
        assert pid_strategy["Kp"] == 1.0
        assert pid_strategy["Ki"] == 0.1

    def test_merge_usecase_configs_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON syntax."""
        config_file = tmp_path / "invalid_config.json"
        config_file.write_text("{ invalid json syntax }")

        with pytest.raises(SystemExit) as exc_info:
            ConfigLoader._merge_usecase_configs(str(config_file))

        assert exc_info.value.code == -1

    def test_merge_usecase_configs_missing_file(self):
        """Test error handling for missing configuration file."""
        with pytest.raises(SystemExit) as exc_info:
            ConfigLoader._merge_usecase_configs("nonexistent_file.json")

        assert exc_info.value.code == -1

    @patch("src.config_loaders.config_loader.validate_strategy_config")
    def test_merge_usecase_configs_validation_failure(self, mock_validate, tmp_path):
        """Test error handling when strategy validation fails."""
        # Mock validation to raise an exception
        mock_validate.side_effect = Exception("Validation failed")

        usecase_config = {
            "shared_settings": {"num_of_rounds": 5, "dataset_keyword": "its"},
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "trust",
                    # Missing required parameters for validation
                }
            ],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(usecase_config, f)

        with pytest.raises(SystemExit) as exc_info:
            ConfigLoader._merge_usecase_configs(str(config_file))

        assert exc_info.value.code == -1

    def test_set_config_success(self, tmp_path):
        """Test successful loading of dataset configuration."""
        dataset_config = {
            "its": "datasets/its",
            "femnist_iid": "datasets/femnist_iid",
            "pneumoniamnist": "datasets/pneumoniamnist",
        }

        config_file = tmp_path / "dataset_config.json"
        with open(config_file, "w") as f:
            json.dump(dataset_config, f)

        result = ConfigLoader._set_config(str(config_file))

        assert result == dataset_config
        assert "its" in result
        assert result["its"] == "datasets/its"

    def test_set_config_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON in dataset config."""
        config_file = tmp_path / "invalid_dataset.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(SystemExit) as exc_info:
            ConfigLoader._set_config(str(config_file))

        assert exc_info.value.code == -1

    def test_set_config_missing_file(self):
        """Test error handling for missing dataset configuration file."""
        with pytest.raises(SystemExit) as exc_info:
            ConfigLoader._set_config("nonexistent_dataset.json")

        assert exc_info.value.code == -1

    def test_get_usecase_config_list(self, tmp_path):
        """Test retrieval of usecase configuration list."""
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 3,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "true",
                "num_of_clients": 5,
                "num_of_malicious_clients": 1,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "true",
                "preserve_dataset": "false",
                "training_subset_fraction": 0.8,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 5,
                "min_evaluate_clients": 5,
                "min_available_clients": 5,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "krum",
                    "num_krum_selections": 3,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 3,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                }
            ],
        }

        dataset_config = {"its": "datasets/its"}

        usecase_file = tmp_path / "usecase.json"
        dataset_file = tmp_path / "dataset.json"

        with open(usecase_file, "w") as f:
            json.dump(usecase_config, f)
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f)

        config_loader = ConfigLoader(str(usecase_file), str(dataset_file))
        result = config_loader.get_usecase_config_list()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["aggregation_strategy_keyword"] == "krum"
        assert result[0]["num_of_rounds"] == 3  # From shared settings

    def test_get_dataset_config_list(self, tmp_path):
        """Test retrieval of dataset configuration list."""
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 3,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "false",
                "num_of_clients": 5,
                "num_of_malicious_clients": 0,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "false",
                "preserve_dataset": "false",
                "training_subset_fraction": 1.0,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 5,
                "min_evaluate_clients": 5,
                "min_available_clients": 5,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "rfa",
                    "attack_schedule": [],
                }
            ],
        }

        dataset_config = {
            "its": "datasets/its",
            "femnist_iid": "datasets/femnist_iid",
            "pneumoniamnist": "datasets/pneumoniamnist",
        }

        usecase_file = tmp_path / "usecase.json"
        dataset_file = tmp_path / "dataset.json"

        with open(usecase_file, "w") as f:
            json.dump(usecase_config, f)
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f)

        config_loader = ConfigLoader(str(usecase_file), str(dataset_file))
        result = config_loader.get_dataset_config_list()

        assert result == dataset_config
        assert "its" in result
        assert "femnist_iid" in result
        assert "pneumoniamnist" in result

    def test_get_dataset_folder_name_success(self, tmp_path):
        """Test successful retrieval of dataset folder name by key."""
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 1,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "false",
                "num_of_clients": 3,
                "num_of_malicious_clients": 0,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "false",
                "preserve_dataset": "false",
                "training_subset_fraction": 1.0,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 3,
                "min_evaluate_clients": 3,
                "min_available_clients": 3,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "bulyan",
                    "attack_schedule": [],
                }
            ],
        }

        dataset_config = {"its": "datasets/its", "femnist_iid": "datasets/femnist_iid"}

        usecase_file = tmp_path / "usecase.json"
        dataset_file = tmp_path / "dataset.json"

        with open(usecase_file, "w") as f:
            json.dump(usecase_config, f)
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f)

        config_loader = ConfigLoader(str(usecase_file), str(dataset_file))

        result = config_loader.get_dataset_folder_name("its")
        assert result == "datasets/its"

        result = config_loader.get_dataset_folder_name("femnist_iid")
        assert result == "datasets/femnist_iid"

    def test_get_dataset_folder_name_key_error(self, tmp_path):
        """Test error handling for invalid dataset key."""
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 1,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "false",
                "num_of_clients": 3,
                "num_of_malicious_clients": 0,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "false",
                "preserve_dataset": "false",
                "training_subset_fraction": 1.0,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 3,
                "min_evaluate_clients": 3,
                "min_available_clients": 3,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "bulyan",
                    "attack_schedule": [],
                }
            ],
        }

        dataset_config = {"its": "datasets/its"}

        usecase_file = tmp_path / "usecase.json"
        dataset_file = tmp_path / "dataset.json"

        with open(usecase_file, "w") as f:
            json.dump(usecase_config, f)
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f)

        config_loader = ConfigLoader(str(usecase_file), str(dataset_file))

        with pytest.raises(SystemExit) as exc_info:
            config_loader.get_dataset_folder_name("nonexistent_dataset")

        assert exc_info.value.code == -1

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_merge_usecase_configs_with_mocked_file_operations(
        self, mock_json_load, mock_file
    ):
        """Test configuration merging with mocked file operations."""
        # Mock the JSON data that would be loaded
        mock_config_data = {
            "shared_settings": {
                "num_of_rounds": 5,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "true",
                "num_of_clients": 10,
                "num_of_malicious_clients": 2,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "true",
                "preserve_dataset": "false",
                "training_subset_fraction": 0.8,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 10,
                "min_evaluate_clients": 10,
                "min_available_clients": 10,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "trust",
                    "begin_removing_from_round": 2,
                    "trust_threshold": 0.7,
                    "beta_value": 0.5,
                    "num_of_clusters": 1,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 5,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                }
            ],
        }

        mock_json_load.return_value = mock_config_data

        result = ConfigLoader._merge_usecase_configs("mock_config.json")

        # Verify file was opened
        mock_file.assert_called_once_with("mock_config.json")
        mock_json_load.assert_called_once()

        # Verify configuration merging
        assert len(result) == 1
        strategy = result[0]
        assert strategy["aggregation_strategy_keyword"] == "trust"
        assert strategy["num_of_rounds"] == 5  # From shared settings
        assert strategy["trust_threshold"] == 0.7  # Strategy-specific

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_set_config_with_mocked_file_operations(self, mock_json_load, mock_file):
        """Test dataset configuration loading with mocked file operations."""
        mock_dataset_config = {
            "its": "datasets/its",
            "femnist_iid": "datasets/femnist_iid",
            "pneumoniamnist": "datasets/pneumoniamnist",
        }

        mock_json_load.return_value = mock_dataset_config

        result = ConfigLoader._set_config("mock_dataset_config.json")

        # Verify file was opened
        mock_file.assert_called_once_with("mock_dataset_config.json")
        mock_json_load.assert_called_once()

        # Verify configuration loading
        assert result == mock_dataset_config
        assert "its" in result
        assert result["its"] == "datasets/its"

    def test_configuration_merging_preserves_strategy_specific_params(self, tmp_path):
        """Test that configuration merging preserves strategy-specific parameters."""
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 10,
                "dataset_keyword": "femnist_iid",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "true",
                "num_of_clients": 20,
                "num_of_malicious_clients": 4,
                "show_plots": "true",
                "save_plots": "true",
                "save_csv": "true",
                "preserve_dataset": "false",
                "training_subset_fraction": 0.9,
                "training_device": "gpu",
                "cpus_per_client": 2,
                "gpus_per_client": 0.5,
                "min_fit_clients": 20,
                "min_evaluate_clients": 20,
                "min_available_clients": 20,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 3,
                "batch_size": 64,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "trimmed_mean",
                    "trim_ratio": 0.3,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 10,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                },
                {
                    "aggregation_strategy_keyword": "multi-krum",
                    "num_krum_selections": 12,
                    "attack_schedule": [
                        {
                            "start_round": 1,
                            "end_round": 10,
                            "attack_type": "label_flipping",
                            "selection_strategy": "percentage",
                            "malicious_percentage": 0.2,
                        }
                    ],
                },
            ],
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(usecase_config, f)

        result = ConfigLoader._merge_usecase_configs(str(config_file))

        # Verify both strategies have shared settings
        assert len(result) == 2
        for strategy in result:
            assert strategy["num_of_rounds"] == 10
            assert strategy["dataset_keyword"] == "femnist_iid"
            assert strategy["num_of_clients"] == 20

        # Verify strategy-specific parameters are preserved
        trimmed_mean_strategy = next(
            s for s in result if s["aggregation_strategy_keyword"] == "trimmed_mean"
        )
        assert trimmed_mean_strategy["trim_ratio"] == 0.3

        multi_krum_strategy = next(
            s for s in result if s["aggregation_strategy_keyword"] == "multi-krum"
        )
        assert multi_krum_strategy["num_krum_selections"] == 12

    def test_dataset_configuration_mapping_functionality(self, tmp_path):
        """Test complete dataset configuration mapping functionality."""
        # Create dataset configuration
        dataset_config = {
            "its": "datasets/its",
            "flair": "datasets/flair",
            "femnist_iid": "datasets/femnist_iid",
            "femnist_niid": "datasets/femnist_niid",
            "pneumoniamnist": "datasets/pneumoniamnist",
            "bloodmnist": "datasets/bloodmnist",
            "lung_photos": "datasets/lung_photos",
        }

        # Minimal valid usecase config
        usecase_config = {
            "shared_settings": {
                "num_of_rounds": 1,
                "dataset_keyword": "its",
                "model_type": "cnn",
                "use_llm": "false",
                "remove_clients": "false",
                "num_of_clients": 3,
                "num_of_malicious_clients": 0,
                "show_plots": "false",
                "save_plots": "false",
                "save_csv": "false",
                "preserve_dataset": "false",
                "training_subset_fraction": 1.0,
                "training_device": "cpu",
                "cpus_per_client": 1,
                "gpus_per_client": 0.0,
                "min_fit_clients": 3,
                "min_evaluate_clients": 3,
                "min_available_clients": 3,
                "evaluate_metrics_aggregation_fn": "weighted_average",
                "num_of_client_epochs": 1,
                "batch_size": 32,
            },
            "simulation_strategies": [
                {
                    "aggregation_strategy_keyword": "rfa",
                    "attack_schedule": [],
                }
            ],
        }

        usecase_file = tmp_path / "usecase.json"
        dataset_file = tmp_path / "dataset.json"

        with open(usecase_file, "w") as f:
            json.dump(usecase_config, f)
        with open(dataset_file, "w") as f:
            json.dump(dataset_config, f)

        config_loader = ConfigLoader(str(usecase_file), str(dataset_file))

        # Test all dataset mappings
        for dataset_key, expected_path in dataset_config.items():
            result = config_loader.get_dataset_folder_name(dataset_key)
            assert result == expected_path

        # Test that the complete dataset config is accessible
        full_config = config_loader.get_dataset_config_list()
        assert full_config == dataset_config
        assert len(full_config) == 7  # All 7 datasets
