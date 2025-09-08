"""
Unit tests for strategy configuration validation.

Tests strategy parameter validation, error handling for invalid JSON and missing parameters,
and clear error message generation according to requirements 2.2, 2.3, 2.5.
"""

import pytest
from jsonschema import ValidationError

from src.config_loaders.validate_strategy_config import (
    validate_dependent_params, validate_strategy_config)


class TestValidateStrategyConfig:
    """Test suite for strategy configuration validation functionality."""

    def test_validate_strategy_config_valid_trust_strategy(self):
        """Test validation of valid trust strategy configuration."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        # Should not raise any exception
        validate_strategy_config(config)

    def test_validate_strategy_config_valid_pid_strategy(self):
        """Test validation of valid PID strategy configuration."""
        config = {
            "aggregation_strategy_keyword": "pid",
            "remove_clients": "true",
            "dataset_keyword": "its",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 3,
            "num_of_clients": 8,
            "num_of_malicious_clients": 1,
            "attack_type": "gaussian_noise",
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "true",
            "preserve_dataset": "true",
            "training_subset_fraction": 1.0,
            "training_device": "gpu",
            "cpus_per_client": 2,
            "gpus_per_client": 0.5,
            "min_fit_clients": 6,
            "min_evaluate_clients": 6,
            "min_available_clients": 8,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 5,
            "batch_size": 64,
            # PID-specific parameters
            "num_std_dev": 2.0,
            "Kp": 1.0,
            "Ki": 0.1,
            "Kd": 0.01,
            # Gaussian noise attack parameters
            "gaussian_noise_mean": 0.0,
            "gaussian_noise_std": 0.1,
            "attack_ratio": 0.2,
        }

        # Should not raise any exception
        validate_strategy_config(config)

    def test_validate_strategy_config_valid_krum_strategy(self):
        """Test validation of valid Krum strategy configuration."""
        config = {
            "aggregation_strategy_keyword": "krum",
            "remove_clients": "false",
            "dataset_keyword": "pneumoniamnist",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 4,
            "num_of_clients": 12,
            "num_of_malicious_clients": 0,
            "attack_type": "label_flipping",
            "show_plots": "true",
            "save_plots": "true",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.9,
            "training_device": "cuda",
            "cpus_per_client": 1,
            "gpus_per_client": 1.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 12,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 2,
            "batch_size": 16,
            # Krum-specific parameters
            "num_krum_selections": 8,
        }

        # Should not raise any exception
        validate_strategy_config(config)

    def test_validate_strategy_config_valid_trimmed_mean_strategy(self):
        """Test validation of valid trimmed mean strategy configuration."""
        config = {
            "aggregation_strategy_keyword": "trimmed_mean",
            "remove_clients": "true",
            "dataset_keyword": "bloodmnist",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 6,
            "num_of_clients": 15,
            "num_of_malicious_clients": 3,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "true",
            "preserve_dataset": "true",
            "training_subset_fraction": 0.7,
            "training_device": "cpu",
            "cpus_per_client": 4,
            "gpus_per_client": 0.0,
            "min_fit_clients": 12,
            "min_evaluate_clients": 12,
            "min_available_clients": 15,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 128,
            # Trimmed mean specific parameters
            "trim_ratio": 0.2,
        }

        # Should not raise any exception
        validate_strategy_config(config)


class TestValidateStrategyConfigMissingRequiredParams:
    """Test validation errors for missing required parameters."""

    def test_missing_aggregation_strategy_keyword(self):
        """Test validation fails when aggregation_strategy_keyword is missing."""
        config = {
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'aggregation_strategy_keyword' is a required property" in str(
            exc_info.value
        )

    def test_missing_dataset_keyword(self):
        """Test validation fails when dataset_keyword is missing."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'dataset_keyword' is a required property" in str(exc_info.value)

    def test_missing_num_of_rounds(self):
        """Test validation fails when num_of_rounds is missing."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'num_of_rounds' is a required property" in str(exc_info.value)

    def test_missing_flower_settings(self):
        """Test validation fails when Flower-specific settings are missing."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            # Missing all Flower settings
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        error_message = str(exc_info.value)
        assert "'training_device' is a required property" in error_message


class TestValidateStrategyConfigInvalidValues:
    """Test validation errors for invalid parameter values."""

    def test_invalid_aggregation_strategy_keyword(self):
        """Test validation fails for invalid aggregation strategy."""
        config = {
            "aggregation_strategy_keyword": "invalid_strategy",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'invalid_strategy' is not one of" in str(exc_info.value)

    def test_invalid_dataset_keyword(self):
        """Test validation fails for invalid dataset keyword."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "invalid_dataset",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'invalid_dataset' is not one of" in str(exc_info.value)

    def test_invalid_boolean_string_values(self):
        """Test validation fails for invalid boolean string values."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "maybe",  # Invalid boolean string
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'maybe' is not one of ['true', 'false']" in str(exc_info.value)

    def test_invalid_attack_type(self):
        """Test validation fails for invalid attack type."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "invalid_attack",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert (
            "'invalid_attack' is not one of ['label_flipping', 'gaussian_noise']"
            in str(exc_info.value)
        )

    def test_invalid_training_device(self):
        """Test validation fails for invalid training device."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "quantum",  # Invalid device
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'quantum' is not one of ['cpu', 'gpu', 'cuda']" in str(exc_info.value)

    def test_invalid_data_types(self):
        """Test validation fails for invalid data types."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": "five",  # Should be integer
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'five' is not of type 'integer'" in str(exc_info.value)


class TestValidateDependentParams:
    """Test validation of strategy-specific dependent parameters."""

    def test_trust_strategy_missing_trust_threshold(self):
        """Test validation fails when trust strategy is missing trust_threshold."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "begin_removing_from_round": 2,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Missing trust_threshold
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter trust_threshold for trust aggregation trust" in str(
            exc_info.value
        )

    def test_trust_strategy_missing_beta_value(self):
        """Test validation fails when trust strategy is missing beta_value."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "num_of_clusters": 1,
            # Missing beta_value
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter beta_value for trust aggregation trust" in str(
            exc_info.value
        )

    def test_trust_strategy_missing_begin_removing_from_round(self):
        """Test validation fails when trust strategy is missing begin_removing_from_round."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Missing begin_removing_from_round
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing parameter begin_removing_from_round for trust aggregation trust"
            in str(exc_info.value)
        )

    def test_trust_strategy_missing_num_of_clusters(self):
        """Test validation fails when trust strategy is missing num_of_clusters."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            # Missing num_of_clusters
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter num_of_clusters for trust aggregation trust" in str(
            exc_info.value
        )

    def test_pid_strategy_missing_kp(self):
        """Test validation fails when PID strategy is missing Kp parameter."""
        config = {
            "aggregation_strategy_keyword": "pid",
            "num_std_dev": 2.0,
            "Ki": 0.1,
            "Kd": 0.01,
            # Missing Kp
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter Kp for PID aggregation pid" in str(exc_info.value)

    def test_pid_scaled_strategy_missing_ki(self):
        """Test validation fails when PID scaled strategy is missing Ki parameter."""
        config = {
            "aggregation_strategy_keyword": "pid_scaled",
            "num_std_dev": 2.0,
            "Kp": 1.0,
            "Kd": 0.01,
            # Missing Ki
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter Ki for PID aggregation pid_scaled" in str(
            exc_info.value
        )

    def test_pid_standardized_strategy_missing_kd(self):
        """Test validation fails when PID standardized strategy is missing Kd parameter."""
        config = {
            "aggregation_strategy_keyword": "pid_standardized",
            "num_std_dev": 2.0,
            "Kp": 1.0,
            "Ki": 0.1,
            # Missing Kd
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter Kd for PID aggregation pid_standardized" in str(
            exc_info.value
        )

    def test_pid_strategy_missing_num_std_dev(self):
        """Test validation fails when PID strategy is missing num_std_dev parameter."""
        config = {
            "aggregation_strategy_keyword": "pid",
            "Kp": 1.0,
            "Ki": 0.1,
            "Kd": 0.01,
            # Missing num_std_dev
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert "Missing parameter num_std_dev for PID aggregation pid" in str(
            exc_info.value
        )

    def test_krum_strategy_missing_num_krum_selections(self):
        """Test validation fails when Krum strategy is missing num_krum_selections."""
        config = {
            "aggregation_strategy_keyword": "krum"
            # Missing num_krum_selections
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing parameter num_krum_selections for Krum-based aggregation krum"
            in str(exc_info.value)
        )

    def test_multi_krum_strategy_missing_num_krum_selections(self):
        """Test validation fails when Multi-Krum strategy is missing num_krum_selections."""
        config = {
            "aggregation_strategy_keyword": "multi-krum"
            # Missing num_krum_selections
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing parameter num_krum_selections for Krum-based aggregation multi-krum"
            in str(exc_info.value)
        )

    def test_multi_krum_based_strategy_missing_num_krum_selections(self):
        """Test validation fails when Multi-Krum-based strategy is missing num_krum_selections."""
        config = {
            "aggregation_strategy_keyword": "multi-krum-based"
            # Missing num_krum_selections
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing parameter num_krum_selections for Krum-based aggregation multi-krum-based"
            in str(exc_info.value)
        )

    def test_trimmed_mean_strategy_missing_trim_ratio(self):
        """Test validation fails when trimmed mean strategy is missing trim_ratio."""
        config = {
            "aggregation_strategy_keyword": "trimmed_mean"
            # Missing trim_ratio
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing parameter trim_ratio for trimmed mean aggregation trimmed_mean"
            in str(exc_info.value)
        )

    def test_gaussian_noise_attack_missing_parameters(self):
        """Test validation fails when gaussian noise attack is missing required parameters."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "attack_type": "gaussian_noise",
            # Trust-specific parameters (required first)
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Missing gaussian_noise_mean, gaussian_noise_std, attack_ratio
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        error_message = str(exc_info.value)
        assert (
            "Missing gaussian_noise_mean that is required for gaussian_noise in configuration"
            in error_message
        )

    def test_gaussian_noise_attack_missing_std(self):
        """Test validation fails when gaussian noise attack is missing std parameter."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "attack_type": "gaussian_noise",
            # Trust-specific parameters (required first)
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Gaussian noise parameters
            "gaussian_noise_mean": 0.0,
            "attack_ratio": 0.2,
            # Missing gaussian_noise_std
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing gaussian_noise_std that is required for gaussian_noise in configuration"
            in str(exc_info.value)
        )

    def test_gaussian_noise_attack_missing_attack_ratio(self):
        """Test validation fails when gaussian noise attack is missing attack_ratio parameter."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "attack_type": "gaussian_noise",
            # Trust-specific parameters (required first)
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Gaussian noise parameters
            "gaussian_noise_mean": 0.0,
            "gaussian_noise_std": 0.1,
            # Missing attack_ratio
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert (
            "Missing attack_ratio that is required for gaussian_noise in configuration"
            in str(exc_info.value)
        )


class TestValidateStrategyConfigErrorMessages:
    """Test that validation provides clear and helpful error messages."""

    def test_clear_error_message_for_invalid_enum_value(self):
        """Test that error messages clearly indicate valid enum options."""
        config = {
            "aggregation_strategy_keyword": "invalid_strategy",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        error_message = str(exc_info.value)
        # Should contain the invalid value and list of valid options
        assert "'invalid_strategy' is not one of" in error_message
        assert "trust" in error_message
        assert "pid" in error_message
        assert "krum" in error_message

    def test_clear_error_message_for_missing_required_field(self):
        """Test that error messages clearly indicate which required field is missing."""
        config = {
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            # Missing aggregation_strategy_keyword and other required fields
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        error_message = str(exc_info.value)
        # Should clearly indicate the missing required property
        assert "is a required property" in error_message

    def test_clear_error_message_for_wrong_data_type(self):
        """Test that error messages clearly indicate expected data type."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": "not_a_number",  # Should be integer
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        error_message = str(exc_info.value)
        # Should indicate the expected type
        assert "is not of type 'integer'" in error_message

    def test_clear_error_message_for_strategy_specific_missing_params(self):
        """Test that error messages clearly indicate which strategy-specific parameter is missing."""
        config = {
            "aggregation_strategy_keyword": "trust"
            # Missing all trust-specific parameters
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        error_message = str(exc_info.value)
        # Should clearly indicate the missing parameter and strategy
        assert "Missing parameter" in error_message
        assert "for trust aggregation trust" in error_message

    def test_clear_error_message_for_attack_specific_missing_params(self):
        """Test that error messages clearly indicate which attack-specific parameter is missing."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "attack_type": "gaussian_noise",
            # Trust-specific parameters (required first)
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Missing gaussian noise specific parameters
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        error_message = str(exc_info.value)
        # Should clearly indicate the missing parameter and attack type
        assert "that is required for gaussian_noise in configuration" in error_message


class TestValidateStrategyConfigEdgeCases:
    """Test edge cases and boundary conditions for configuration validation."""

    def test_rfa_strategy_no_additional_params_required(self):
        """Test that RFA strategy doesn't require additional parameters."""
        config = {
            "aggregation_strategy_keyword": "rfa",
            "remove_clients": "true",
            "dataset_keyword": "flair",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 4,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "true",
            "preserve_dataset": "true",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
        }

        # Should not raise any exception
        validate_strategy_config(config)

    def test_bulyan_strategy_no_additional_params_required(self):
        """Test that Bulyan strategy doesn't require additional parameters."""
        config = {
            "aggregation_strategy_keyword": "bulyan",
            "remove_clients": "false",
            "dataset_keyword": "lung_photos",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 3,
            "num_of_clients": 15,
            "num_of_malicious_clients": 0,
            "attack_type": "label_flipping",
            "show_plots": "true",
            "save_plots": "true",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "gpu",
            "cpus_per_client": 2,
            "gpus_per_client": 1.0,
            "min_fit_clients": 12,
            "min_evaluate_clients": 12,
            "min_available_clients": 15,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 2,
            "batch_size": 64,
        }

        # Should not raise any exception
        validate_strategy_config(config)

    def test_label_flipping_attack_no_additional_params_required(self):
        """Test that label flipping attack doesn't require additional parameters."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "attack_type": "label_flipping",
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        # Should not raise any exception for dependent params
        validate_dependent_params(config)

    def test_empty_config_validation(self):
        """Test validation of completely empty configuration."""
        config = {}

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        # Should indicate multiple missing required properties
        error_message = str(exc_info.value)
        assert "is a required property" in error_message

    def test_config_with_extra_unknown_fields(self):
        """Test that configuration with extra unknown fields still validates if required fields are present."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_type": "label_flipping",
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            # Extra unknown field
            "unknown_field": "some_value",
        }

        # Should not raise any exception (JSON schema allows additional properties by default)
        validate_strategy_config(config)
