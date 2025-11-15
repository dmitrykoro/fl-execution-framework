"""
Unit tests for attack schedule validation.

Tests attack schedule parameter validation, round range validation,
attack type parameters, and client selection strategies.
"""

from tests.common import pytest
from jsonschema import ValidationError
from src.config_loaders.validate_strategy_config import validate_strategy_config


class TestValidateAttackSchedule:
    """Test attack_schedule validation functionality."""

    def test_invalid_round_range_start_greater_than_end(self):
        """Test validation fails when start_round > end_round."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 8,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0],
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "start_round (8) cannot be greater than end_round (5)" in str(
            exc_info.value
        )

    def test_invalid_round_range_end_exceeds_num_of_rounds(self):
        """Test validation fails when end_round exceeds num_of_rounds."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 5,
                    "end_round": 15,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0],
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "end_round (15) exceeds num_of_rounds (10)" in str(exc_info.value)

    def test_gaussian_noise_missing_target_noise_snr(self):
        """Test validation fails when gaussian_noise is missing target_noise_snr."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "gaussian_noise",
                    "attack_ratio": 0.3,
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0],
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "gaussian_noise attack requires 'target_noise_snr' parameter" in str(
            exc_info.value
        )

    def test_gaussian_noise_missing_attack_ratio(self):
        """Test validation fails when gaussian_noise is missing attack_ratio."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "gaussian_noise",
                    "target_noise_snr": 10.0,
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0],
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "gaussian_noise attack requires 'attack_ratio' parameter" in str(
            exc_info.value
        )

    def test_specific_selection_missing_malicious_client_ids(self):
        """Test validation fails when specific selection is missing malicious_client_ids."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert (
            "'specific' selection strategy requires 'malicious_client_ids' list"
            in str(exc_info.value)
        )

    def test_random_selection_missing_malicious_client_count(self):
        """Test validation fails when random selection is missing malicious_client_count."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "random",
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert (
            "'random' selection strategy requires 'malicious_client_count' integer"
            in str(exc_info.value)
        )

    def test_percentage_selection_missing_malicious_percentage(self):
        """Test validation fails when percentage selection is missing malicious_percentage."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
            "num_of_malicious_clients": 1,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'percentage' selection strategy requires 'malicious_percentage'" in str(
            exc_info.value
        )

    def test_overlapping_attacks_same_type_raises_error(self):
        """Test that overlapping attacks with same type raises ValidationError."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 5,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0],
                },
                {
                    "start_round": 3,
                    "end_round": 7,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [1],
                },
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "Overlapping rounds with same attack type" in str(exc_info.value)
        assert "label_flipping" in str(exc_info.value)

    def test_overlapping_attacks_different_types_logs_info(self, caplog):
        """Test that overlapping attacks with different types log info."""
        import logging

        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 5,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0],
                },
                {
                    "start_round": 3,
                    "end_round": 7,
                    "attack_type": "gaussian_noise",
                    "target_noise_snr": 10.0,
                    "attack_ratio": 0.3,
                    "selection_strategy": "specific",
                    "malicious_client_ids": [1],
                },
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with caplog.at_level(logging.INFO):
            validate_strategy_config(config)

        assert any(
            "overlapping rounds with different attack types" in record.message
            for record in caplog.records
        )
        assert any(
            "Both attacks will be stacked and applied sequentially" in record.message
            for record in caplog.records
        )


class TestValidateStrategyConfigLlmIntegration:
    """Test LLM integration in the main validation function."""

    def test_validate_strategy_config_calls_llm_validation(self):
        """Test that main validation calls LLM validation when use_llm is true."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "medquad",
            "model_type": "transformer",
            "use_llm": "true",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                    "malicious_percentage": 0.2,
                }
            ],
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
            # Missing LLM parameters should cause error
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        # Should call LLM validation and fail due to missing parameters
        assert "Missing parameter llm_model for LLM finetuning" in str(exc_info.value)

    def test_validate_strategy_config_skips_llm_validation_when_disabled(self):
        """Test that main validation skips LLM validation when use_llm is false."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                    "malicious_percentage": 0.2,
                }
            ],
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
            # No LLM parameters needed when use_llm is false
        }

        # Should not raise any exception
        validate_strategy_config(config)

    def test_validate_strategy_config_checks_client_numbers(self):
        """Test that main validation checks client number consistency."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,  # Too few clients
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                    "malicious_percentage": 0.4,
                }
            ],
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",
            "training_subset_fraction": 0.8,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 8,  # More than num_of_clients
            "min_evaluate_clients": 8,  # More than num_of_clients
            "min_available_clients": 10,  # More than num_of_clients
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        # Should fail due to insufficient clients
        error_message = str(exc_info.value)
        assert "EXPERIMENT STOPPED: Client configuration error" in error_message

    def test_preserve_dataset_forced_false_with_attack_schedule(self):
        """Test that preserve_dataset=true with attack_schedule raises ValidationError."""
        config = {
            "aggregation_strategy_keyword": "krum",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 10,
            "num_of_malicious_clients": 3,
            "attack_schedule": [
                {
                    "start_round": 3,
                    "end_round": 8,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0, 1, 2],
                }
            ],
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "true",
            "training_subset_fraction": 0.5,
            "training_device": "cpu",
            "cpus_per_client": 2,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 2,
            "batch_size": 32,
            # Krum-specific parameters
            "num_krum_selections": 6,
        }

        # Should raise ValidationError with CONFIG REJECTED message
        with pytest.raises(ValidationError, match="CONFIG REJECTED"):
            validate_strategy_config(config)

    def test_preserve_dataset_remains_false_with_attack_schedule(self):
        """Test that preserve_dataset remains false when already set correctly."""
        config = {
            "aggregation_strategy_keyword": "krum",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 10,
            "num_of_malicious_clients": 3,
            "attack_schedule": [
                {
                    "start_round": 3,
                    "end_round": 8,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0, 1, 2],
                }
            ],
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "false",  # Already false
            "training_subset_fraction": 0.5,
            "training_device": "cpu",
            "cpus_per_client": 2,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 2,
            "batch_size": 32,
            # Krum-specific parameters
            "num_krum_selections": 6,
        }

        # Should not raise exception
        validate_strategy_config(config)

        # preserve_dataset should remain false
        assert config["preserve_dataset"] == "false"

    def test_preserve_dataset_unchanged_without_attack_schedule(self):
        """Test that preserve_dataset is not modified when no attack_schedule is present."""
        config = {
            "aggregation_strategy_keyword": "krum",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 10,
            "num_of_clients": 10,
            "num_of_malicious_clients": 3,
            "attack_schedule": [],  # Empty attack schedule
            "show_plots": "false",
            "save_plots": "true",
            "save_csv": "true",
            "preserve_dataset": "true",  # Can remain true with empty schedule
            "training_subset_fraction": 0.5,
            "training_device": "cpu",
            "cpus_per_client": 2,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 2,
            "batch_size": 32,
            # Krum-specific parameters
            "num_krum_selections": 6,
        }

        # Should not raise exception
        validate_strategy_config(config)

        # preserve_dataset should remain true (no attack schedule)
        assert config["preserve_dataset"] == "true"

    def test_percentage_selection_populates_selected_clients(self):
        """Test that percentage selection strategy populates _selected_clients."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                    "malicious_percentage": 0.2,  # 20% of 10 = 2 clients
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        validate_strategy_config(config)

        assert "_selected_clients" in config["attack_schedule"][0]
        selected = config["attack_schedule"][0]["_selected_clients"]

        assert len(selected) == 2
        assert all(0 <= cid < 10 for cid in selected)
        assert selected == sorted(selected)

    def test_random_selection_populates_selected_clients(self):
        """Test that random selection strategy populates _selected_clients."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 3,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "gaussian_noise",
                    "target_noise_snr": 10.0,
                    "attack_ratio": 1.0,
                    "selection_strategy": "random",
                    "malicious_client_count": 3,
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        validate_strategy_config(config)

        assert "_selected_clients" in config["attack_schedule"][0]
        selected = config["attack_schedule"][0]["_selected_clients"]

        assert len(selected) == 3
        assert all(0 <= cid < 10 for cid in selected)
        assert selected == sorted(selected)

    def test_client_selection_with_custom_seed_is_deterministic(self):
        """Test that same random_seed produces same client selection."""
        config1 = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                    "malicious_percentage": 0.2,
                    "random_seed": 42,
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        config2 = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "percentage",
                    "malicious_percentage": 0.2,
                    "random_seed": 42,
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        validate_strategy_config(config1)
        validate_strategy_config(config2)

        # Same seed should produce same selection
        selected1 = config1["attack_schedule"][0]["_selected_clients"]
        selected2 = config2["attack_schedule"][0]["_selected_clients"]
        assert selected1 == selected2

    def test_random_selection_different_seeds_produce_different_selections(self):
        """Test that different seeds produce different client selections for random strategy."""
        config_seed_42 = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 20,
            "num_of_malicious_clients": 5,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "gaussian_noise",
                    "target_noise_snr": 10.0,
                    "attack_ratio": 1.0,
                    "selection_strategy": "random",
                    "malicious_client_count": 5,
                    "random_seed": 42,
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 20,
            "min_evaluate_clients": 20,
            "min_available_clients": 20,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        config_seed_99 = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 20,
            "num_of_malicious_clients": 5,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "gaussian_noise",
                    "target_noise_snr": 10.0,
                    "attack_ratio": 1.0,
                    "selection_strategy": "random",
                    "malicious_client_count": 5,
                    "random_seed": 99,
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 20,
            "min_evaluate_clients": 20,
            "min_available_clients": 20,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        config_seed_42_again = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 20,
            "num_of_malicious_clients": 5,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "gaussian_noise",
                    "target_noise_snr": 10.0,
                    "attack_ratio": 1.0,
                    "selection_strategy": "random",
                    "malicious_client_count": 5,
                    "random_seed": 42,
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 20,
            "min_evaluate_clients": 20,
            "min_available_clients": 20,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        validate_strategy_config(config_seed_42)
        validate_strategy_config(config_seed_99)
        validate_strategy_config(config_seed_42_again)

        selected_42 = config_seed_42["attack_schedule"][0]["_selected_clients"]
        selected_99 = config_seed_99["attack_schedule"][0]["_selected_clients"]
        selected_42_again = config_seed_42_again["attack_schedule"][0][
            "_selected_clients"
        ]

        # Different seeds should produce different selections
        assert selected_42 != selected_99, (
            f"Different seeds should produce different selections: "
            f"seed 42 -> {selected_42}, seed 99 -> {selected_99}"
        )

        # Same seed should produce same selection (reproducibility)
        assert selected_42 == selected_42_again, (
            f"Same seed should produce same selection: "
            f"first run -> {selected_42}, second run -> {selected_42_again}"
        )

        # Verify all selections are valid
        for selected in [selected_42, selected_99, selected_42_again]:
            assert len(selected) == 5
            assert all(0 <= cid < 20 for cid in selected)
            assert selected == sorted(selected)

    def test_percentage_selection_too_many_clients_fails(self):
        """Test that percentage selection fails if calculated count exceeds num_of_clients."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
            "num_of_malicious_clients": 6,  # More than available
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "random",
                    "malicious_client_count": 10,  # More than available
                }
            ],
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
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(
            ValidationError, match="Cannot select 10 clients when only 5 clients exist"
        ):
            validate_strategy_config(config)

    def test_specific_selection_unchanged(self):
        """Test that specific selection strategy is not modified."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "num_of_malicious_clients": 2,
            "attack_schedule": [
                {
                    "start_round": 1,
                    "end_round": 5,
                    "attack_type": "label_flipping",
                    "selection_strategy": "specific",
                    "malicious_client_ids": [0, 3, 7],
                }
            ],
            "show_plots": "false",
            "save_plots": "false",
            "save_csv": "false",
            "preserve_dataset": "false",
            "training_subset_fraction": 1.0,
            "training_device": "cpu",
            "cpus_per_client": 1,
            "gpus_per_client": 0.0,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 1,
            "batch_size": 32,
            "begin_removing_from_round": 1,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        validate_strategy_config(config)

        assert "_selected_clients" not in config["attack_schedule"][0]
        assert config["attack_schedule"][0]["malicious_client_ids"] == [0, 3, 7]
