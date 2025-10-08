"""
Unit tests for strategy configuration validation.

Tests strategy parameter validation, error handling for invalid JSON and missing parameters,
and clear error message generation.
"""

from tests.common import pytest
from jsonschema import ValidationError
from src.config_loaders.validate_strategy_config import (
    check_llm_specific_parameters,
    validate_dependent_params,
    validate_strategy_config,
    validate_huggingface_dataset,
)


# Strategy-specific required parameters mapping
STRATEGY_REQUIRED_PARAMS = {
    "trust": [
        "trust_threshold",
        "beta_value",
        "begin_removing_from_round",
        "num_of_clusters",
    ],
    "pid": ["Kp", "Ki", "Kd", "num_std_dev"],
    "pid_scaled": ["Kp", "Ki", "Kd", "num_std_dev"],
    "pid_standardized": ["Kp", "Ki", "Kd", "num_std_dev"],
    "krum": ["num_krum_selections"],
    "multi-krum": ["num_krum_selections"],
    "multi-krum-based": ["num_krum_selections"],
    "trimmed_mean": ["trim_ratio"],
}

# Flatten to (strategy, param, expected_error) tuples
MISSING_PARAM_TEST_CASES = []
for strategy, params in STRATEGY_REQUIRED_PARAMS.items():
    for param in params:
        if strategy.startswith("pid"):
            error_msg = f"Missing parameter {param} for PID aggregation {strategy}"
        elif "krum" in strategy:
            error_msg = (
                f"Missing parameter {param} for Krum-based aggregation {strategy}"
            )
        elif strategy == "trust":
            error_msg = f"Missing parameter {param} for trust aggregation trust"
        elif strategy == "trimmed_mean":
            error_msg = (
                f"Missing parameter {param} for trimmed mean aggregation trimmed_mean"
            )
        else:
            error_msg = f"Missing parameter {param}"

        MISSING_PARAM_TEST_CASES.append((strategy, param, error_msg))

# Add attack-specific parameter tests
MISSING_PARAM_TEST_CASES.extend(
    [
        (
            "gaussian_noise",
            "gaussian_noise_mean",
            "Missing gaussian_noise_mean that is required for gaussian_noise",
        ),
        (
            "gaussian_noise",
            "gaussian_noise_std",
            "Missing gaussian_noise_std that is required for gaussian_noise",
        ),
        (
            "gaussian_noise",
            "attack_ratio",
            "Missing attack_ratio that is required for gaussian_noise",
        ),
    ]
)

# Valid strategy configurations
VALID_STRATEGY_CONFIGS = [
    (
        "trust",
        {
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
            "strict_mode": "true",
        },
    ),
    (
        "pid",
        {
            "dataset_keyword": "its",
            "attack_type": "gaussian_noise",
            "num_std_dev": 2.0,
            "Kp": 1.0,
            "Ki": 0.1,
            "Kd": 0.01,
            "gaussian_noise_mean": 0.0,
            "gaussian_noise_std": 0.1,
            "attack_ratio": 0.2,
            "strict_mode": "true",
        },
    ),
    (
        "krum",
        {
            "dataset_keyword": "pneumoniamnist",
            "remove_clients": "false",
            "num_of_malicious_clients": 0,
            "num_krum_selections": 8,
            "training_device": "cuda",
            "gpus_per_client": 1.0,
        },
    ),
    (
        "trimmed_mean",
        {
            "dataset_keyword": "bloodmnist",
            "trim_ratio": 0.2,
        },
    ),
]

# Invalid field value test cases
INVALID_VALUE_TEST_CASES = [
    (
        "aggregation_strategy_keyword",
        "invalid_strategy",
        "'invalid_strategy' is not one of",
    ),
    ("dataset_keyword", "invalid_dataset", "'invalid_dataset' is not one of"),
    ("remove_clients", "maybe", "'maybe' is not one of ['true', 'false']"),
    (
        "attack_type",
        "invalid_attack",
        "'invalid_attack' is not one of ['label_flipping', 'gaussian_noise']",
    ),
    ("training_device", "quantum", "'quantum' is not one of ['cpu', 'gpu', 'cuda']"),
    ("num_of_rounds", "five", "'five' is not of type 'integer'"),
]

# LLM parameter test cases
LLM_MISSING_PARAM_TESTS = [
    (
        {
            "model_type": "transformer",
            "llm_finetuning": "full",
            "llm_task": "classification",
            "llm_chunk_size": 512,
        },
        "Missing parameter llm_model for LLM finetuning",
    ),
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_task": "classification",
            "llm_chunk_size": 512,
        },
        "Missing parameter llm_finetuning for LLM finetuning",
    ),
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "full",
            "llm_chunk_size": 512,
        },
        "Missing parameter llm_task for LLM finetuning",
    ),
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "full",
            "llm_task": "classification",
        },
        "Missing parameter llm_chunk_size for LLM finetuning",
    ),
]

# LORA-specific parameters
LORA_MISSING_PARAM_TESTS = [
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "lora",
            "llm_task": "classification",
            "llm_chunk_size": 512,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["query", "value"],
        },
        "Missing parameter lora_rank for LORA",
    ),
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "lora",
            "llm_task": "classification",
            "llm_chunk_size": 512,
            "lora_rank": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["query", "value"],
        },
        "Missing parameter lora_alpha for LORA",
    ),
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "lora",
            "llm_task": "classification",
            "llm_chunk_size": 512,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_target_modules": ["query", "value"],
        },
        "Missing parameter lora_dropout for LORA",
    ),
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "lora",
            "llm_task": "classification",
            "llm_chunk_size": 512,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        },
        "Missing parameter lora_target_modules for LORA",
    ),
]

# MLM task-specific parameters
MLM_MISSING_PARAM_TESTS = [
    (
        {
            "model_type": "transformer",
            "llm_model": "bert-base-uncased",
            "llm_finetuning": "full",
            "llm_task": "mlm",
            "llm_chunk_size": 512,
        },
        "Missing parameter mlm_probability for LLM task mlm",
    ),
]

# Valid LLM configurations
VALID_LLM_CONFIGS = [
    {
        "model_type": "transformer",
        "llm_model": "bert-base-uncased",
        "llm_finetuning": "full",
        "llm_task": "classification",
        "llm_chunk_size": 512,
    },
    {
        "model_type": "transformer",
        "llm_model": "bert-base-uncased",
        "llm_finetuning": "lora",
        "llm_task": "classification",
        "llm_chunk_size": 512,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["query", "value"],
    },
    {
        "model_type": "transformer",
        "llm_model": "bert-base-uncased",
        "llm_finetuning": "full",
        "llm_task": "mlm",
        "llm_chunk_size": 512,
        "mlm_probability": 0.15,
    },
]


class TestValidateStrategyConfig:
    """Test suite for strategy configuration validation functionality."""

    @pytest.mark.parametrize(
        "strategy_keyword,strategy_specific_params", VALID_STRATEGY_CONFIGS
    )
    def test_valid_strategy_configurations(
        self, strategy_keyword, strategy_specific_params, base_valid_config
    ):
        """Test validation of valid strategy configurations."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = strategy_keyword
        config.update(strategy_specific_params)

        # Should not raise any exception
        validate_strategy_config(config)


class TestValidateStrategyConfigMissingRequiredParams:
    """Test validation errors for missing required parameters."""

    def test_missing_aggregation_strategy_keyword(self, base_valid_config):
        """Test validation fails when aggregation_strategy_keyword is missing."""
        config = base_valid_config.copy()
        # Don't set aggregation_strategy_keyword

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'aggregation_strategy_keyword' is a required property" in str(
            exc_info.value
        )

    def test_missing_dataset_keyword(self, base_valid_config):
        """Test validation fails when dataset_keyword is missing."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = "trust"
        del config["dataset_keyword"]

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'dataset_keyword' is a required property" in str(exc_info.value)

    def test_missing_num_of_rounds(self, base_valid_config):
        """Test validation fails when num_of_rounds is missing."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = "trust"
        del config["num_of_rounds"]

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'num_of_rounds' is a required property" in str(exc_info.value)

    def test_missing_flower_settings(self, base_valid_config):
        """Test validation fails when Flower-specific settings are missing."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = "trust"
        # Remove Flower settings
        del config["training_device"]
        del config["cpus_per_client"]
        del config["gpus_per_client"]

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        error_message = str(exc_info.value)
        assert "'training_device' is a required property" in error_message


class TestValidateStrategyConfigInvalidValues:
    """Test validation errors for invalid parameter values."""

    @pytest.mark.parametrize(
        "invalid_field,invalid_value,expected_error_fragment", INVALID_VALUE_TEST_CASES
    )
    def test_invalid_config_values(
        self, invalid_field, invalid_value, expected_error_fragment, base_valid_config
    ):
        """Test validation fails for invalid parameter values."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = "trust"
        config.update(
            {
                "begin_removing_from_round": 2,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
                "num_of_clusters": 1,
                "strict_mode": "true",
            }
        )
        config[invalid_field] = invalid_value

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert expected_error_fragment in str(exc_info.value)


class TestValidateDependentParams:
    """Test validation of strategy-specific dependent parameters."""

    @pytest.mark.parametrize(
        "strategy,missing_param,expected_error", MISSING_PARAM_TEST_CASES
    )
    def test_missing_strategy_specific_params(
        self, strategy, missing_param, expected_error
    ):
        """Test validation fails when strategy-specific parameters are missing."""
        if strategy == "gaussian_noise":
            # Special case: attack type, not aggregation strategy
            config = {
                "aggregation_strategy_keyword": "trust",
                "attack_type": "gaussian_noise",
                # Trust params required first
                "begin_removing_from_round": 2,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
                "num_of_clusters": 1,
                # Add all gaussian noise params except the one being tested
                "gaussian_noise_mean": 0.0,
                "gaussian_noise_std": 0.1,
                "attack_ratio": 0.2,
            }
            # Remove the missing param
            config.pop(missing_param, None)
        else:
            # Standard strategy parameter test
            # Start with all params for this strategy, then remove the one being tested
            config = {"aggregation_strategy_keyword": strategy}

            # Add all required params for this strategy
            if strategy in STRATEGY_REQUIRED_PARAMS:
                for param in STRATEGY_REQUIRED_PARAMS[strategy]:
                    if param == "trust_threshold":
                        config[param] = 0.7
                    elif param == "beta_value":
                        config[param] = 0.5
                    elif param == "begin_removing_from_round":
                        config[param] = 2
                    elif param == "num_of_clusters":
                        config[param] = 1
                    elif param in ["Kp", "Ki", "Kd"]:
                        config[param] = 1.0
                    elif param == "num_std_dev":
                        config[param] = 2.0
                    elif param == "num_krum_selections":
                        config[param] = 8
                    elif param == "trim_ratio":
                        config[param] = 0.2

            # Remove the missing param
            config.pop(missing_param, None)

        with pytest.raises(ValidationError) as exc_info:
            validate_dependent_params(config)

        assert expected_error in str(exc_info.value)

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


class TestValidateStrategyConfigErrorMessages:
    """Test that validation provides clear and helpful error messages."""

    def test_clear_error_message_for_invalid_enum_value(self, base_valid_config):
        """Test that error messages clearly indicate valid enum options."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = "invalid_strategy"

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

    def test_clear_error_message_for_wrong_data_type(self, base_valid_config):
        """Test that error messages clearly indicate expected data type."""
        config = base_valid_config.copy()
        config["aggregation_strategy_keyword"] = "trust"
        config["num_of_rounds"] = "not_a_number"  # Should be integer

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


class TestCheckLlmSpecificParameters:
    """Test suite for LLM-specific parameter validation."""

    def test_llm_enabled_with_cnn_model_fails(self):
        """Test that LLM is rejected for non-transformer models."""
        config = {"model_type": "cnn", "use_llm": "true"}

        with pytest.raises(ValidationError) as exc_info:
            check_llm_specific_parameters(config)

        assert "LLM finetuning is only supported for transformer models" in str(
            exc_info.value
        )

    @pytest.mark.parametrize(
        "config,expected_error",
        LLM_MISSING_PARAM_TESTS,
        ids=[
            "missing_llm_model",
            "missing_llm_finetuning",
            "missing_llm_task",
            "missing_llm_chunk_size",
        ],
    )
    def test_llm_missing_base_parameters(self, config, expected_error):
        """Test that validation fails when LLM config is missing required base parameters."""
        with pytest.raises(ValidationError) as exc_info:
            check_llm_specific_parameters(config)

        assert expected_error in str(exc_info.value)

    @pytest.mark.parametrize(
        "config,expected_error",
        MLM_MISSING_PARAM_TESTS,
        ids=["missing_mlm_probability"],
    )
    def test_mlm_missing_parameters(self, config, expected_error):
        """Test that validation fails when MLM task is missing required parameters."""
        with pytest.raises(ValidationError) as exc_info:
            check_llm_specific_parameters(config)

        assert expected_error in str(exc_info.value)

    @pytest.mark.parametrize(
        "config,expected_error",
        LORA_MISSING_PARAM_TESTS,
        ids=[
            "missing_lora_rank",
            "missing_lora_alpha",
            "missing_lora_dropout",
            "missing_lora_target_modules",
        ],
    )
    def test_lora_missing_parameters(self, config, expected_error):
        """Test that validation fails when LORA finetuning is missing required parameters."""
        with pytest.raises(ValidationError) as exc_info:
            check_llm_specific_parameters(config)

        assert expected_error in str(exc_info.value)

    @pytest.mark.parametrize(
        "config",
        VALID_LLM_CONFIGS,
        ids=["valid_full_finetuning", "valid_lora_finetuning", "valid_mlm_task"],
    )
    def test_valid_llm_configurations(self, config):
        """Test that valid LLM configurations pass validation."""
        # Should not raise any exception
        check_llm_specific_parameters(config)


class TestStrictModeValidation:
    """Test suite for strict_mode validation functionality."""

    def test_strict_mode_defaults_to_true(self):
        """Test that strict_mode defaults to 'true' when not specified."""
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
            # strict_mode not specified - should default to "true"
        }

        # Should not raise exception and auto-configure clients
        validate_strategy_config(config)
        assert config["strict_mode"] == "true"
        assert config["min_fit_clients"] == 10
        assert config["min_evaluate_clients"] == 10
        assert config["min_available_clients"] == 10

    def test_strict_mode_enabled_auto_configures_clients(self):
        """Test that strict_mode=true forces all min_* values to equal num_of_clients."""
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
            "min_fit_clients": 5,  # Will be auto-configured to 10
            "min_evaluate_clients": 7,  # Will be auto-configured to 10
            "min_available_clients": 8,  # Will be auto-configured to 10
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            "strict_mode": "true",
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        # Should not raise exception and auto-configure all client values
        validate_strategy_config(config)
        assert config["min_fit_clients"] == 10
        assert config["min_evaluate_clients"] == 10
        assert config["min_available_clients"] == 10

    def test_strict_mode_disabled_preserves_client_config(self):
        """Test that strict_mode=false preserves original client configuration."""
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
            "min_fit_clients": 5,
            "min_evaluate_clients": 7,
            "min_available_clients": 8,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            "strict_mode": "false",
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        # Should not raise exception and preserve original values
        validate_strategy_config(config)
        assert config["min_fit_clients"] == 5
        assert config["min_evaluate_clients"] == 7
        assert config["min_available_clients"] == 8

    def test_strict_mode_fails_when_min_clients_exceed_total(self):
        """Test that validation fails when min_* > num_of_clients regardless of strict_mode."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "remove_clients": "true",
            "dataset_keyword": "femnist_iid",
            "model_type": "cnn",
            "use_llm": "false",
            "num_of_rounds": 5,
            "num_of_clients": 5,
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
            "min_fit_clients": 8,  # > num_of_clients
            "min_evaluate_clients": 6,  # > num_of_clients
            "min_available_clients": 5,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            "strict_mode": "false",
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        error_message = str(exc_info.value)
        assert "EXPERIMENT STOPPED: Client configuration error" in error_message
        assert "Cannot require more clients than available" in error_message
        assert "Total clients: 5" in error_message
        assert "min_fit_clients: 8" in error_message
        assert "min_evaluate_clients: 6" in error_message

    def test_strict_mode_with_already_configured_clients(self):
        """Test that strict_mode=true does not modify already correct client config."""
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
            "min_fit_clients": 10,  # Already correct
            "min_evaluate_clients": 10,  # Already correct
            "min_available_clients": 10,  # Already correct
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            "strict_mode": "true",
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        # Should not raise exception and preserve correct values
        validate_strategy_config(config)
        assert config["min_fit_clients"] == 10
        assert config["min_evaluate_clients"] == 10
        assert config["min_available_clients"] == 10

    def test_strict_mode_invalid_value_fails_schema_validation(self):
        """Test that invalid strict_mode values fail schema validation."""
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
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
            "evaluate_metrics_aggregation_fn": "weighted_average",
            "num_of_client_epochs": 3,
            "batch_size": 32,
            "strict_mode": "maybe",  # Invalid value
            # Trust-specific parameters
            "begin_removing_from_round": 2,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
            "num_of_clusters": 1,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_config(config)

        assert "'maybe' is not one of ['true', 'false']" in str(exc_info.value)


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
            "attack_type": "label_flipping",
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


class TestValidateHuggingFaceDataset:
    """Test suite for HuggingFace dataset validation functionality."""

    def test_local_dataset_source_no_validation_required(self):
        """Test that local dataset source doesn't require HF-specific parameters."""
        config = {
            "dataset_source": "local",
            "dataset_keyword": "femnist_iid",
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_default_dataset_source_no_validation_required(self):
        """Test that missing dataset_source defaults to local and doesn't require HF parameters."""
        config = {
            "dataset_keyword": "femnist_iid",
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_huggingface_source_requires_dataset_name(self):
        """Test that HuggingFace source requires hf_dataset_name parameter."""
        config = {
            "dataset_source": "huggingface",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        assert "hf_dataset_name is required when dataset_source='huggingface'" in str(
            exc_info.value
        )

    def test_huggingface_source_with_empty_dataset_name_fails(self):
        """Test that empty hf_dataset_name is treated as missing."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        assert "hf_dataset_name is required when dataset_source='huggingface'" in str(
            exc_info.value
        )

    def test_huggingface_with_valid_iid_strategy(self):
        """Test that IID partitioning strategy is valid for HuggingFace datasets."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "iid",
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_huggingface_with_default_iid_strategy(self):
        """Test that partitioning_strategy defaults to IID when not specified."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "cifar10",
        }

        # Should not raise any exception (defaults to iid)
        validate_huggingface_dataset(config)

    def test_huggingface_with_valid_dirichlet_strategy(self):
        """Test that Dirichlet partitioning strategy is valid for HuggingFace datasets."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": 0.5},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_huggingface_with_valid_pathological_strategy(self):
        """Test that pathological partitioning strategy is valid for HuggingFace datasets."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "pathological",
            "partitioning_params": {"num_classes_per_partition": 2},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_huggingface_with_invalid_strategy_fails(self):
        """Test that invalid partitioning strategy fails validation."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "invalid_strategy",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        error_message = str(exc_info.value)
        assert "Invalid partitioning_strategy: invalid_strategy" in error_message
        assert "Must be one of: iid, dirichlet, pathological" in error_message

    def test_dirichlet_alpha_minimum_boundary(self):
        """Test that Dirichlet alpha at minimum boundary (just above 0) is valid."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": 0.01},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_dirichlet_alpha_maximum_boundary(self):
        """Test that Dirichlet alpha at maximum boundary (10) is valid."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": 10.0},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_dirichlet_alpha_zero_fails(self):
        """Test that Dirichlet alpha of 0 fails validation."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": 0.0},
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        error_message = str(exc_info.value)
        assert "Dirichlet alpha must be in range (0, 10]" in error_message
        assert "got: 0" in error_message

    def test_dirichlet_alpha_negative_fails(self):
        """Test that negative Dirichlet alpha fails validation."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": -1.0},
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        error_message = str(exc_info.value)
        assert "Dirichlet alpha must be in range (0, 10]" in error_message
        assert "got: -1" in error_message

    def test_dirichlet_alpha_above_maximum_fails(self):
        """Test that Dirichlet alpha above 10 fails validation."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": 10.5},
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        error_message = str(exc_info.value)
        assert "Dirichlet alpha must be in range (0, 10]" in error_message
        assert "got: 10.5" in error_message

    def test_dirichlet_with_default_alpha(self):
        """Test that Dirichlet strategy uses default alpha of 0.5 when not specified."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
        }

        # Should not raise any exception (defaults to alpha=0.5)
        validate_huggingface_dataset(config)

    def test_dirichlet_with_empty_params(self):
        """Test that Dirichlet strategy works with empty partitioning_params."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {},
        }

        # Should not raise any exception (defaults to alpha=0.5)
        validate_huggingface_dataset(config)

    def test_pathological_num_classes_minimum_boundary(self):
        """Test that pathological with 1 class per partition is valid."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "pathological",
            "partitioning_params": {"num_classes_per_partition": 1},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_pathological_num_classes_zero_fails(self):
        """Test that pathological with 0 classes per partition fails validation."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "pathological",
            "partitioning_params": {"num_classes_per_partition": 0},
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        error_message = str(exc_info.value)
        assert "num_classes_per_partition must be >= 1" in error_message
        assert "got: 0" in error_message

    def test_pathological_num_classes_negative_fails(self):
        """Test that pathological with negative classes per partition fails validation."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "pathological",
            "partitioning_params": {"num_classes_per_partition": -1},
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_huggingface_dataset(config)

        error_message = str(exc_info.value)
        assert "num_classes_per_partition must be >= 1" in error_message
        assert "got: -1" in error_message

    def test_pathological_with_default_num_classes(self):
        """Test that pathological strategy uses default of 2 classes when not specified."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "pathological",
        }

        # Should not raise any exception (defaults to 2 classes)
        validate_huggingface_dataset(config)

    def test_pathological_with_empty_params(self):
        """Test that pathological strategy works with empty partitioning_params."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "mnist",
            "partitioning_strategy": "pathological",
            "partitioning_params": {},
        }

        # Should not raise any exception (defaults to 2 classes)
        validate_huggingface_dataset(config)

    def test_huggingface_complete_config_with_dirichlet(self):
        """Test complete HuggingFace configuration with Dirichlet partitioning."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "uoft-cs/cifar10",
            "partitioning_strategy": "dirichlet",
            "partitioning_params": {"alpha": 1.0},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)

    def test_huggingface_complete_config_with_pathological(self):
        """Test complete HuggingFace configuration with pathological partitioning."""
        config = {
            "dataset_source": "huggingface",
            "hf_dataset_name": "ylecun/mnist",
            "partitioning_strategy": "pathological",
            "partitioning_params": {"num_classes_per_partition": 3},
        }

        # Should not raise any exception
        validate_huggingface_dataset(config)
