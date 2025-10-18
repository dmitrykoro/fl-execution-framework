from jsonschema import validate, ValidationError

config_schema = {
    "type": "object",
    "properties": {
        # Common parameters
        "aggregation_strategy_keyword": {
            "type": "string",
            "enum": ["trust", "pid", "pid_scaled", "pid_standardized",
                    "multi-krum", "krum", "multi-krum-based", "trimmed_mean",
                    "rfa", "bulyan"]
        },
        "remove_clients": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "dataset_keyword": {
            "type": "string",
            "enum": ["femnist_iid", "femnist_niid", "its", "pneumoniamnist", "flair", "bloodmnist", "medquad", "lung_photos"]
        },
        "model_type": {
            "type": "string",
            "enum": ["cnn", "transformer"]
        },
        "num_of_rounds": {
            "type": "integer"
        },
        "num_of_clients": {
            "type": "integer"
        },
        "num_of_malicious_clients": {
            "type": "integer"
        },
        "attack_type": {
            "type": "string",
            "enum": ["label_flipping", "gaussian_noise"]
        },
        "show_plots": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "save_plots": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "save_csv": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "preserve_dataset": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "training_subset_fraction": {
            "type": "number"
        },

        # LLM settings
        "use_llm": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "llm_model": {
            "type": "string",
            "enum": ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"]
        },
        "llm_task": {
            "type": "string",
            "enum": ["mlm"]
        },
        "mlm_probability": {
            "type": "number"
        },
        "llm_chunk_size": {
            "type": "integer"
        },
        "llm_finetuning": {
            "type": "string",
            "enum": ["full", "lora"]
        },
        "lora_rank": {
            "type": "integer"
        },
        "lora_alpha": {
            "type": "integer"
        },
        "lora_dropout": {
            "type": "number"
        },
        "lora_target_modules": {
            "type": "array",
            "items": {
                "type": "string",
            }
        },

        # Flower settings
        "training_device": {
            "type": "string",
            "enum": ["cpu", "gpu", "cuda"]
        },
        "cpus_per_client": {
            "type": "integer"
        },
        "gpus_per_client": {
            "type": "number"
        },
        "min_fit_clients": {
            "type": "integer"
        },
        "min_evaluate_clients": {
            "type": "integer"
        },
        "min_available_clients": {
            "type": "integer"
        },
        "evaluate_metrics_aggregation_fn": {
            "type": "string",
        },
        "num_of_client_epochs": {
            "type": "integer"
        },
        "batch_size": {
            "type": "integer"
        },

        # Strategy specific parameters

        # Trust
        "begin_removing_from_round": {
            "type": "integer"
        },
        "trust_threshold": {
            "type": "number"
        },
        "beta_value": {
            "type": "number"
        },
        "num_of_clusters": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1
        },

        # PID, PID scaled, PID standardized
        "num_std_dev": {
            "type": "number"
        },
        "Kp": {
            "type": "number"
        },
        "Ki": {
            "type": "number"
        },
        "Kd": {
            "type": "number"
        },

        # Krum-based strategies
        "num_krum_selections": {
            "type": "integer"
        },

        # Trimmed mean
        "trim_ratio": {
            "type": "number"
        },

        # Dynamic poisoning attacks
        "dynamic_attacks": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean"
                },
                "schedule": {
                    "type": "array"
                }
            }
        }
    },
    "required": [
        "aggregation_strategy_keyword", "remove_clients", "dataset_keyword", "model_type",
        "use_llm", "num_of_rounds", "num_of_clients", "num_of_malicious_clients",
        "attack_type", "show_plots", "save_plots", "save_csv", "preserve_dataset",
        "training_subset_fraction", "training_device", "cpus_per_client",
        "gpus_per_client", "min_fit_clients", "min_evaluate_clients",
        "min_available_clients", "evaluate_metrics_aggregation_fn",
        "num_of_client_epochs", "batch_size"
    ]
}


def validate_dependent_params(strategy_config: dict) -> None:
    """Validate that all params that require additional params are correct."""

    aggregation_strategy_keyword = strategy_config["aggregation_strategy_keyword"]

    if aggregation_strategy_keyword == "trust":
        trust_specific_parameters = [
            "begin_removing_from_round", "trust_threshold", "beta_value", "num_of_clusters"
        ]
        for param in trust_specific_parameters:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing parameter {param} for trust aggregation {aggregation_strategy_keyword}"
                )
    elif aggregation_strategy_keyword in ("pid", "pid_scaled", "pid_standardized"):
        pid_specific_parameters = [
            "num_std_dev", "Kp", "Ki", "Kd"
        ]
        for param in pid_specific_parameters:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing parameter {param} for PID aggregation {aggregation_strategy_keyword}"
                )
    elif aggregation_strategy_keyword in ["multi-krum", "krum", "multi-krum-based"]:
        if "num_krum_selections" not in strategy_config:
            raise ValidationError(
                f"Missing parameter num_krum_selections for Krum-based aggregation {aggregation_strategy_keyword}"
            )
    elif aggregation_strategy_keyword == "trimmed_mean":
        if "trim_ratio" not in strategy_config:
            raise ValidationError(
                f"Missing parameter trim_ratio for trimmed mean aggregation {aggregation_strategy_keyword}"
            )

    attack_type = strategy_config["attack_type"]

    if attack_type == "gaussian_noise":
        gaussian_noise_specific_params = [
            "gaussian_noise_mean", "gaussian_noise_std", "attack_ratio"
        ]
        for param in gaussian_noise_specific_params:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing {param} that is required for {attack_type} in configuration."
                )

def check_llm_specific_parameters(strategy_config: dict) -> None:
    """Check if LLM specific parameters are valid"""

    if strategy_config["model_type"] != "transformer":
        raise ValidationError(
            "LLM finetuning is only supported for transformer models"
        )

    llm_specific_parameters = [
        "llm_model", "llm_finetuning", "llm_task", "llm_chunk_size"
    ]
    for param in llm_specific_parameters:
        if param not in strategy_config:
            raise ValidationError(
                f"Missing parameter {param} for LLM finetuning"
            )

    if strategy_config["llm_task"] == "mlm":
        if "mlm_probability" not in strategy_config:
            raise ValidationError(
                "Missing parameter mlm_probability for LLM task mlm"
            )

    finetuning_keyword = strategy_config["llm_finetuning"]
    if finetuning_keyword == "lora":
        lora_specific_parameters = [
            "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules"
        ]
        for param in lora_specific_parameters:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing parameter {param} for LORA"
                )


def validate_strategy_config(config: dict) -> None:
    """Validate config based on the schema, will raise an exception if invalid"""

    # Validates any shared settings
    validate(instance=config, schema=config_schema)

    validate_dependent_params(config)

    use_llm_keyword = config["use_llm"]
    if use_llm_keyword == "true":
        check_llm_specific_parameters(config)

    num_of_clients = config["num_of_clients"]
    num_fit_clients = config["min_fit_clients"]
    num_evaluate_clients = config["min_evaluate_clients"]
    num_available_clients = config["min_available_clients"]

    # Check if the number of clients is enough for the simulation parameters
    if num_fit_clients > num_of_clients or num_evaluate_clients > num_of_clients or num_available_clients > num_of_clients:
        raise ValidationError(
            "Number of clients for fit, evaluate or available cannot be greater than total number of clients"
        )