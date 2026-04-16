import logging
import random

from jsonschema import validate, ValidationError

config_schema = {
    "type": "object",
    "properties": {
        # Common parameters
        "aggregation_strategy_keyword": {
            "type": "string",
            "enum": ["trust", "pid", "pid_scaled", "pid_standardized", "pid_standardized_score_based",
                    "multi-krum", "krum", "multi-krum-based", "trimmed_mean",
                    "rfa", "bulyan", "fools_gold", "pid_made",  "freq_fed", "fl_detector",
                    "pid_v5_stateless_mean_based", "pid_v5_stateless_median_based", "fed_nga"]
        },
        "strict_mode": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "remove_clients": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "dataset_keyword": {
            "type": "string",
            "enum": [
                "femnist_iid",
                "femnist_niid",
                "its",
                "pneumoniamnist",
                "flair",
                "bloodmnist",
                "medquad",
                "medmentions",
                "lung_photos",
                "breastmnist",
                "pathmnist",
                "dermamnist",
                "octmnist",
                "retinamnist",
                "tissuemnist",
                "organamnist",
                "organcmnist",
                "organsmnist",
                "cifar100_a01",
                "cifar100_a05",
                "cifar100_a20",
                "air_quality_uci",
                "smores",
            ]
        },
        "model_type": {
            "type": "string",
            "enum": ["cnn", "transformer", "linear_regression", "lstm"]
        },
        "num_of_rounds": {
            "type": "integer"
        },
        "num_of_clients": {
            "type": "integer"
        },
        # Pre-training poisoning fields (And all backwards compatible attacks)
        "num_of_malicious_clients": {
            "type": "integer",
            "minimum": 0
        },
        "attack_type": {
            "type": "string",
            "enum": ["label_flipping", "gaussian_noise", "ner_label_flipping", "target_flipping",
                     "gradient_scaling", "little_is_enough",]
        },
        "attack_ratio": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "target_noise_snr": {
            "type": "number"
        },
        "scale_factor": {           # for gradient scaling
            "type": "number", 
            "exclusiveMinimum": 0
        },
        # Visualization and dataset fields
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
            "enum": ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "gpt2"]
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
            "enum": ["cpu", "gpu"]
        },
        "cpus_per_client": {
            "type": "number"
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

        # Dataset specific parameters

        # Regression Tasks Dataset Parameters
        "target_name": {
            "type": "string",
            "enum": [
                        "CO", "C6H6", "NOx", "NO2", # AirQualityUCI
                        "DO",                       # SMORES
                    ]
        },
        
        # SMORES
        "sequence_len": {
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

        # Bulyan
        "num_bulyan_benign_clients": {
            "type": "integer"
        },

        # Trimmed mean
        "trim_ratio": {
            "type": "number"
        },

        # FreqFed
        "min_cluster_size": {
            "type": "integer",
            "minimum": 2
        },

        # PID-MADE
        "lam": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "exclusiveMaximum": 1.0
        },
        "threshold_mode": {
            "type": "string",
            "enum": ["fixed", "chebyshev", "gaussian"]
        },
        "tau": {
            "type": "number",
            "exclusiveMinimum": 0.0
        },
        "alpha": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "exclusiveMaximum": 1.0
        },

        # FLDetector
        "fl_detector_history_size": {
            "type": "integer",
            "minimum": 1
        },
        "fl_detector_z_score_threshold": {
            "type": "number",
            "exclusiveMinimum": 0.0
        },

        # Attack scheduling
        "attack_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_round": {"type": "integer", "minimum": 1},
                    "end_round": {"type": "integer", "minimum": 1},
                    "attack_type": {
                        "type": "string",
                        "enum": ["label_flipping", "gaussian_noise", "target_flipping", 
                                 "gradient_scaling", "little_is_enough"]
                    },
                    # old format entries for backward-compatibility - do not use
                    "selection_strategy": {"type": "string", "enum": ["specific", "random", "percentage"]},
                    "malicious_client_ids": {"type": "array", "items": {"type": "integer"}},
                    "malicious_client_count": {"type": "integer", "minimum": 1},
                    "malicious_percentage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    # end of old format entries
                    "num_malicious_clients": {"type": "integer", "items": {"type": "integer"}},
                    "specific_malicious_clients": {"type": "array", "items": {"type": "integer"}},
                    "proportion_malicious_clients": {"type": "number", "items": {"type": "integer"}},
                    "random_seed": {"type": "integer", "minimum": 0},
                    "target_noise_snr": {"type": "number"},
                    "attack_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "scale_factor": {"type": "number", "exclusiveMinimum": 0}
                },
                "required": ["start_round", "end_round", "attack_type"]
            }
        },

        # Attack snapshot saving
        "save_attack_snapshots": {
            "type": "string",
            "enum": ["true", "false"]
        },
        "attack_snapshot_format": {
            "type": "string",
            "enum": ["pickle", "visual", "pickle_and_visual"]
        },
        "snapshot_max_samples": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50
        },

        "config_is_ai_generated": {
            "type": "string"
        },

        "user_prompt_for_ai_generated": {
            "type": "string"
        },

        "ai_agent_config_reasoning": {
            "type": "string"
        },

        "ai_configuration_error": {
            "type": "string"
        },

        "num_ai_config_corrections": {
            "type": "number"
        }

    },
    "required": [
        "aggregation_strategy_keyword", "remove_clients", "dataset_keyword", "model_type",
        "use_llm", "num_of_rounds", "num_of_clients",
        "show_plots", "save_plots", "save_csv", "preserve_dataset",
        "training_subset_fraction", "training_device", "cpus_per_client",
        "gpus_per_client", "min_fit_clients", "min_evaluate_clients",
        "min_available_clients", "evaluate_metrics_aggregation_fn",
        "num_of_client_epochs", "batch_size"
    ]
}


def _validate_dependent_params(strategy_config: dict) -> None:
    """Validate that all params that require additional params are correct."""

    # Aggregation Strategy Parameters

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
    elif aggregation_strategy_keyword in ("pid", "pid_scaled", "pid_standardized", "pid_standardized_score_based", 
                                          "pid_v5_stateless_mean_based", "pid_v5_stateless_median_based"):
        pid_specific_parameters = [
            "num_std_dev", "Kp", "Ki", "Kd", "begin_removing_from_round"
        ]
        for param in pid_specific_parameters:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing parameter {param} for PID aggregation {aggregation_strategy_keyword}"
                )
    elif aggregation_strategy_keyword == "pid_made":
        pid_made_required = ["Kp", "Ki", "Kd", "lam", "threshold_mode", "begin_removing_from_round"]
        for param in pid_made_required:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing parameter '{param}' for pid_made aggregation"
                )
        threshold_mode = strategy_config["threshold_mode"]
        if threshold_mode == "fixed":
            if "tau" not in strategy_config:
                raise ValidationError(
                    "pid_made with threshold_mode='fixed' requires 'tau' parameter"
                )
        elif threshold_mode in ("chebyshev", "gaussian"):
            if "alpha" not in strategy_config:
                raise ValidationError(
                    f"pid_made with threshold_mode='{threshold_mode}' requires 'alpha' parameter"
                )
        else:
            raise ValidationError(
                f"pid_made: invalid threshold_mode '{threshold_mode}'. Must be 'fixed', 'chebyshev', or 'gaussian'"
            )
    elif aggregation_strategy_keyword in ["multi-krum", "krum", "multi-krum-based"]:
        if "num_krum_selections" not in strategy_config:
            raise ValidationError(
                f"Missing parameter num_krum_selections for Krum-based aggregation {aggregation_strategy_keyword}"
            )
    elif aggregation_strategy_keyword == "trimmed_mean":
        if "trim_ratio" not in strategy_config or "begin_removing_from_round" not in strategy_config:
            raise ValidationError(
                f"Missing parameter trim_ratio for trimmed mean aggregation {aggregation_strategy_keyword}"
            )
    elif aggregation_strategy_keyword == "bulyan":
        if "num_bulyan_benign_clients" not in strategy_config:
            raise ValidationError(
                f"Missing parameter num_bulyan_benign_clients for aggregation: {aggregation_strategy_keyword}"
            )
    elif aggregation_strategy_keyword == "freq_fed":
        if "begin_removing_from_round" not in strategy_config:
            raise ValidationError(
                "Missing parameter 'begin_removing_from_round' for freq_fed aggregation"
            )
    elif aggregation_strategy_keyword == "fl_detector":
        fl_detector_required = [
            "fl_detector_history_size",
            "fl_detector_z_score_threshold",
            "begin_removing_from_round",
        ]
        for param in fl_detector_required:
            if param not in strategy_config:
                raise ValidationError(
                    f"Missing parameter '{param}' for fl_detector aggregation"
                )

    if aggregation_strategy_keyword in ("bulyan", "krum", "multi-krum-based", "multi-krum", "rfa", "trimmed_mean", "pid"):
        if "begin_removing_from_round" not in strategy_config:
            raise ValidationError(
                f"Missing parameter begin_removing_from_round for aggregation {aggregation_strategy_keyword}"
            )
        
    
    # Attack Parameters (for old attack_type backwards copatibility)

    if "attack_type" in strategy_config:
        attack_type = strategy_config["attack_type"]
        if attack_type == "gaussian_noise":
            if "target_noise_snr" not in strategy_config:
                raise ValidationError(
                    f"Missing parameter 'target_noise_snr' for attack '{attack_type}'"
                )

        elif attack_type == "gradient_scaling":
            if "scale_factor" not in strategy_config:
                raise ValidationError(
                    f"Missing parameter 'scale_factor' for attack '{attack_type}'"
                )
            if strategy_config["scale_factor"] <= 0:
                raise ValidationError(
                    f"Parameter 'scale_factor' in attack '{attack_type}' must be > 0"
                )

    # Dataset Parameters
        
    dataset_keyword = strategy_config["dataset_keyword"]

    if dataset_keyword == "air_quality_uci":
        if "target_name" not in strategy_config:
            raise ValidationError(
                f"Missing parameter target_name for UCI Air Quality dataset {dataset_keyword}"
            )
        elif ("target_name" in strategy_config
              and strategy_config.get("target_name","") not in ["CO", "C6H6", "NOx", "NO2"]):
            raise ValidationError(
                f"For UCI Air Quality dataset {dataset_keyword}, target_name  must be one of ['CO', 'C6H6', 'NOx', 'NO2']"
            ) 
    elif dataset_keyword == "smores":
        if "target_name" not in strategy_config:
            raise ValidationError(
                f"Missing parameter target_name for SMORES dataset {dataset_keyword}"
            )
        elif ("target" in strategy_config
              and strategy_config.get("target_name","") not in ["DO"]):
            raise ValidationError(
                f"For SMORES dataset {dataset_keyword}, target_name must be one of ['DO']"
            )

        if "sequence_len" not in strategy_config:
            raise ValidationError(
                f"Missing parameter sequence_len for SMORES dataset {dataset_keyword}"
            )
        
        if "attack_type" in strategy_config and strategy_config["attack_type"] == "target_flipping":
            raise ValidationError(
                f"Static Target Flipping is not supported for SMORES dataset {dataset_keyword}.\n"
                        f"Please use 'attack_schedule' instead of 'attack_type'."
            )
        
    

def _populate_client_selection(config: dict) -> None:
    """
    Populate malicious_client_ids for random and percentage selection strategies.

    For reproducibility, uses deterministic seeding:
    - If attack entry has 'random_seed': use it
    - Otherwise: auto-generate seed from attack parameters

    Research-backed approach: Fixed client selection per attack window
    (not re-randomized each round) for reproducibility.
    """

    schedule = config.get("attack_schedule", [])
    num_of_clients = config.get("num_of_clients")

    for idx, entry in enumerate(schedule):
        num_malicious_clients = entry.get("num_malicious_clients", entry.get("malicious_client_count", None))
        proportion_malicious_clients = entry.get("proportion_malicious_clients", entry.get("malicious_percentage", None))
        specific_malicious_clients = entry.get("specific_malicious_clients", entry.get("malicious_client_ids", None))

        if specific_malicious_clients:
            entry["malicious_client_ids"] = specific_malicious_clients
            continue

        if "random_seed" in entry:
            seed = entry["random_seed"]
        else:
            # Auto-generate seed from attack parameters for reproducibility
            seed_string = f"{entry.get('attack_type')}_{entry.get('start_round')}_{entry.get('end_round')}_{idx}"
            seed = hash(seed_string) % (2 ** 32)
            random.seed(seed)

        if proportion_malicious_clients:
            num_malicious_clients = int(num_of_clients * proportion_malicious_clients)

        if specific_malicious_clients:
            num_malicious_clients = len(specific_malicious_clients)

        if num_malicious_clients > num_of_clients:
            raise ValidationError(
                f"attack_schedule entry {idx}: Cannot select {num_malicious_clients} clients "
                f"when only {num_of_clients} clients exist"
            )

        if num_malicious_clients <= 0:
            raise ValidationError(
                    f"attack_schedule entry {idx}: Must select at least 1 malicious client "
                    f"(got {num_malicious_clients})"
            )

        if specific_malicious_clients:
            entry["malicious_client_ids"] = specific_malicious_clients

        else:
            all_client_ids = list(range(num_of_clients))
            selected_clients = sorted(random.sample(all_client_ids, num_malicious_clients))

            entry["malicious_client_ids"] = selected_clients

        logging.debug(
                f"Selected clients {selected_clients} for {entry.get('attack_type')} attack "
                f"(seed={seed})"
        )


def _validate_attack_schedule(config: dict) -> None:
    """
    Validate attack_schedule entries and enforce related constraints.

    Ensures each schedule entry has:
    - Valid round ranges
    - Required attack-type-specific parameters
    - Proper selection strategy configuration
    - Preserve_dataset set to false
    """
    schedule = config.get("attack_schedule", [])

    for idx, entry in enumerate(schedule):
        entry_desc = f"attack_schedule entry {idx}"

        # Validate round range
        start_round = entry.get("start_round")
        end_round = entry.get("end_round")
        if start_round > end_round:
            raise ValidationError(
                f"{entry_desc}: start_round ({start_round}) cannot be greater than end_round ({end_round})"
            )

        # Validate attack type and its required parameters
        num_of_rounds = config.get("num_of_rounds")
        if end_round > num_of_rounds:
            raise ValidationError(
                f"{entry_desc}: end_round ({end_round}) exceeds num_of_rounds ({num_of_rounds})"
            )


        attack_type = entry.get("attack_type")

        if attack_type == "gaussian_noise":
            if "target_noise_snr" not in entry:
                raise ValidationError(
                    f"{entry_desc}: gaussian_noise attack requires 'target_noise_snr' parameter"
                )
            if "attack_ratio" not in entry:
                raise ValidationError(
                    f"{entry_desc}: gaussian_noise attack requires 'attack_ratio' parameter"
                )
        elif attack_type == "target_flipping":
            if "attack_ratio" not in entry:
                raise ValidationError(
                    f"{entry_desc}: target_flipping attack requires 'attack_ratio' parameter"
                )

        elif attack_type == "gradient_scaling":
            if "scale_factor" not in entry:
                raise ValidationError(
                    f"{entry_desc}: gradient_scaling attack requires 'scale_factor' parameter"
                )
            if entry["scale_factor"] <= 0:
                raise ValidationError(
                    f"{entry_desc}: 'scale_factor' must be > 0"
                )

    # Check for overlapping rounds (attack stacking)
    for i, entry1 in enumerate(schedule):
        for j, entry2 in enumerate(schedule[i + 1:], start=i + 1):
            if not (
                    entry1["end_round"] < entry2["start_round"] or
                    entry2["end_round"] < entry1["start_round"]
            ):
                # Check if same attack type
                if entry1.get("attack_type") == entry2.get("attack_type"):
                    raise ValidationError(
                        f"CONFIG REJECTED: Overlapping rounds with same attack type\n"
                        f"\n"
                        f"Conflict:\n"
                        f"  - Entry {i}: {entry1.get('attack_type')} (rounds {entry1['start_round']}-{entry1['end_round']})\n"
                        f"  - Entry {j}: {entry2.get('attack_type')} (rounds {entry2['start_round']}-{entry2['end_round']})\n"
                        f"\n"
                        f"Reason:\n"
                        f"  - Cannot have the same attack type active in overlapping rounds\n"
                        f"  - Config is the single source of truth - it must be unambiguous\n"
                        f"\n"
                        f"Fix:\n"
                        f"  - Adjust round ranges to not overlap for same attack type\n"
                        f"  - Or use different attack types if you want stacked attacks\n"
                    )
                else:
                    logging.info(
                        f"attack_schedule entries {i} and {j} have overlapping rounds with different attack types "
                        f"({entry1.get('attack_type')} and {entry2.get('attack_type')}). "
                        f"Both attacks will be stacked and applied sequentially."
                    )

    # Strict rejection: preserve_dataset incompatible with data-poisoning attack_schedule
    # (gradient_scaling and little_is_enough is post-training and does not touch the dataset, so it is exempt)
    data_poisoning_entries = [e for e in schedule if e.get("attack_type") not in ["gradient_scaling", "little_is_enough"]]
    if data_poisoning_entries and config.get("preserve_dataset") == "true":
        raise ValidationError(
            "CONFIG REJECTED: Cannot use attack_schedule with preserve_dataset=true\n"
            "\n"
            "Reason:\n"
            "  - attack_schedule applies attacks in-memory during training rounds\n"
            "  - preserve_dataset=true expects pre-poisoned filesystem dataset\n"
            "  - These modes are incompatible\n"
            "\n"
            "Fix:\n"
            "  Set 'preserve_dataset': 'false' in your config file\n"
            "  Use 'save_attack_snapshots': 'true' to inspect poisoned data\n"
            "\n"
            "Config is the single source of truth - it must reflect intended execution."
        )


def _validate_llm_parameters(strategy_config: dict) -> None:
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


def _apply_strict_mode(config: dict) -> None:
    """Handle strict_mode validation and client configuration logic."""

    # set strict mode if AI generated, try-catch for legacy configs
    try:
        if config["config_is_ai_generated"] == "true":
            config["strict_mode"] = "true"

    except Exception as e: # noqa
        pass

    # Set strict_mode to "true" by default if not specified
    if "strict_mode" not in config:
        config["strict_mode"] = "true"

    strict_mode = config["strict_mode"] == "true"
    num_of_clients = config["num_of_clients"]
    num_fit_clients = config["min_fit_clients"]
    num_evaluate_clients = config["min_evaluate_clients"]
    num_available_clients = config["min_available_clients"]

    # Always check: min_* cannot be greater than num_of_clients
    if (
            num_fit_clients > num_of_clients or
            num_evaluate_clients > num_of_clients or
            num_available_clients > num_of_clients
    ):
        raise ValidationError(
            f"EXPERIMENT STOPPED: Client configuration error.\n"
            f"Cannot require more clients than available:\n"
            f"  - Total clients: {num_of_clients}\n"
            f"  - min_fit_clients: {num_fit_clients}\n"
            f"  - min_evaluate_clients: {num_evaluate_clients}\n"
            f"  - min_available_clients: {num_available_clients}\n"
            f"Please ensure all min_* values are <= {num_of_clients}"
        )

    # If strict_mode is enabled, reject configs where min_* != num_of_clients
    if strict_mode:
        if (
                num_fit_clients != num_of_clients or
                num_evaluate_clients != num_of_clients or
                num_available_clients != num_of_clients
        ):
            raise ValidationError(
                f"CONFIG REJECTED: strict_mode requires all clients to participate.\n"
                f"\n"
                f"Current values:\n"
                f"  - num_of_clients: {num_of_clients}\n"
                f"  - min_fit_clients: {num_fit_clients}\n"
                f"  - min_evaluate_clients: {num_evaluate_clients}\n"
                f"  - min_available_clients: {num_available_clients}\n"
                f"\n"
                f"Fix:\n"
                f"  Set all min_* values equal to num_of_clients ({num_of_clients})\n"
                f"  Or set 'strict_mode': 'false' to allow partial participation\n"
            )


def validate_strategy_config(config: dict) -> None:
    """Validate config based on the schema, will raise an exception if invalid"""

    # Validates config structure, types, and basic constraints against JSON schema
    validate(instance=config, schema=config_schema)

    _validate_dependent_params(config)

    if config["use_llm"] == "true":
        _validate_llm_parameters(config)

    _validate_attack_schedule(config)
    _populate_client_selection(config)
    _apply_strict_mode(config)
