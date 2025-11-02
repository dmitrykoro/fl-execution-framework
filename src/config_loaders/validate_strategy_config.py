import logging
from jsonschema import validate, ValidationError

config_schema = {
    "type": "object",
    "properties": {
        # Common parameters
        "aggregation_strategy_keyword": {
            "type": "string",
            "enum": ["trust", "pid", "pid_scaled", "pid_standardized", "pid_standardized_score_based",
                     "multi-krum", "krum", "multi-krum-based", "trimmed_mean",
                     "rfa", "bulyan", "fedavg"]
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
                "financial_phrasebank",
                "lexglue",
                "lung_photos",
                "breastmnist",
                "pathmnist",
                "dermamnist",
                "octmnist",
                "retinamnist",
                "tissuemnist",
                "organamnist",
                "organcmnist",
                "organsmnist"
            ]
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
            "enum": [
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                "distilbert-base-uncased"
            ]
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
                        "enum": ["label_flipping", "gaussian_noise", "brightness", "token_replacement"]
                    },
                    "selection_strategy": {"type": "string", "enum": ["specific", "random", "percentage"]},
                    "malicious_client_ids": {"type": "array", "items": {"type": "integer"}},
                    "malicious_client_count": {"type": "integer", "minimum": 1},
                    "malicious_percentage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "random_seed": {"type": "integer", "minimum": 0},
                    "flip_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "target_noise_snr": {"type": "number"},
                    "attack_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["start_round", "end_round", "attack_type", "selection_strategy"]
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
        }
    },
    "required": [
        "aggregation_strategy_keyword", "remove_clients", "dataset_keyword", "model_type",
        "use_llm", "num_of_rounds", "num_of_clients", "num_of_malicious_clients",
        "show_plots", "save_plots", "save_csv", "preserve_dataset",
        "training_subset_fraction", "training_device", "cpus_per_client",
        "gpus_per_client", "min_fit_clients", "min_evaluate_clients",
        "min_available_clients", "evaluate_metrics_aggregation_fn",
        "num_of_client_epochs", "batch_size", "attack_schedule"
    ]
}


def _validate_dependent_params(strategy_config: dict) -> None:
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
    elif aggregation_strategy_keyword in ("pid", "pid_scaled", "pid_standardized", "pid_standardized_score_based"):
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


def _populate_client_selection(config: dict) -> None:
    """
    Populate _selected_clients for random and percentage selection strategies.

    For reproducibility, uses deterministic seeding:
    - If attack entry has 'random_seed': use it
    - Otherwise: auto-generate seed from attack parameters

    Research-backed approach: Fixed client selection per attack window
    (not re-randomized each round) for reproducibility.
    """
    import random

    schedule = config.get("attack_schedule", [])
    num_of_clients = config.get("num_of_clients")

    for idx, entry in enumerate(schedule):
        selection_strategy = entry.get("selection_strategy")

        if selection_strategy == "specific":
            # Already has explicit malicious_client_ids, nothing to do
            continue

        elif selection_strategy in ("random", "percentage"):
            if "random_seed" in entry:
                seed = entry["random_seed"]
            else:
                # Auto-generate seed from attack parameters for reproducibility
                seed_string = f"{entry.get('attack_type')}_{entry.get('start_round')}_{entry.get('end_round')}_{idx}"
                seed = hash(seed_string) % (2 ** 32)

            random.seed(seed)

            if selection_strategy == "random":
                num_malicious = entry.get("malicious_client_count")
            else:
                malicious_percentage = entry.get("malicious_percentage")
                num_malicious = int(num_of_clients * malicious_percentage)

            if num_malicious > num_of_clients:
                raise ValidationError(
                    f"attack_schedule entry {idx}: Cannot select {num_malicious} clients "
                    f"when only {num_of_clients} clients exist"
                )

            if num_malicious <= 0:
                raise ValidationError(
                    f"attack_schedule entry {idx}: Must select at least 1 malicious client "
                    f"(got {num_malicious})"
                )

            all_client_ids = list(range(num_of_clients))
            selected_clients = sorted(random.sample(all_client_ids, num_malicious))

            entry["_selected_clients"] = selected_clients

            logging.debug(
                f"attack_schedule entry {idx} ({selection_strategy}): "
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

        if attack_type == "label_flipping":
            if "flip_fraction" not in entry:
                raise ValidationError(
                    f"{entry_desc}: label_flipping attack requires 'flip_fraction' parameter"
                )

        elif attack_type == "gaussian_noise":
            if "target_noise_snr" not in entry:
                raise ValidationError(
                    f"{entry_desc}: gaussian_noise attack requires 'target_noise_snr' parameter"
                )
            if "attack_ratio" not in entry:
                raise ValidationError(
                    f"{entry_desc}: gaussian_noise attack requires 'attack_ratio' parameter"
                )

        # Validate selection strategy requirements
        selection_strategy = entry.get("selection_strategy")

        if selection_strategy == "specific":
            if "malicious_client_ids" not in entry:
                raise ValidationError(
                    f"{entry_desc}: 'specific' selection strategy requires 'malicious_client_ids' list"
                )

        elif selection_strategy == "random":
            if "malicious_client_count" not in entry:
                raise ValidationError(
                    f"{entry_desc}: 'random' selection strategy requires 'malicious_client_count' integer"
                )

        elif selection_strategy == "percentage":
            if "malicious_percentage" not in entry:
                raise ValidationError(
                    f"{entry_desc}: 'percentage' selection strategy requires 'malicious_percentage' (0.0-1.0)"
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

    # Strict rejection: preserve_dataset incompatible with attack_schedule
    if schedule and config.get("preserve_dataset") == "true":
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

    # If strict_mode is enabled, force all min_* = num_of_clients
    if strict_mode:
        if (
                num_fit_clients != num_of_clients or
                num_evaluate_clients != num_of_clients or
                num_available_clients != num_of_clients
        ):
            # Force all to equal total clients
            config["min_fit_clients"] = num_of_clients
            config["min_evaluate_clients"] = num_of_clients
            config["min_available_clients"] = num_of_clients

            logging.info(f"STRICT MODE ENABLED: Auto-configured client participation")
            logging.info(f"  - Set min_fit_clients = {num_of_clients}")
            logging.info(f"  - Set min_evaluate_clients = {num_of_clients}")
            logging.info(f"  - Set min_available_clients = {num_of_clients}")
            logging.info(f"  - This ensures all clients participate in every round")


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
