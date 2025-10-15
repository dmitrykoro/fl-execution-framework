"""Shared fixtures for config validation tests."""

import pytest


@pytest.fixture
def base_valid_config():
    """Base configuration with all common required fields."""
    return {
        "remove_clients": "true",
        "dataset_keyword": "femnist_iid",
        "dataset_source": "local",
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
