"""
Shared fixtures for dataset loader tests.

Provides common configuration and setup fixtures for testing dataset loaders.
"""

from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def text_classification_loader_config() -> Dict[str, Any]:
    """Default configuration for TextClassificationLoader tests."""
    return {
        "dataset_name": "stanfordnlp/sst2",
        "tokenizer_model": "distilbert-base-uncased",
        "num_of_clients": 3,
        "batch_size": 8,
        "training_subset_fraction": 1.0,
        "max_seq_length": 128,
        "text_column": "text",
        "text2_column": None,
        "label_column": "label",
        "partitioning_strategy": "iid",
        "partitioning_params": None,
    }


@pytest.fixture
def medquad_temp_dataset_dir(tmp_path: Path) -> str:
    """Create temporary MedQuAD-style dataset directory with JSON files."""
    dataset_dir = tmp_path / "medquad_dataset"
    dataset_dir.mkdir()

    for i in range(3):
        client_dir = dataset_dir / f"client_{i}"
        client_dir.mkdir()
        json_file = client_dir / f"data_{i}.json"
        json_file.write_text(
            '{"question": "What is test?", "answer": "This is a test answer"}'
        )

    return str(dataset_dir)


@pytest.fixture
def medquad_loader_config(medquad_temp_dataset_dir: str) -> Dict[str, Any]:
    """Default configuration for MedQuADDatasetLoader tests."""
    return {
        "dataset_dir": medquad_temp_dataset_dir,
        "num_of_clients": 3,
        "training_subset_fraction": 0.8,
        "model_name": "bert-base-uncased",
        "batch_size": 16,
        "chunk_size": 256,
        "mlm_probability": 0.15,
        "num_poisoned_clients": 1,
    }


@pytest.fixture
def image_dataset_loader_config() -> Dict[str, Any]:
    """Default configuration for ImageDatasetLoader tests."""
    return {
        "dataset_keyword": "bloodmnist",
        "num_of_clients": 5,
        "batch_size": 32,
        "training_subset_fraction": 0.8,
        "num_poisoned_clients": 0,
    }


@pytest.fixture
def federated_dataset_loader_config(temp_dataset_dir: Path) -> Dict[str, Any]:
    """Default configuration for FederatedDatasetLoader tests."""
    return {
        "dataset_dir": str(temp_dataset_dir),
        "num_of_clients": 3,
        "batch_size": 16,
        "training_subset_fraction": 0.8,
    }
