"""
Test fixtures and utilities for federated learning simulation testing.
"""

from .mock_datasets import (
    MockDataset,
    MockDatasetHandler,
    MockFederatedDataset,
    generate_byzantine_client_parameters,
    generate_mock_client_metrics,
    generate_mock_client_parameters,
)
from .sample_models import (
    MockCNNNetwork,
    MockFlowerClient,
    MockNetwork,
    MockNetworkFactory,
    create_mock_aggregated_parameters,
    create_mock_client_models,
    generate_mock_model_parameters,
)

__all__ = [
    # Mock datasets
    "MockDataset",
    "MockFederatedDataset",
    "MockDatasetHandler",
    "generate_mock_client_parameters",
    "generate_mock_client_metrics",
    "generate_byzantine_client_parameters",
    # Mock models
    "MockNetwork",
    "MockCNNNetwork",
    "MockFlowerClient",
    "MockNetworkFactory",
    "create_mock_client_models",
    "generate_mock_model_parameters",
    "create_mock_aggregated_parameters",
]
