"""
Shared fixtures for simulation strategy tests.

Provides common fixtures used across all strategy test files.
"""

from typing import Any, Callable, List, Tuple
from unittest.mock import patch

import pytest
from flwr.common import EvaluateRes

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from tests.common import (
    ClientProxy,
    Mock,
    generate_mock_client_data,
    np,
)


# =============================================================================
# CLIENT RESULT FIXTURES
# =============================================================================


@pytest.fixture
def mock_client_results() -> List[Tuple[Any, Any]]:
    """Generate mock client results for 5 clients with varied parameters."""
    return generate_mock_client_data(num_clients=5)


@pytest.fixture(scope="session")
def mock_client_results_factory() -> Callable[[int], List[Tuple[Any, Any]]]:
    """Factory for creating mock client results with custom client count."""

    def _create_results(num_clients: int) -> List[Tuple[Any, Any]]:
        return generate_mock_client_data(num_clients=num_clients)

    return _create_results


# =============================================================================
# EVALUATE RESULT FIXTURES
# =============================================================================


@pytest.fixture
def mock_evaluate_results() -> List[Tuple[Any, EvaluateRes]]:
    """Generate mock evaluate results for 5 clients with progressive accuracy/loss values."""
    results = []
    for i in range(5):
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = str(i)
        eval_res = Mock(spec=EvaluateRes)
        eval_res.num_examples = 100 + (i * 10)
        eval_res.loss = 0.5 - (i * 0.1)
        eval_res.metrics = {"accuracy": 0.8 + (i * 0.02)}
        results.append((client_proxy, eval_res))
    return results


@pytest.fixture(scope="session")
def mock_evaluate_results_factory() -> Callable:
    """Factory for creating custom mock evaluate results."""

    def _create_results(
        num_clients: int = 5,
        base_accuracy: float = 0.8,
        base_loss: float = 0.5,
        accuracy_increment: float = 0.02,
        loss_decrement: float = 0.1,
    ) -> List[Tuple[Any, EvaluateRes]]:
        results = []
        for i in range(num_clients):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            eval_res = Mock(spec=EvaluateRes)
            eval_res.num_examples = 100 + (i * 10)
            eval_res.loss = base_loss - (i * loss_decrement)
            eval_res.metrics = {"accuracy": base_accuracy + (i * accuracy_increment)}
            results.append((client_proxy, eval_res))
        return results

    return _create_results


# =============================================================================
# STRATEGY HISTORY FIXTURE
# =============================================================================


@pytest.fixture
def mock_strategy_history() -> Mock:
    """Create mock SimulationStrategyHistory for testing."""
    return Mock(spec=SimulationStrategyHistory)


# =============================================================================
# METRICS AGGREGATION FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def krum_fit_metrics_fn() -> Callable:
    """Provide fit_metrics_aggregation_fn for Krum-based strategies."""
    return lambda x: x


@pytest.fixture(scope="session")
def fit_metrics_aggregation_fn() -> Callable:
    """General fit metrics aggregation function for strategy testing."""
    return lambda x: x


# =============================================================================
# CLUSTERING MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_clustering_components():
    """Mock clustering components (KMeans, MinMaxScaler) for strategy testing."""
    mock_kmeans_instance = Mock()
    mock_kmeans_instance.transform.return_value = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.5]]
    )

    mock_scaler_instance = Mock()
    mock_scaler_instance.transform.return_value = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.5]]
    )

    return {
        "kmeans_instance": mock_kmeans_instance,
        "scaler_instance": mock_scaler_instance,
    }


@pytest.fixture
def mock_krum_clustering():
    """Fixture for Krum strategy clustering mocks."""
    with (
        patch(
            "src.simulation_strategies.krum_based_removal_strategy.KMeans"
        ) as mock_kmeans,
        patch(
            "src.simulation_strategies.krum_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler,
        patch("flwr.server.strategy.Krum.aggregate_fit") as mock_parent_aggregate,
    ):
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.array(
            [[0.1], [0.2], [0.3], [0.4], [0.5]]
        )
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.array(
            [[0.1], [0.2], [0.3], [0.4], [0.5]]
        )
        mock_scaler.return_value = mock_scaler_instance

        mock_parent_aggregate.return_value = (Mock(), {})

        yield {
            "kmeans": mock_kmeans,
            "scaler": mock_scaler,
            "parent_aggregate": mock_parent_aggregate,
            "kmeans_instance": mock_kmeans_instance,
            "scaler_instance": mock_scaler_instance,
        }


@pytest.fixture
def mock_multi_krum_clustering():
    """Fixture for Multi-Krum strategy clustering mocks."""
    with (
        patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.KMeans"
        ) as mock_kmeans,
        patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler,
        patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
    ):
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.array(
            [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
        )
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.array(
            [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
        )
        mock_scaler.return_value = mock_scaler_instance

        mock_parent_aggregate.return_value = (Mock(), {})

        yield {
            "kmeans": mock_kmeans,
            "scaler": mock_scaler,
            "parent_aggregate": mock_parent_aggregate,
            "kmeans_instance": mock_kmeans_instance,
            "scaler_instance": mock_scaler_instance,
        }


# =============================================================================
# NETWORK MODEL FIXTURE
# =============================================================================


@pytest.fixture
def mock_network_model() -> Mock:
    """Create mock network model for PID and other strategies requiring network models."""
    return Mock()
