"""
Tests for mock Flower FL components to ensure behavior consistency with real Flower components.

Tests verify that mock implementations behave correctly and provide consistent
results for federated learning simulation testing.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.fixtures.mock_flower_components import (
    MockClient,
    MockClientProxy,
    MockEvaluateRes,
    MockFitRes,
    MockNumPyClient,
    MockParameters,
    MockServerConfig,
    create_mock_client_proxies,
    create_mock_evaluate_results,
    create_mock_fit_results,
    create_mock_flower_client,
    mock_ndarrays_to_parameters,
    mock_parameters_to_ndarrays,
    mock_start_simulation,
    mock_weighted_loss_avg,
)


class TestMockParameters:
    """Test MockParameters class behavior."""

    def test_parameters_initialization(self):
        """Test MockParameters initialization with tensors."""
        tensors = [np.array([1, 2, 3]).tobytes(), np.array([4, 5]).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        assert params.tensors == tensors
        assert params.tensor_type == "numpy.ndarray"

    def test_parameters_equality(self):
        """Test MockParameters equality comparison."""
        tensors1 = [np.array([1, 2, 3]).tobytes()]
        tensors2 = [np.array([1, 2, 3]).tobytes()]
        tensors3 = [np.array([4, 5, 6]).tobytes()]

        params1 = MockParameters(tensors1, "numpy.ndarray")
        params2 = MockParameters(tensors2, "numpy.ndarray")
        params3 = MockParameters(tensors3, "numpy.ndarray")

        assert params1 == params2
        assert params1 != params3
        assert params1 != "not_parameters"


class TestMockFitRes:
    """Test MockFitRes class behavior."""

    def test_fit_res_initialization(self):
        """Test MockFitRes initialization with required fields."""
        tensors = [np.array([1, 2, 3]).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")
        metrics = {"loss": 0.5, "accuracy": 0.8}

        fit_res = MockFitRes(params, 100, metrics)

        assert fit_res.parameters == params
        assert fit_res.num_examples == 100
        assert fit_res.metrics == metrics

    def test_fit_res_default_metrics(self):
        """Test MockFitRes with default empty metrics."""
        tensors = [np.array([1, 2, 3]).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        fit_res = MockFitRes(params, 50)

        assert fit_res.metrics == {}


class TestMockEvaluateRes:
    """Test MockEvaluateRes class behavior."""

    def test_evaluate_res_initialization(self):
        """Test MockEvaluateRes initialization with required fields."""
        metrics = {"accuracy": 0.9, "f1_score": 0.85}

        eval_res = MockEvaluateRes(0.3, 75, metrics)

        assert eval_res.loss == 0.3
        assert eval_res.num_examples == 75
        assert eval_res.metrics == metrics

    def test_evaluate_res_default_metrics(self):
        """Test MockEvaluateRes with default empty metrics."""
        eval_res = MockEvaluateRes(0.4, 60)

        assert eval_res.metrics == {}


class TestMockClientProxy:
    """Test MockClientProxy class behavior."""

    def test_client_proxy_initialization(self):
        """Test MockClientProxy initialization."""
        proxy = MockClientProxy("client_1")

        assert proxy.cid == "client_1"
        assert proxy.client_fn is None
        assert proxy._training_rounds == 0

    def test_client_proxy_fit(self):
        """Test MockClientProxy fit method."""
        proxy = MockClientProxy("0")

        # Create mock parameters
        original_tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensors = [original_tensor.tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        config = {"epochs": 1}
        fit_res = proxy.fit(params, config)

        # Verify result structure
        assert isinstance(fit_res, MockFitRes)
        assert isinstance(fit_res.parameters, MockParameters)
        assert fit_res.num_examples > 0
        assert "loss" in fit_res.metrics
        assert "accuracy" in fit_res.metrics
        assert fit_res.metrics["round"] == 1

        # Verify parameters were updated (should be different due to noise)
        updated_tensor = np.frombuffer(fit_res.parameters.tensors[0], dtype=np.float32)
        assert not np.array_equal(original_tensor, updated_tensor)

    def test_client_proxy_evaluate(self):
        """Test MockClientProxy evaluate method."""
        proxy = MockClientProxy("0")

        # Create mock parameters
        tensors = [np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        config = {}
        eval_res = proxy.evaluate(params, config)

        # Verify result structure
        assert isinstance(eval_res, MockEvaluateRes)
        assert 0.1 <= eval_res.loss <= 1.5
        assert eval_res.num_examples > 0
        assert "accuracy" in eval_res.metrics
        assert "f1_score" in eval_res.metrics

    def test_client_proxy_consistent_behavior(self):
        """Test that MockClientProxy produces consistent results with same seed."""
        proxy1 = MockClientProxy("0")
        proxy2 = MockClientProxy("0")

        tensors = [np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        # Both proxies should produce similar results (same client ID = same seed)
        fit_res1 = proxy1.fit(params, {})
        fit_res2 = proxy2.fit(params, {})

        # Results should be similar (not exactly equal due to training round differences)
        assert abs(fit_res1.metrics["loss"] - fit_res2.metrics["loss"]) < 0.1


class TestMockServerConfig:
    """Test MockServerConfig class behavior."""

    def test_server_config_initialization(self):
        """Test MockServerConfig initialization."""
        config = MockServerConfig(10)

        assert config.num_rounds == 10


class TestMockNumPyClient:
    """Test MockNumPyClient class behavior."""

    def test_numpy_client_initialization(self):
        """Test MockNumPyClient initialization."""
        client = MockNumPyClient(client_id=5)

        assert client.client_id == 5

    def test_numpy_client_get_parameters(self):
        """Test MockNumPyClient get_parameters method."""
        client = MockNumPyClient(client_id=0)

        params = client.get_parameters({})

        assert isinstance(params, list)
        assert len(params) == 2  # Two parameter arrays
        assert all(isinstance(p, np.ndarray) for p in params)
        assert params[0].shape == (100, 10)
        assert params[1].shape == (10,)

    def test_numpy_client_fit(self):
        """Test MockNumPyClient fit method."""
        client = MockNumPyClient(client_id=0)

        # Create input parameters
        input_params = [
            np.random.randn(100, 10).astype(np.float32),
            np.random.randn(10).astype(np.float32),
        ]

        updated_params, num_examples, metrics = client.fit(input_params, {})

        # Verify output structure
        assert isinstance(updated_params, list)
        assert len(updated_params) == len(input_params)
        assert isinstance(num_examples, int)
        assert num_examples > 0
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics

        # Verify parameters were updated
        for orig, updated in zip(input_params, updated_params):
            assert orig.shape == updated.shape
            assert not np.array_equal(orig, updated)  # Should be different due to noise

    def test_numpy_client_evaluate(self):
        """Test MockNumPyClient evaluate method."""
        client = MockNumPyClient(client_id=0)

        # Create input parameters
        input_params = [
            np.random.randn(100, 10).astype(np.float32),
            np.random.randn(10).astype(np.float32),
        ]

        loss, num_examples, metrics = client.evaluate(input_params, {})

        # Verify output structure
        assert isinstance(loss, float)
        assert 0.1 <= loss <= 1.5
        assert isinstance(num_examples, int)
        assert num_examples > 0
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_score" in metrics


class TestMockClient:
    """Test MockClient class behavior."""

    def test_client_initialization(self):
        """Test MockClient initialization."""
        numpy_client = MockNumPyClient(0)
        client = MockClient(numpy_client)

        assert client.numpy_client == numpy_client

    def test_client_fit_delegation(self):
        """Test MockClient fit method delegates to NumPy client."""
        numpy_client = MockNumPyClient(0)
        client = MockClient(numpy_client)

        # Create mock parameters
        tensors = [np.random.randn(10).astype(np.float32).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        fit_res = client.fit(params, {})

        assert isinstance(fit_res, MockFitRes)
        assert isinstance(fit_res.parameters, MockParameters)
        assert fit_res.num_examples > 0
        assert "loss" in fit_res.metrics

    def test_client_evaluate_delegation(self):
        """Test MockClient evaluate method delegates to NumPy client."""
        numpy_client = MockNumPyClient(0)
        client = MockClient(numpy_client)

        # Create mock parameters
        tensors = [np.random.randn(10).astype(np.float32).tobytes()]
        params = MockParameters(tensors, "numpy.ndarray")

        eval_res = client.evaluate(params, {})

        assert isinstance(eval_res, MockEvaluateRes)
        assert 0.1 <= eval_res.loss <= 1.5
        assert eval_res.num_examples > 0


class TestMockStartSimulation:
    """Test mock_start_simulation function behavior."""

    def test_start_simulation_basic(self):
        """Test basic mock_start_simulation functionality."""

        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(3)
        strategy = MagicMock()  # Mock strategy

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=5, config=config, strategy=strategy
        )

        # Verify result structure
        assert "history" in results
        assert "num_rounds" in results
        assert "num_clients" in results

        assert results["num_rounds"] == 3
        assert results["num_clients"] == 5

        # Verify history structure
        history = results["history"]
        assert "losses_distributed" in history
        assert "metrics_distributed" in history
        assert len(history["losses_distributed"]) == 3  # 3 rounds

    def test_start_simulation_with_client_resources(self):
        """Test mock_start_simulation with client resources."""

        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(2)
        strategy = MagicMock()
        client_resources = {"num_cpus": 2, "num_gpus": 0.5}

        results = mock_start_simulation(
            client_fn=client_fn,
            num_clients=3,
            config=config,
            strategy=strategy,
            client_resources=client_resources,
        )

        # Should complete without errors
        assert results["num_rounds"] == 2
        assert results["num_clients"] == 3


class TestUtilityFunctions:
    """Test utility functions for parameter conversion."""

    def test_ndarrays_to_parameters(self):
        """Test mock_ndarrays_to_parameters conversion."""
        arrays = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5], dtype=np.float32),
        ]

        params = mock_ndarrays_to_parameters(arrays)

        assert isinstance(params, MockParameters)
        assert len(params.tensors) == 2
        assert params.tensor_type == "numpy.ndarray"

    def test_parameters_to_ndarrays(self):
        """Test mock_parameters_to_ndarrays conversion."""
        arrays = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5], dtype=np.float32),
        ]

        params = mock_ndarrays_to_parameters(arrays)
        converted_arrays = mock_parameters_to_ndarrays(params)

        assert len(converted_arrays) == 2
        assert all(isinstance(arr, np.ndarray) for arr in converted_arrays)

        # Note: Exact equality might not hold due to serialization/deserialization
        # but shapes should match
        for orig, conv in zip(arrays, converted_arrays):
            assert orig.shape == conv.shape

    def test_weighted_loss_avg(self):
        """Test mock_weighted_loss_avg function."""
        results = [(100, 0.5), (200, 0.3), (50, 0.8)]

        avg_loss = mock_weighted_loss_avg(results)

        # Calculate expected weighted average
        total_examples = 100 + 200 + 50
        expected = (100 * 0.5 + 200 * 0.3 + 50 * 0.8) / total_examples

        assert abs(avg_loss - expected) < 1e-6

    def test_weighted_loss_avg_empty(self):
        """Test mock_weighted_loss_avg with empty results."""
        avg_loss = mock_weighted_loss_avg([])
        assert avg_loss == 0.0

    def test_weighted_loss_avg_zero_examples(self):
        """Test mock_weighted_loss_avg with zero examples."""
        results = [(0, 0.5), (0, 0.3)]
        avg_loss = mock_weighted_loss_avg(results)
        assert avg_loss == 0.0


class TestFactoryFunctions:
    """Test factory functions for creating mock components."""

    def test_create_mock_flower_client(self):
        """Test create_mock_flower_client factory function."""
        client = create_mock_flower_client(client_id=5)

        assert isinstance(client, MockClient)
        assert isinstance(client.numpy_client, MockNumPyClient)
        assert client.numpy_client.client_id == 5

    def test_create_mock_client_proxies(self):
        """Test create_mock_client_proxies factory function."""
        proxies = create_mock_client_proxies(3)

        assert len(proxies) == 3
        assert all(isinstance(p, MockClientProxy) for p in proxies)
        assert [p.cid for p in proxies] == ["0", "1", "2"]

    def test_create_mock_fit_results(self):
        """Test create_mock_fit_results factory function."""
        param_shapes = [(10, 5), (5,)]
        results = create_mock_fit_results(3, param_shapes)

        assert len(results) == 3
        assert all(isinstance(r, MockFitRes) for r in results)

        # Verify parameter shapes
        for result in results:
            assert len(result.parameters.tensors) == 2
            # Check that tensors have correct sizes (shape info is lost in serialization)
            tensor1 = np.frombuffer(result.parameters.tensors[0], dtype=np.float32)
            tensor2 = np.frombuffer(result.parameters.tensors[1], dtype=np.float32)
            assert tensor1.size == 10 * 5
            assert tensor2.size == 5

    def test_create_mock_evaluate_results(self):
        """Test create_mock_evaluate_results factory function."""
        results = create_mock_evaluate_results(4)

        assert len(results) == 4
        assert all(isinstance(r, MockEvaluateRes) for r in results)

        # Verify all results have required fields
        for result in results:
            assert 0.1 <= result.loss <= 1.5
            assert result.num_examples > 0
            assert "accuracy" in result.metrics
            assert "f1_score" in result.metrics


class TestMockBehaviorConsistency:
    """Test that mock behavior is consistent and predictable."""

    def test_deterministic_behavior_with_seeds(self):
        """Test that mocks produce deterministic results with proper seeding."""
        # Create two identical clients and reset their random state
        client1 = MockNumPyClient(client_id=42)
        client2 = MockNumPyClient(client_id=42)

        # Reset numpy random state to ensure deterministic behavior
        np.random.seed(42 + 42)  # Same seed as used in MockNumPyClient
        result1 = client1.get_parameters({})

        np.random.seed(42 + 42)  # Reset to same seed
        result2 = client2.get_parameters({})

        assert len(result1) == len(result2)
        for p1, p2 in zip(result1, result2):
            np.testing.assert_array_equal(p1, p2)

    def test_different_clients_produce_different_results(self):
        """Test that different client IDs produce different results."""
        client1 = MockNumPyClient(client_id=1)
        client2 = MockNumPyClient(client_id=2)

        params1 = client1.get_parameters({})
        params2 = client2.get_parameters({})

        # Should be different due to different seeds
        assert not np.array_equal(params1[0], params2[0])

    def test_simulation_produces_reasonable_metrics(self):
        """Test that simulation produces reasonable FL metrics."""

        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(5)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=10, config=config, strategy=strategy
        )

        # Verify metrics are in reasonable ranges
        losses = results["history"]["losses_distributed"]
        assert len(losses) == 5
        assert all(0.1 <= loss <= 2.0 for loss in losses)

        if "accuracy" in results["history"]["metrics_distributed"]:
            accuracies = results["history"]["metrics_distributed"]["accuracy"]
            assert all(0.0 <= acc <= 1.0 for acc in accuracies)


@pytest.mark.parametrize("num_clients", [1, 5, 10, 20])
def test_scalability_with_different_client_counts(num_clients):
    """Test that mocks work correctly with different numbers of clients."""

    def client_fn(cid: str):
        return create_mock_flower_client(int(cid))

    config = MockServerConfig(2)
    strategy = MagicMock()

    results = mock_start_simulation(
        client_fn=client_fn, num_clients=num_clients, config=config, strategy=strategy
    )

    assert results["num_clients"] == num_clients
    assert len(results["history"]["losses_distributed"]) == 2


@pytest.mark.parametrize("num_rounds", [1, 3, 5, 10])
def test_scalability_with_different_round_counts(num_rounds):
    """Test that mocks work correctly with different numbers of rounds."""

    def client_fn(cid: str):
        return create_mock_flower_client(int(cid))

    config = MockServerConfig(num_rounds)
    strategy = MagicMock()

    results = mock_start_simulation(
        client_fn=client_fn, num_clients=5, config=config, strategy=strategy
    )

    assert results["num_rounds"] == num_rounds
    assert len(results["history"]["losses_distributed"]) == num_rounds
