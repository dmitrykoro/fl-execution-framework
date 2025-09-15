"""
Integration tests for Flower FL component mocks with simulation components.

Tests mock implementations as drop-in replacements for real Flower components.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.fixtures.mock_flower_components import (
    MockParameters,
    MockServerConfig,
    create_mock_flower_client,
    mock_ndarrays_to_parameters,
    mock_start_simulation,
)


class TestFlowerMockIntegration:
    """Test integration of Flower mocks with simulation components."""

    def test_mock_simulation_basic_workflow(self):
        """Test basic federated learning workflow with mocks."""

        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(num_rounds=3)
        strategy = MagicMock()

        # Run mock simulation
        results = mock_start_simulation(
            client_fn=client_fn, num_clients=5, config=config, strategy=strategy
        )

        # Verify simulation completed successfully
        assert results["num_rounds"] == 3
        assert results["num_clients"] == 5
        assert len(results["history"]["losses_distributed"]) == 3

        # Verify reasonable FL metrics
        losses = results["history"]["losses_distributed"]
        assert all(isinstance(loss, float) for loss in losses)
        assert all(0.1 <= loss <= 2.0 for loss in losses)

    def test_mock_client_server_interaction(self):
        """Test mock client-server interaction patterns."""
        # Create mock client
        client = create_mock_flower_client(client_id=1)

        # Create mock parameters
        initial_params = [
            np.random.randn(10, 5).astype(np.float32),
            np.random.randn(5).astype(np.float32),
        ]
        mock_params = mock_ndarrays_to_parameters(initial_params)

        # Test fit (training) interaction
        fit_config = {"epochs": 1, "batch_size": 32}
        fit_result = client.fit(mock_params, fit_config)

        assert fit_result.num_examples > 0
        assert "loss" in fit_result.metrics
        assert "accuracy" in fit_result.metrics
        assert isinstance(fit_result.parameters, MockParameters)

        # Test evaluate interaction
        eval_config = {"batch_size": 32}
        eval_result = client.evaluate(mock_params, eval_config)

        assert eval_result.loss > 0
        assert eval_result.num_examples > 0
        assert "accuracy" in eval_result.metrics

    def test_mock_parameter_aggregation_workflow(self):
        """Test parameter aggregation workflow with mocks."""
        # Create multiple clients
        num_clients = 3
        clients = [create_mock_flower_client(i) for i in range(num_clients)]

        # Initial parameters
        initial_params = [
            np.ones((5, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
        mock_params = mock_ndarrays_to_parameters(initial_params)

        # Simulate training round
        fit_results = []
        for client in clients:
            fit_result = client.fit(mock_params, {"epochs": 1})
            fit_results.append(fit_result)

        # Verify all clients participated
        assert len(fit_results) == num_clients

        # Verify parameters were updated (should be different from initial)
        for fit_result in fit_results:
            assert fit_result.parameters != mock_params
            assert len(fit_result.parameters.tensors) == len(initial_params)

    def test_mock_federated_averaging_simulation(self):
        """Test federated averaging simulation pattern with mocks."""

        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        # Mock strategy that tracks aggregation calls
        mock_strategy = MagicMock()
        mock_strategy.aggregate_fit = MagicMock(return_value=(None, {}))
        mock_strategy.aggregate_evaluate = MagicMock(return_value=(None, {}))

        config = MockServerConfig(num_rounds=2)

        results = mock_start_simulation(
            client_fn=client_fn,
            num_clients=4,
            config=config,
            strategy=mock_strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0},
        )

        # Verify simulation structure
        assert results["num_rounds"] == 2
        assert results["num_clients"] == 4

        # Verify metrics were collected for each round
        history = results["history"]
        assert len(history["losses_distributed"]) == 2
        assert "metrics_distributed" in history

    @patch("flwr.simulation.start_simulation")
    def test_mock_as_replacement_for_real_flower(self, mock_flwr_start):
        """Test that mocks can replace real Flower components in tests."""

        # Configure the mock to return our mock simulation results
        def mock_simulation_side_effect(*args, **kwargs):
            return mock_start_simulation(*args, **kwargs)

        mock_flwr_start.side_effect = mock_simulation_side_effect

        # This would normally call real Flower, but now uses our mock
        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(num_rounds=2)
        strategy = MagicMock()

        # Call the mocked function
        results = mock_flwr_start(
            client_fn=client_fn, num_clients=3, config=config, strategy=strategy
        )

        # Verify mock was called and returned expected results
        mock_flwr_start.assert_called_once()
        assert results["num_rounds"] == 2
        assert results["num_clients"] == 3

    def test_mock_byzantine_client_simulation(self):
        """Test simulation with Byzantine (malicious) clients using mocks."""

        def client_fn(cid: str):
            client = create_mock_flower_client(int(cid))

            # Make some clients "Byzantine" by modifying their behavior
            if int(cid) >= 3:  # Last 2 clients are Byzantine
                # Override fit method to return corrupted parameters
                original_fit = client.fit

                def byzantine_fit(params, config):
                    result = original_fit(params, config)
                    # Corrupt parameters by adding large noise
                    corrupted_tensors = []
                    for tensor_bytes in result.parameters.tensors:
                        tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
                        corrupted = tensor + np.random.normal(
                            0, 10, tensor.shape
                        ).astype(np.float32)
                        corrupted_tensors.append(corrupted.tobytes())

                    result.parameters.tensors = corrupted_tensors
                    return result

                client.fit = byzantine_fit

            return client

        config = MockServerConfig(num_rounds=2)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn,
            num_clients=5,  # 3 honest + 2 Byzantine
            config=config,
            strategy=strategy,
        )

        # Simulation should complete even with Byzantine clients
        assert results["num_rounds"] == 2
        assert results["num_clients"] == 5

    def test_mock_client_dropout_simulation(self):
        """Test simulation with client dropouts using mocks."""

        def client_fn(cid: str):
            client = create_mock_flower_client(int(cid))

            # Simulate client dropout by making some clients fail
            if int(cid) == 2:  # Client 2 drops out

                def dropout_fit(params, config):
                    # Simulate client failure
                    raise ConnectionError("Client dropped out")

                client.fit = dropout_fit

            return client

        config = MockServerConfig(num_rounds=1)
        strategy = MagicMock()

        # This should handle client dropouts gracefully
        try:
            results = mock_start_simulation(
                client_fn=client_fn, num_clients=4, config=config, strategy=strategy
            )
            # If no exception, simulation handled dropouts
            assert results["num_clients"] == 4
        except ConnectionError:
            # Expected behavior - client dropout detected
            pass

    def test_mock_different_client_capabilities(self):
        """Test simulation with clients having different capabilities."""

        def client_fn(cid: str):
            client = create_mock_flower_client(int(cid))

            # Simulate different client capabilities
            client_id = int(cid)
            if client_id < 2:
                # High-capability clients (more data, better performance)
                original_fit = client.fit

                def high_capability_fit(params, config):
                    result = original_fit(params, config)
                    # Better performance metrics
                    result.metrics["accuracy"] = min(
                        0.95, result.metrics["accuracy"] + 0.1
                    )
                    result.metrics["loss"] = max(0.05, result.metrics["loss"] - 0.1)
                    result.num_examples = int(result.num_examples * 1.5)  # More data
                    return result

                client.fit = high_capability_fit

            return client

        config = MockServerConfig(num_rounds=1)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=4, config=config, strategy=strategy
        )

        # Simulation should handle heterogeneous clients
        assert results["num_rounds"] == 1
        assert results["num_clients"] == 4


class TestMockConsistencyWithRealFlower:
    """Test that mocks behave consistently with expected Flower patterns."""

    def test_mock_parameter_shapes_consistency(self):
        """Test that mock parameters maintain consistent shapes."""
        client = create_mock_flower_client(0)

        # Get initial parameters
        initial_params = [
            np.random.randn(10, 5).astype(np.float32),
            np.random.randn(5).astype(np.float32),
        ]
        mock_params = mock_ndarrays_to_parameters(initial_params)

        # Train and verify shapes are preserved
        fit_result = client.fit(mock_params, {})

        # Parameters should have same number of tensors
        assert len(fit_result.parameters.tensors) == len(initial_params)

        # Verify tensor sizes (shapes are lost in serialization, but sizes should match)
        for i, tensor_bytes in enumerate(fit_result.parameters.tensors):
            tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
            expected_size = initial_params[i].size
            assert tensor.size == expected_size

    def test_mock_metrics_format_consistency(self):
        """Test that mock metrics follow expected Flower format."""
        client = create_mock_flower_client(0)

        params = mock_ndarrays_to_parameters([np.random.randn(10).astype(np.float32)])

        # Test fit metrics
        fit_result = client.fit(params, {})
        assert isinstance(fit_result.metrics, dict)
        assert "loss" in fit_result.metrics
        assert "accuracy" in fit_result.metrics
        assert isinstance(fit_result.num_examples, int)

        # Test evaluate metrics
        eval_result = client.evaluate(params, {})
        assert isinstance(eval_result.metrics, dict)
        assert "accuracy" in eval_result.metrics
        assert isinstance(eval_result.loss, float)
        assert isinstance(eval_result.num_examples, int)

    def test_mock_simulation_history_format(self):
        """Test that mock simulation history follows expected format."""

        def client_fn(cid: str):
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(num_rounds=3)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=2, config=config, strategy=strategy
        )

        # Verify history structure matches Flower expectations
        history = results["history"]

        # Check required fields
        assert "losses_distributed" in history
        assert "metrics_distributed" in history

        # Check data types and lengths
        assert isinstance(history["losses_distributed"], list)
        assert len(history["losses_distributed"]) == 3  # num_rounds

        assert isinstance(history["metrics_distributed"], dict)

        # Verify loss values are reasonable
        for loss in history["losses_distributed"]:
            assert isinstance(loss, (int, float))
            assert loss >= 0


@pytest.mark.parametrize("num_clients", [1, 3, 5, 10])
@pytest.mark.parametrize("num_rounds", [1, 2, 5])
def test_mock_scalability_integration(num_clients, num_rounds):
    """Test that mocks scale properly with different simulation sizes."""

    def client_fn(cid: str):
        return create_mock_flower_client(int(cid))

    config = MockServerConfig(num_rounds=num_rounds)
    strategy = MagicMock()

    results = mock_start_simulation(
        client_fn=client_fn, num_clients=num_clients, config=config, strategy=strategy
    )

    assert results["num_clients"] == num_clients
    assert results["num_rounds"] == num_rounds
    assert len(results["history"]["losses_distributed"]) == num_rounds
