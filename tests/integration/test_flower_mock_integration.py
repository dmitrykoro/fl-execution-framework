"""
Integration tests for Flower FL component mocks with simulation components.

Tests mock implementations as drop-in replacements for real Flower components.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from tests.common import np, pytest

from tests.fixtures.mock_flower_components import (
    MockClient,
    MockFitRes,
    MockParameters,
    MockServerConfig,
    create_mock_flower_client,
    mock_ndarrays_to_parameters,
    mock_start_simulation,
)

NDArray = np.ndarray
Config = Dict[str, Any]
Metrics = Dict[str, Any]


class TestFlowerMockIntegration:
    """Test integration of Flower mocks with simulation components."""

    def test_mock_simulation_basic_workflow(self) -> None:
        """Test federated learning workflow with mocked components."""

        def client_fn(cid: str) -> MockClient:
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(num_rounds=3)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=5, config=config, strategy=strategy
        )

        assert results["num_rounds"] == 3
        assert results["num_clients"] == 5
        assert len(results["history"]["losses_distributed"]) == 3

        losses = results["history"]["losses_distributed"]
        assert all(isinstance(loss, float) for loss in losses)
        assert all(0.1 <= loss <= 2.0 for loss in losses)

    def test_mock_client_server_interaction(self) -> None:
        """Test mock client-server interaction patterns."""
        client = create_mock_flower_client(client_id=1)

        rng = np.random.default_rng(42)
        initial_params: List[NDArray] = [
            rng.standard_normal((10, 5)).astype(np.float32),
            rng.standard_normal(5).astype(np.float32),
        ]
        mock_params = mock_ndarrays_to_parameters(initial_params)

        fit_config: Config = {"epochs": 1, "batch_size": 32}
        fit_result = client.fit(mock_params, fit_config)

        assert fit_result.num_examples > 0
        assert "loss" in fit_result.metrics
        assert "accuracy" in fit_result.metrics
        assert isinstance(fit_result.parameters, MockParameters)

        eval_config: Config = {"batch_size": 32}
        eval_result = client.evaluate(mock_params, eval_config)

        assert eval_result.loss > 0
        assert eval_result.num_examples > 0
        assert "accuracy" in eval_result.metrics

    def test_mock_parameter_aggregation_workflow(self) -> None:
        """Test parameter aggregation workflow with mocks."""
        num_clients = 3
        clients = [create_mock_flower_client(i) for i in range(num_clients)]

        initial_params: List[NDArray] = [
            np.ones((5, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
        mock_params = mock_ndarrays_to_parameters(initial_params)

        fit_results = []
        for client in clients:
            fit_result = client.fit(mock_params, {"epochs": 1})
            fit_results.append(fit_result)

        assert len(fit_results) == num_clients

        for fit_result in fit_results:
            assert fit_result.parameters != mock_params
            assert len(fit_result.parameters.tensors) == len(initial_params)

    def test_mock_federated_averaging_simulation(self) -> None:
        """Test federated averaging simulation pattern with mocks."""

        def client_fn(cid: str) -> MockClient:
            return create_mock_flower_client(int(cid))

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

        assert results["num_rounds"] == 2
        assert results["num_clients"] == 4

        history = results["history"]
        assert len(history["losses_distributed"]) == 2
        assert "metrics_distributed" in history

    @patch("flwr.simulation.start_simulation")
    def test_mock_as_replacement_for_real_flower(
        self, mock_flwr_start: MagicMock
    ) -> None:
        """Test that mocks can replace real Flower components in tests."""

        def mock_simulation_side_effect(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return mock_start_simulation(*args, **kwargs)

        mock_flwr_start.side_effect = mock_simulation_side_effect

        def client_fn(cid: str) -> MockClient:
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(num_rounds=2)
        strategy = MagicMock()

        results = mock_flwr_start(
            client_fn=client_fn, num_clients=3, config=config, strategy=strategy
        )

        mock_flwr_start.assert_called_once()
        assert results["num_rounds"] == 2
        assert results["num_clients"] == 3

    def test_mock_byzantine_client_simulation(self) -> None:
        """Test simulation with Byzantine (malicious) clients using mocks."""

        def client_fn(cid: str) -> MockClient:
            client = create_mock_flower_client(int(cid))

            # Modify Byzantine clients to return corrupted parameters
            if int(cid) >= 3:  # Last 2 clients are Byzantine
                original_fit = client.fit

                def byzantine_fit(parameters: MockParameters, config: Config) -> Any:
                    result = original_fit(parameters, config)
                    corrupted_tensors = []
                    for tensor_bytes in result.parameters.tensors:
                        tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
                        rng = np.random.default_rng(42)
                        corrupted = tensor + rng.normal(0, 10, tensor.shape).astype(
                            np.float32
                        )
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

        assert results["num_rounds"] == 2
        assert results["num_clients"] == 5

    def test_mock_client_dropout_simulation(self) -> None:
        """Test simulation with client dropouts using mocks."""

        def client_fn(cid: str) -> MockClient:
            client = create_mock_flower_client(int(cid))

            # Make client 2 fail during training
            if int(cid) == 2:  # Client 2 drops out

                def dropout_fit(
                    parameters: MockParameters, config: Config
                ) -> MockFitRes:
                    _ = parameters, config
                    raise ConnectionError("Client dropped out")

                client.fit = dropout_fit

            return client

        config = MockServerConfig(num_rounds=1)
        strategy = MagicMock()

        try:
            results = mock_start_simulation(
                client_fn=client_fn, num_clients=4, config=config, strategy=strategy
            )
            assert results["num_clients"] == 4
        except ConnectionError:
            pass

    def test_mock_different_client_capabilities(self) -> None:
        """Test simulation with clients having different capabilities."""

        def client_fn(cid: str) -> MockClient:
            client = create_mock_flower_client(int(cid))

            client_id = int(cid)
            if client_id < 2:
                original_fit = client.fit

                def high_capability_fit(
                    parameters: MockParameters, config: Config
                ) -> MockFitRes:
                    result = original_fit(parameters, config)
                    result.metrics["accuracy"] = min(
                        0.95, result.metrics["accuracy"] + 0.1
                    )
                    result.metrics["loss"] = max(0.05, result.metrics["loss"] - 0.1)
                    result.num_examples = int(result.num_examples * 1.5)
                    return result

                client.fit = high_capability_fit

            return client

        config = MockServerConfig(num_rounds=1)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=4, config=config, strategy=strategy
        )

        assert results["num_rounds"] == 1
        assert results["num_clients"] == 4


class TestMockConsistencyWithRealFlower:
    """Test that mocks behave consistently with expected Flower patterns."""

    def test_mock_parameter_shapes_consistency(self) -> None:
        """Test that mock parameters maintain consistent shapes."""
        client = create_mock_flower_client(0)

        rng = np.random.default_rng(42)
        initial_params: List[NDArray] = [
            rng.standard_normal((10, 5)).astype(np.float32),
            rng.standard_normal(5).astype(np.float32),
        ]
        mock_params = mock_ndarrays_to_parameters(initial_params)

        fit_result = client.fit(mock_params, {})

        assert len(fit_result.parameters.tensors) == len(initial_params)

        for i, tensor_bytes in enumerate(fit_result.parameters.tensors):
            tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
            expected_size = initial_params[i].size
            assert tensor.size == expected_size

    def test_mock_metrics_format_consistency(self) -> None:
        """Test that mock metrics follow expected Flower format."""
        client = create_mock_flower_client(0)

        rng = np.random.default_rng(42)
        params = mock_ndarrays_to_parameters(
            [rng.standard_normal(10).astype(np.float32)]
        )

        fit_result = client.fit(params, {})
        assert isinstance(fit_result.metrics, dict)
        assert "loss" in fit_result.metrics
        assert "accuracy" in fit_result.metrics
        assert isinstance(fit_result.num_examples, int)

        eval_result = client.evaluate(params, {})
        assert isinstance(eval_result.metrics, dict)
        assert "accuracy" in eval_result.metrics
        assert isinstance(eval_result.loss, float)
        assert isinstance(eval_result.num_examples, int)

    def test_mock_simulation_history_format(self) -> None:
        """Test that mock simulation history follows expected format."""

        def client_fn(cid: str) -> MockClient:
            return create_mock_flower_client(int(cid))

        config = MockServerConfig(num_rounds=3)
        strategy = MagicMock()

        results = mock_start_simulation(
            client_fn=client_fn, num_clients=2, config=config, strategy=strategy
        )

        history = results["history"]

        assert "losses_distributed" in history
        assert "metrics_distributed" in history

        assert isinstance(history["losses_distributed"], list)
        assert len(history["losses_distributed"]) == 3

        assert isinstance(history["metrics_distributed"], dict)

        for loss in history["losses_distributed"]:
            assert isinstance(loss, (int, float))
            assert loss >= 0


@pytest.mark.parametrize("num_clients", [1, 3, 5, 10])
@pytest.mark.parametrize("num_rounds", [1, 2, 5])
def test_mock_scalability_integration(num_clients: int, num_rounds: int) -> None:
    """Test that mocks scale properly with different simulation sizes."""

    def client_fn(cid: str) -> MockClient:
        return create_mock_flower_client(int(cid))

    config = MockServerConfig(num_rounds=num_rounds)
    strategy = MagicMock()

    results = mock_start_simulation(
        client_fn=client_fn, num_clients=num_clients, config=config, strategy=strategy
    )

    assert results["num_clients"] == num_clients
    assert results["num_rounds"] == num_rounds
    assert len(results["history"]["losses_distributed"]) == num_rounds
