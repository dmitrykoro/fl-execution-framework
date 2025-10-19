"""
Unit tests for FedAvgStrategy.

Tests FedAvg wrapper with metrics tracking, round-level aggregation, and history management.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.fedavg_strategy import FedAvgStrategy


class TestFedAvgStrategy:
    """Test cases for FedAvgStrategy."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        history = Mock(spec=SimulationStrategyHistory)
        history.rounds_history = Mock()
        history.rounds_history.aggregated_loss_history = []
        history.rounds_history.average_accuracy_history = []
        history.insert_single_client_history_entry = Mock()
        return history

    @pytest.fixture
    def fedavg_strategy(self, mock_strategy_history):
        """Create FedAvgStrategy instance for testing."""
        return FedAvgStrategy(
            strategy_history=mock_strategy_history,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )

    @pytest.fixture
    def mock_fit_results(self, mock_client_results_factory):
        """Generate mock fit results for testing."""
        return mock_client_results_factory(3)

    @pytest.fixture
    def mock_evaluate_results(self, mock_evaluate_results_factory):
        """Generate mock evaluate results for testing."""
        return mock_evaluate_results_factory(
            num_clients=3, base_accuracy=0.8, base_loss=0.5, accuracy_increment=0.05
        )

    def test_initialization(self, fedavg_strategy, mock_strategy_history):
        """Test FedAvgStrategy initialization."""
        assert fedavg_strategy.strategy_history == mock_strategy_history
        assert fedavg_strategy.current_round == 0
        assert fedavg_strategy.logger is not None
        assert fedavg_strategy.fraction_fit == 1.0
        assert fedavg_strategy.fraction_evaluate == 1.0

    def test_aggregate_fit_updates_round(self, fedavg_strategy, mock_fit_results):
        """Test aggregate_fit updates current_round."""
        server_round = 5

        fedavg_strategy.aggregate_fit(server_round, mock_fit_results, [])

        assert fedavg_strategy.current_round == server_round

    def test_aggregate_fit_calls_parent(self, fedavg_strategy, mock_fit_results):
        """Test aggregate_fit calls parent FedAvg implementation."""
        server_round = 1

        result = fedavg_strategy.aggregate_fit(server_round, mock_fit_results, [])

        # Should return tuple of (Parameters, metrics dict)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aggregate_evaluate_empty_results(self, fedavg_strategy):
        """Test aggregate_evaluate with empty results."""
        result = fedavg_strategy.aggregate_evaluate(1, [], [])

        assert result == (None, {})

    def test_aggregate_evaluate_collects_per_client_metrics(
        self, fedavg_strategy, mock_evaluate_results, mock_strategy_history
    ):
        """Test aggregate_evaluate collects per-client metrics."""
        server_round = 1

        fedavg_strategy.aggregate_evaluate(server_round, mock_evaluate_results, [])

        # Should call insert_single_client_history_entry for each client
        assert mock_strategy_history.insert_single_client_history_entry.call_count == 3

    def test_aggregate_evaluate_stores_per_client_data(
        self, fedavg_strategy, mock_evaluate_results, mock_strategy_history
    ):
        """Test aggregate_evaluate stores correct per-client data."""
        server_round = 2
        fedavg_strategy.current_round = 2

        fedavg_strategy.aggregate_evaluate(server_round, mock_evaluate_results, [])

        # Verify correct data was stored for first client
        first_call = (
            mock_strategy_history.insert_single_client_history_entry.call_args_list[0]
        )
        assert first_call[1]["client_id"] == 0
        assert first_call[1]["current_round"] == 2
        assert first_call[1]["loss"] == 0.5
        assert first_call[1]["accuracy"] == 0.8

    def test_aggregate_evaluate_calculates_aggregated_loss(
        self, fedavg_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate calculates weighted aggregated loss."""
        server_round = 1

        loss_aggregated, _ = fedavg_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, []
        )

        # Should return a float loss value
        assert isinstance(loss_aggregated, float)
        assert loss_aggregated >= 0

    def test_aggregate_evaluate_calculates_average_accuracy(
        self, fedavg_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate calculates weighted average accuracy."""
        server_round = 1

        _, metrics = fedavg_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, []
        )

        # Should include accuracy in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_aggregate_evaluate_weighted_accuracy_calculation(
        self, fedavg_strategy, mock_strategy_history
    ):
        """Test aggregate_evaluate uses weighted average for accuracy."""
        # Create results with known values for verification
        results = []

        # Client 0: 100 examples, accuracy 0.8
        client0 = Mock(spec=ClientProxy)
        client0.cid = "0"
        eval0 = Mock(spec=EvaluateRes)
        eval0.num_examples = 100
        eval0.loss = 0.5
        eval0.metrics = {"accuracy": 0.8}
        results.append((client0, eval0))

        # Client 1: 200 examples, accuracy 0.9
        client1 = Mock(spec=ClientProxy)
        client1.cid = "1"
        eval1 = Mock(spec=EvaluateRes)
        eval1.num_examples = 200
        eval1.loss = 0.4
        eval1.metrics = {"accuracy": 0.9}
        results.append((client1, eval1))

        _, metrics = fedavg_strategy.aggregate_evaluate(1, results, [])

        # Weighted average: (100 * 0.8 + 200 * 0.9) / 300 = (80 + 180) / 300 = 0.8667
        expected_accuracy = (100 * 0.8 + 200 * 0.9) / 300
        assert abs(metrics["accuracy"] - expected_accuracy) < 1e-6

    def test_aggregate_evaluate_stores_round_metrics(
        self, fedavg_strategy, mock_evaluate_results, mock_strategy_history
    ):
        """Test aggregate_evaluate stores round-level metrics in history."""
        server_round = 1

        loss_aggregated, metrics = fedavg_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, []
        )

        # Should append to history lists
        assert len(mock_strategy_history.rounds_history.aggregated_loss_history) == 1
        assert len(mock_strategy_history.rounds_history.average_accuracy_history) == 1

        # Verify correct values stored
        assert (
            mock_strategy_history.rounds_history.aggregated_loss_history[0]
            == loss_aggregated
        )
        assert (
            mock_strategy_history.rounds_history.average_accuracy_history[0]
            == metrics["accuracy"]
        )

    def test_aggregate_evaluate_multiple_rounds(
        self, fedavg_strategy, mock_evaluate_results, mock_strategy_history
    ):
        """Test aggregate_evaluate over multiple rounds."""
        for round_num in range(1, 4):
            fedavg_strategy.aggregate_evaluate(round_num, mock_evaluate_results, [])

        # Should have 3 entries in history
        assert len(mock_strategy_history.rounds_history.aggregated_loss_history) == 3
        assert len(mock_strategy_history.rounds_history.average_accuracy_history) == 3

    def test_aggregate_evaluate_handles_missing_accuracy(
        self, fedavg_strategy, mock_strategy_history
    ):
        """Test aggregate_evaluate handles missing accuracy metric."""
        # Create result without accuracy
        client = Mock(spec=ClientProxy)
        client.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.num_examples = 100
        eval_res.loss = 0.5
        eval_res.metrics = {}  # No accuracy

        results = [(client, eval_res)]

        _, metrics = fedavg_strategy.aggregate_evaluate(1, results, [])

        # Should default to 0.0 for missing accuracy
        assert metrics["accuracy"] == 0.0

    def test_aggregate_evaluate_zero_examples(self, fedavg_strategy):
        """Test aggregate_evaluate with zero total examples raises ZeroDivisionError."""
        # Create results with zero examples
        results = []
        for i in range(2):
            client = Mock(spec=ClientProxy)
            client.cid = str(i)
            eval_res = Mock(spec=EvaluateRes)
            eval_res.num_examples = 0
            eval_res.loss = 0.5
            eval_res.metrics = {"accuracy": 0.8}
            results.append((client, eval_res))

        # Should raise ZeroDivisionError due to weighted_loss_avg
        with pytest.raises(ZeroDivisionError):
            fedavg_strategy.aggregate_evaluate(1, results, [])

    def test_aggregate_evaluate_single_client(
        self, fedavg_strategy, mock_strategy_history
    ):
        """Test aggregate_evaluate with single client."""
        client = Mock(spec=ClientProxy)
        client.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.num_examples = 100
        eval_res.loss = 0.5
        eval_res.metrics = {"accuracy": 0.85}

        results = [(client, eval_res)]

        loss_aggregated, metrics = fedavg_strategy.aggregate_evaluate(1, results, [])

        # With single client, accuracy should match client's accuracy
        assert metrics["accuracy"] == 0.85
        assert loss_aggregated == 0.5

    def test_aggregate_evaluate_with_failures(
        self, fedavg_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate handles failures parameter."""
        server_round = 1
        failures = [Exception("Test failure")]

        # Should process successfully despite failures
        loss_aggregated, metrics = fedavg_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, failures
        )

        assert isinstance(loss_aggregated, float)
        assert "accuracy" in metrics

    def test_aggregate_fit_with_failures(self, fedavg_strategy, mock_fit_results):
        """Test aggregate_fit handles failures parameter."""
        server_round = 1
        failures = [Exception("Test failure")]

        # Should process successfully despite failures
        result = fedavg_strategy.aggregate_fit(server_round, mock_fit_results, failures)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_logger_naming(self, mock_strategy_history):
        """Test logger has unique name based on instance."""
        strategy1 = FedAvgStrategy(strategy_history=mock_strategy_history)
        strategy2 = FedAvgStrategy(strategy_history=mock_strategy_history)

        # Each instance should have unique logger
        assert strategy1.logger.name != strategy2.logger.name
        assert "fedavg_strategy" in strategy1.logger.name
        assert "fedavg_strategy" in strategy2.logger.name

    def test_aggregate_evaluate_accuracy_values_in_range(
        self, fedavg_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate produces accuracy values in valid range."""
        # Test with various accuracy values
        for round_num in range(1, 6):
            _, metrics = fedavg_strategy.aggregate_evaluate(
                round_num, mock_evaluate_results, []
            )

            # Accuracy should be between 0 and 1
            assert 0 <= metrics["accuracy"] <= 1

    def test_aggregate_evaluate_consistent_round_tracking(
        self, fedavg_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate uses current_round consistently."""
        fedavg_strategy.current_round = 5
        server_round = 5

        fedavg_strategy.aggregate_evaluate(server_round, mock_evaluate_results, [])

        # Verify all client entries use current_round
        calls = fedavg_strategy.strategy_history.insert_single_client_history_entry.call_args_list
        for call in calls:
            assert call[1]["current_round"] == 5

    def test_initialization_with_custom_parameters(self, mock_strategy_history):
        """Test initialization with custom FedAvg parameters."""
        strategy = FedAvgStrategy(
            strategy_history=mock_strategy_history,
            fraction_fit=0.5,
            fraction_evaluate=0.7,
            min_fit_clients=5,
            min_evaluate_clients=3,
            min_available_clients=10,
        )

        assert strategy.fraction_fit == 0.5
        assert strategy.fraction_evaluate == 0.7

    def test_aggregate_evaluate_large_number_of_clients(
        self, fedavg_strategy, mock_strategy_history
    ):
        """Test aggregate_evaluate with large number of clients."""
        # Create 100 clients
        results = []
        for i in range(100):
            client = Mock(spec=ClientProxy)
            client.cid = str(i)
            eval_res = Mock(spec=EvaluateRes)
            eval_res.num_examples = 100
            eval_res.loss = 0.5 + (i * 0.001)
            eval_res.metrics = {"accuracy": 0.7 + (i * 0.001)}
            results.append((client, eval_res))

        loss_aggregated, metrics = fedavg_strategy.aggregate_evaluate(1, results, [])

        # Should handle large number of clients
        assert isinstance(loss_aggregated, float)
        assert 0 <= metrics["accuracy"] <= 1
        assert (
            mock_strategy_history.insert_single_client_history_entry.call_count == 100
        )

    def test_aggregate_evaluate_numerical_stability(self, fedavg_strategy):
        """Test aggregate_evaluate maintains numerical stability."""
        # Create results with very small and very large values
        results = []

        client1 = Mock(spec=ClientProxy)
        client1.cid = "0"
        eval1 = Mock(spec=EvaluateRes)
        eval1.num_examples = 1
        eval1.loss = 0.0001
        eval1.metrics = {"accuracy": 0.0001}
        results.append((client1, eval1))

        client2 = Mock(spec=ClientProxy)
        client2.cid = "1"
        eval2 = Mock(spec=EvaluateRes)
        eval2.num_examples = 1000000
        eval2.loss = 100.0
        eval2.metrics = {"accuracy": 0.9999}
        results.append((client2, eval2))

        loss_aggregated, metrics = fedavg_strategy.aggregate_evaluate(1, results, [])

        # Should handle extreme values without errors
        assert isinstance(loss_aggregated, float)
        assert not np.isnan(loss_aggregated)
        assert not np.isinf(loss_aggregated)
        assert not np.isnan(metrics["accuracy"])
        assert not np.isinf(metrics["accuracy"])

    def test_aggregate_fit_empty_results(self, fedavg_strategy):
        """Test aggregate_fit with empty results."""
        result = fedavg_strategy.aggregate_fit(1, [], [])

        # Should return None parameters and empty metrics
        assert result[0] is None
        assert result[1] == {}

    def test_current_round_increments_correctly(
        self, fedavg_strategy, mock_fit_results
    ):
        """Test current_round increments correctly over multiple rounds."""
        for round_num in range(1, 6):
            fedavg_strategy.aggregate_fit(round_num, mock_fit_results, [])
            assert fedavg_strategy.current_round == round_num
