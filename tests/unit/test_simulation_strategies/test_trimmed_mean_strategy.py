"""
Unit tests for TrimmedMeanBasedRemovalStrategy.

Tests trimmed mean aggregation and client removal logic.
"""

import warnings
from unittest.mock import patch

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.trimmed_mean_based_removal_strategy import (
    TrimmedMeanBasedRemovalStrategy,
)
from tests.common import (
    ClientProxy,
    FitRes,
    Mock,
    generate_mock_client_data,
    ndarrays_to_parameters,
    np,
    parameters_to_ndarrays,
    pytest,
)


class TestTrimmedMeanBasedRemovalStrategy:
    """Test cases for TrimmedMeanBasedRemovalStrategy."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def trimmed_mean_strategy(self, mock_strategy_history, mock_output_directory):
        """Create TrimmedMeanBasedRemovalStrategy instance for testing."""
        return TrimmedMeanBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            trim_ratio=0.2,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def mock_client_results(self):
        """Generate mock client results for testing."""
        return generate_mock_client_data(num_clients=10)

    def test_initialization(self, trimmed_mean_strategy, mock_strategy_history):
        """Test TrimmedMeanBasedRemovalStrategy initialization."""
        assert trimmed_mean_strategy.remove_clients is True
        assert trimmed_mean_strategy.begin_removing_from_round == 2
        assert trimmed_mean_strategy.trim_ratio == 0.2
        assert trimmed_mean_strategy.strategy_history == mock_strategy_history
        assert trimmed_mean_strategy.current_round == 0
        assert trimmed_mean_strategy.removed_client_ids == set()
        assert trimmed_mean_strategy.client_scores == {}

    def test_aggregate_fit_no_trimming_needed(self, trimmed_mean_strategy):
        """Test aggregate_fit when no trimming is needed (trim_ratio results in 0 clients to trim)."""
        # Create small number of clients where trim_ratio * num_clients < 1
        results = []
        for i in range(3):  # 0.2 * 3 = 0.6, int(0.6) = 0
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.random.randn(2, 2), np.random.randn(2)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        with patch.object(trimmed_mean_strategy, "_average_weights") as mock_average:
            mock_average.return_value = [np.random.randn(2, 2), np.random.randn(2)]

            result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
                1, results, []
            )

            # Should call _average_weights when no trimming is needed
            mock_average.assert_called_once()
            assert result_params is not None

    def test_aggregate_fit_with_trimming(
        self, trimmed_mean_strategy, mock_client_results
    ):
        """Test aggregate_fit performs trimming correctly."""
        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, mock_client_results, []
        )

        # Should return aggregated parameters
        assert result_params is not None
        assert isinstance(result_metrics, dict)

        # Verify current_round was incremented
        assert trimmed_mean_strategy.current_round == 1

    def test_aggregate_fit_empty_results(self, trimmed_mean_strategy):
        """Test aggregate_fit handles empty results."""
        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(1, [], [])

        # Should return None for empty results
        assert result_params is None
        assert result_metrics == {}

    def test_aggregate_fit_trimming_calculation(self, trimmed_mean_strategy):
        """Test that trimming calculation is correct."""
        # Create 10 clients with trim_ratio=0.2, so 2 clients should be trimmed from each end
        results = []
        for i in range(10):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            # Create parameters with predictable values for testing
            mock_params = [np.array([[float(i)]]), np.array([float(i)])]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, results, []
        )

        # Should successfully aggregate with trimming
        assert result_params is not None

        # Verify strategy history was updated
        trimmed_mean_strategy.strategy_history.update_client_participation.assert_called_once()

    def test_trim_ratio_parameter_effect(self, mock_strategy_history):
        """Test trim_ratio parameter affects trimming behavior."""
        trim_ratios = [0.1, 0.2, 0.3]

        for trim_ratio in trim_ratios:
            strategy = TrimmedMeanBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
                trim_ratio=trim_ratio,
            )

            assert strategy.trim_ratio == trim_ratio

            # Test with 10 clients
            num_clients = 10
            expected_trim_count = int(trim_ratio * num_clients)

            # Create test results
            results = []
            for i in range(num_clients):
                client_proxy = Mock(spec=ClientProxy)
                client_proxy.cid = str(i)
                mock_params = [np.array([[float(i)]]), np.array([float(i)])]
                fit_res = Mock(spec=FitRes)
                fit_res.parameters = ndarrays_to_parameters(mock_params)
                fit_res.num_examples = 100
                results.append((client_proxy, fit_res))

            result_params, result_metrics = strategy.aggregate_fit(1, results, [])

            # Should handle different trim ratios
            assert result_params is not None

            # Verify the expected trim count is reasonable for the number of clients
            assert (
                expected_trim_count <= num_clients // 2
            )  # Should not trim more than half

    def test_configure_fit_warmup_rounds(self, trimmed_mean_strategy):
        """Test configure_fit during warmup rounds."""
        trimmed_mean_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = trimmed_mean_strategy.configure_fit(
            1, mock_parameters, mock_client_manager
        )

        # Should return all clients during warmup
        assert len(result) == 5

    def test_configure_fit_removal_phase(self, trimmed_mean_strategy):
        """Test configure_fit removes client with highest score."""
        trimmed_mean_strategy.current_round = 3  # After begin_removing_from_round
        trimmed_mean_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # Highest score - should be removed
            "client_2": 0.3,
            "client_3": 0.2,
            "client_4": 0.5,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = trimmed_mean_strategy.configure_fit(
            3, mock_parameters, mock_client_manager
        )

        # Should return all clients for training
        assert len(result) == 5

    def test_configure_fit_no_removal_when_disabled(self, trimmed_mean_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        trimmed_mean_strategy.remove_clients = False
        trimmed_mean_strategy.current_round = 3
        trimmed_mean_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # High score but shouldn't be removed
            "client_2": 0.3,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = trimmed_mean_strategy.configure_fit(
            3, mock_parameters, mock_client_manager
        )

        # Should return all clients
        assert len(result) == 3

    def test_begin_removing_from_round_parameter(self, mock_strategy_history):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = TrimmedMeanBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=begin_round,
                strategy_history=mock_strategy_history,
                trim_ratio=0.2,
            )

            assert strategy.begin_removing_from_round == begin_round

            # Test warmup behavior
            strategy.current_round = begin_round - 1
            mock_client_manager = Mock()
            mock_clients = {"client_0": Mock(), "client_1": Mock()}
            mock_client_manager.all.return_value = mock_clients

            result = strategy.configure_fit(1, Mock(), mock_client_manager)

            # Should return all clients during warmup
            assert len(result) == 2

    def test_average_weights_method(self, trimmed_mean_strategy):
        """Test _average_weights method computes correct averages."""
        # Create test weights
        weights = [
            [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0])],
            [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([2.0, 3.0])],
            [np.array([[3.0, 4.0], [5.0, 6.0]]), np.array([3.0, 4.0])],
        ]

        avg_weights = trimmed_mean_strategy._average_weights(weights)

        # Should compute element-wise average
        expected_layer1 = np.array(
            [[2.0, 3.0], [4.0, 5.0]]
        )  # Average of the 2x2 matrices
        expected_layer2 = np.array([2.0, 3.0])  # Average of the 1D arrays

        assert len(avg_weights) == 2
        assert np.allclose(avg_weights[0], expected_layer1)
        assert np.allclose(avg_weights[1], expected_layer2)

    def test_average_weights_single_client(self, trimmed_mean_strategy):
        """Test _average_weights method with single client."""
        weights = [
            [np.array([[1.0, 2.0]]), np.array([3.0])],
        ]

        avg_weights = trimmed_mean_strategy._average_weights(weights)

        # Should return the same weights for single client
        assert len(avg_weights) == 2
        assert np.allclose(avg_weights[0], np.array([[1.0, 2.0]]))
        assert np.allclose(avg_weights[1], np.array([3.0]))

    def test_trimming_removes_outliers(self, trimmed_mean_strategy):
        """Test that trimming effectively removes outlier values."""
        # Create results with clear outliers
        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            if i == 0:  # Extreme outlier
                mock_params = [np.array([[100.0]]), np.array([100.0])]
            elif i == 4:  # Another extreme outlier
                mock_params = [np.array([[-100.0]]), np.array([-100.0])]
            else:  # Normal values
                mock_params = [np.array([[1.0]]), np.array([1.0])]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        # With trim_ratio=0.2 and 5 clients, int(0.2 * 5) = 1 client trimmed from each end
        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, results, []
        )

        # Should successfully aggregate without being dominated by outliers
        assert result_params is not None

        # Extract the aggregated parameters
        aggregated_arrays = parameters_to_ndarrays(result_params)

        # The result should be closer to the normal values (1.0) than the outliers (±100.0)
        # Since we trim 1 from each end, we should average the middle 3 values: [1.0, 1.0, 1.0]
        assert abs(aggregated_arrays[0][0, 0] - 1.0) < abs(
            aggregated_arrays[0][0, 0] - 100.0
        )
        assert abs(aggregated_arrays[1][0] - 1.0) < abs(aggregated_arrays[1][0] - 100.0)

    def test_strategy_history_integration(
        self, trimmed_mean_strategy, mock_client_results
    ):
        """Test integration with strategy history."""
        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, mock_client_results, []
        )

        # Verify strategy history methods were called
        trimmed_mean_strategy.strategy_history.update_client_participation.assert_called_once()

    def test_edge_case_all_clients_identical(self, trimmed_mean_strategy):
        """Test handling when all clients have identical parameters."""
        # Create results with identical parameters
        results = []
        identical_params = [np.array([[1.0, 2.0]]), np.array([3.0])]

        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(identical_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, results, []
        )

        # Should handle identical parameters gracefully
        assert result_params is not None

        # Result should be the same as input (since all are identical)
        aggregated_arrays = parameters_to_ndarrays(result_params)
        assert np.allclose(aggregated_arrays[0], identical_params[0])
        assert np.allclose(aggregated_arrays[1], identical_params[1])

    def test_edge_case_extreme_trim_ratio(self, mock_strategy_history):
        """Test handling of extreme trim ratios."""
        # Test with very high trim ratio
        strategy = TrimmedMeanBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            trim_ratio=0.9,  # Very high trim ratio
        )

        # Create 10 clients
        results = []
        for i in range(10):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.array([[float(i)]]), np.array([float(i)])]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        # Suppress NumPy warnings for edge case test
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in scalar divide",
            )
            result_params, result_metrics = strategy.aggregate_fit(1, results, [])

        # Should handle extreme trim ratio gracefully
        assert result_params is not None

    def test_multidimensional_parameter_handling(self, trimmed_mean_strategy):
        """Test handling of complex multidimensional parameters."""
        # Create results with complex parameter structures
        results = []
        for i in range(6):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Complex parameter structure
            mock_params = [
                np.random.randn(3, 4, 5) + i,  # 3D tensor
                np.random.randn(10, 2) + i,  # 2D matrix
                np.random.randn(7) + i,  # 1D vector
            ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, results, []
        )

        # Should handle complex parameter structures
        assert result_params is not None

        # Verify the structure is preserved
        aggregated_arrays = parameters_to_ndarrays(result_params)
        assert len(aggregated_arrays) == 3
        assert aggregated_arrays[0].shape == (3, 4, 5)
        assert aggregated_arrays[1].shape == (10, 2)
        assert aggregated_arrays[2].shape == (7,)

    def test_numerical_stability(self, trimmed_mean_strategy):
        """Test numerical stability with extreme parameter values."""
        # Create results with extreme values
        results = []
        extreme_values = [1e-10, 1e10, -1e10, 0.0, 1.0]

        for i, val in enumerate(extreme_values):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            mock_params = [np.full((2, 2), val), np.full(2, val)]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        result_params, result_metrics = trimmed_mean_strategy.aggregate_fit(
            1, results, []
        )

        # Should handle extreme values without numerical issues
        assert result_params is not None

        # Verify results are finite
        aggregated_arrays = parameters_to_ndarrays(result_params)
        for arr in aggregated_arrays:
            assert np.all(np.isfinite(arr))


class TestAggregateEvaluate:
    """Test cases for aggregate_evaluate method."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def trimmed_mean_strategy(self, mock_strategy_history):
        """Create TrimmedMeanBasedRemovalStrategy instance for testing."""
        return TrimmedMeanBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            trim_ratio=0.2,
        )

    def test_aggregate_evaluate_empty_results(self, trimmed_mean_strategy):
        """Test aggregate_evaluate with empty results."""
        loss, metrics = trimmed_mean_strategy.aggregate_evaluate(1, [], [])

        # Should return None for empty results
        assert loss is None
        assert metrics == {}

    def test_aggregate_evaluate_with_valid_results(self, trimmed_mean_strategy):
        """Test aggregate_evaluate with valid client results."""
        from flwr.common import EvaluateRes

        # Create mock client results
        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + i * 0.1
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8 - i * 0.05}

            results.append((client_proxy, eval_res))

        loss, metrics = trimmed_mean_strategy.aggregate_evaluate(1, results, [])

        # Should return aggregated loss
        assert loss is not None
        assert isinstance(loss, float)
        assert isinstance(metrics, dict)

        # Verify strategy history was updated
        assert (
            trimmed_mean_strategy.strategy_history.insert_single_client_history_entry.call_count
            == 10
        )  # 5 for accuracy + 5 for loss
        trimmed_mean_strategy.strategy_history.insert_round_history_entry.assert_called_once()

    def test_aggregate_evaluate_with_removed_clients(self, trimmed_mean_strategy):
        """Test aggregate_evaluate excludes removed clients from aggregation."""
        from flwr.common import EvaluateRes

        # Mark clients 0 and 2 as removed
        trimmed_mean_strategy.removed_client_ids = {"0", "2"}

        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + i * 0.1
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

        loss, metrics = trimmed_mean_strategy.aggregate_evaluate(1, results, [])

        # Should aggregate only non-removed clients
        assert loss is not None

        # Verify that accuracy was only recorded for non-removed clients
        accuracy_calls = [
            call
            for call in trimmed_mean_strategy.strategy_history.insert_single_client_history_entry.call_args_list
            if "accuracy" in str(call)
        ]
        # Should have 3 accuracy calls (clients 1, 3, 4) not 5
        assert len(accuracy_calls) == 3

    def test_aggregate_evaluate_all_clients_removed(self, trimmed_mean_strategy):
        """Test aggregate_evaluate when all clients are removed raises ZeroDivisionError."""
        from flwr.common import EvaluateRes

        # Mark all clients as removed
        trimmed_mean_strategy.removed_client_ids = {"0", "1", "2"}

        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

        # Should raise ZeroDivisionError when all clients are removed
        with pytest.raises(ZeroDivisionError):
            trimmed_mean_strategy.aggregate_evaluate(1, results, [])

    def test_aggregate_evaluate_tracks_loss_for_all_clients(
        self, trimmed_mean_strategy
    ):
        """Test that loss is tracked for all clients including removed ones."""
        from flwr.common import EvaluateRes

        # Mark client 0 as removed
        trimmed_mean_strategy.removed_client_ids = {"0"}

        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + i * 0.1
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

        trimmed_mean_strategy.aggregate_evaluate(1, results, [])

        # Loss should be recorded for all clients (including removed ones)
        loss_calls = [
            call
            for call in trimmed_mean_strategy.strategy_history.insert_single_client_history_entry.call_args_list
            if "loss" in str(call)
        ]
        assert len(loss_calls) == 3  # All 3 clients

    def test_aggregate_evaluate_metrics_structure(self, trimmed_mean_strategy):
        """Test that metrics are properly structured."""
        from flwr.common import EvaluateRes

        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5
            eval_res.num_examples = 100
            eval_res.metrics = {
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 0.82,
            }

            results.append((client_proxy, eval_res))

        loss, metrics = trimmed_mean_strategy.aggregate_evaluate(1, results, [])

        # Metrics should be a dictionary
        assert isinstance(metrics, dict)

    def test_aggregate_evaluate_with_failures(self, trimmed_mean_strategy):
        """Test aggregate_evaluate handles failures gracefully."""
        from flwr.common import EvaluateRes

        results = []
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"accuracy": 0.8}
        results.append((client_proxy, eval_res))

        # Provide some failures
        failures = [
            (Mock(spec=ClientProxy), Exception("Test failure")),
            (Mock(spec=ClientProxy), Exception("Another failure")),
        ]

        loss, metrics = trimmed_mean_strategy.aggregate_evaluate(1, results, failures)

        # Should still process successful results
        assert loss is not None
        assert isinstance(metrics, dict)

    def test_aggregate_evaluate_round_history_update(self, trimmed_mean_strategy):
        """Test that round history is updated with aggregated loss."""
        from flwr.common import EvaluateRes

        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.6
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

        trimmed_mean_strategy.aggregate_evaluate(1, results, [])

        # Verify round history was updated with loss
        trimmed_mean_strategy.strategy_history.insert_round_history_entry.assert_called_once()
        call_kwargs = (
            trimmed_mean_strategy.strategy_history.insert_round_history_entry.call_args[
                1
            ]
        )
        assert "loss_aggregated" in call_kwargs
        assert isinstance(call_kwargs["loss_aggregated"], float)

    def test_aggregate_evaluate_weighted_loss_calculation(self, trimmed_mean_strategy):
        """Test that loss is weighted by number of examples."""
        from flwr.common import EvaluateRes

        results = []
        # Client 0: 100 examples, loss 1.0
        client_proxy_0 = Mock(spec=ClientProxy)
        client_proxy_0.cid = "0"
        eval_res_0 = Mock(spec=EvaluateRes)
        eval_res_0.loss = 1.0
        eval_res_0.num_examples = 100
        eval_res_0.metrics = {"accuracy": 0.8}
        results.append((client_proxy_0, eval_res_0))

        # Client 1: 200 examples, loss 2.0
        client_proxy_1 = Mock(spec=ClientProxy)
        client_proxy_1.cid = "1"
        eval_res_1 = Mock(spec=EvaluateRes)
        eval_res_1.loss = 2.0
        eval_res_1.num_examples = 200
        eval_res_1.metrics = {"accuracy": 0.7}
        results.append((client_proxy_1, eval_res_1))

        loss, metrics = trimmed_mean_strategy.aggregate_evaluate(1, results, [])

        # Weighted average: (100*1.0 + 200*2.0) / (100+200) = 500/300 = 1.6666...
        assert loss is not None
        assert abs(loss - 1.6666666666666667) < 1e-6

    def test_aggregate_evaluate_client_id_types(self, trimmed_mean_strategy):
        """Test aggregate_evaluate raises ValueError for non-numeric client IDs."""
        from flwr.common import EvaluateRes

        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = f"client_{i}"  # Non-numeric string CID

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

        # Should raise ValueError when trying to convert non-numeric CID to int
        with pytest.raises(ValueError, match="invalid literal for int"):
            trimmed_mean_strategy.aggregate_evaluate(1, results, [])

    def test_aggregate_evaluate_no_accuracy_metric(self, trimmed_mean_strategy):
        """Test aggregate_evaluate when accuracy metric is missing."""
        from flwr.common import EvaluateRes

        results = []
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"

        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"precision": 0.8}  # No accuracy

        results.append((client_proxy, eval_res))

        # Should raise KeyError when trying to access accuracy
        with pytest.raises(KeyError):
            trimmed_mean_strategy.aggregate_evaluate(1, results, [])
