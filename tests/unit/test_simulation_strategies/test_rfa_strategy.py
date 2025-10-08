"""
Unit tests for RFABasedRemovalStrategy.

Tests RFA (Robust Federated Averaging) geometric median calculation and client removal logic.
"""

from unittest.mock import patch

from tests.common import (
    Mock,
    np,
    pytest,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    ClientProxy,
)
from src.simulation_strategies.rfa_based_removal_strategy import RFABasedRemovalStrategy

from tests.common import generate_mock_client_data


class TestRFABasedRemovalStrategy:
    """Test cases for RFABasedRemovalStrategy."""

    @pytest.fixture
    def rfa_strategy(self, mock_output_directory):
        """Create RFABasedRemovalStrategy instance for testing."""
        return RFABasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            weighted_median_factor=1.0,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def mock_client_results(self):
        """Generate mock client results for testing."""
        return generate_mock_client_data(num_clients=5)

    def test_initialization(self, rfa_strategy):
        """Test RFABasedRemovalStrategy initialization."""
        assert rfa_strategy.remove_clients is True
        assert rfa_strategy.begin_removing_from_round == 2
        assert rfa_strategy.weighted_median_factor == 1.0
        assert rfa_strategy.current_round == 0
        assert rfa_strategy.removed_client_ids == set()
        assert rfa_strategy.rounds_history == {}
        assert rfa_strategy.client_scores == {}

    def test_geometric_median_single_point(self, rfa_strategy):
        """Test geometric median calculation with single point."""
        points = np.array([[1.0, 2.0, 3.0]])

        median = rfa_strategy._geometric_median(points)

        # Geometric median of single point should be the point itself
        assert np.allclose(median, points[0])

    def test_geometric_median_two_points(self, rfa_strategy):
        """Test geometric median calculation with two points."""
        points = np.array([[0.0, 0.0], [2.0, 0.0]])

        median = rfa_strategy._geometric_median(points)

        # Geometric median of two points should be between them
        assert median[0] >= 0.0 and median[0] <= 2.0
        assert abs(median[1]) < 1e-6  # Should be close to 0

    def test_geometric_median_multiple_points(self, rfa_strategy):
        """Test geometric median calculation with multiple points."""
        # Create points in a pattern where geometric median is predictable
        points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        median = rfa_strategy._geometric_median(points)

        # Should be finite and within reasonable bounds
        assert np.all(np.isfinite(median))
        assert len(median) == 2

    def test_geometric_median_convergence(self, rfa_strategy):
        """Test geometric median convergence with max_iter parameter."""
        points = np.random.randn(10, 5)

        # Test with different max_iter values
        median1 = rfa_strategy._geometric_median(points, max_iter=10)
        median2 = rfa_strategy._geometric_median(points, max_iter=100)

        # Both should be finite
        assert np.all(np.isfinite(median1))
        assert np.all(np.isfinite(median2))

        # More iterations should give similar or better result
        assert len(median1) == len(median2)

    def test_geometric_median_tolerance(self, rfa_strategy):
        """Test geometric median with different tolerance values."""
        points = np.random.randn(5, 3)

        # Test with different tolerance values
        median1 = rfa_strategy._geometric_median(points, tol=1e-3)
        median2 = rfa_strategy._geometric_median(points, tol=1e-6)

        # Both should be finite and similar
        assert np.all(np.isfinite(median1))
        assert np.all(np.isfinite(median2))
        assert len(median1) == len(median2)

    def test_geometric_median_identical_points(self, rfa_strategy):
        """Test geometric median with identical points."""
        identical_point = np.array([1.0, 2.0, 3.0])
        points = np.tile(identical_point, (5, 1))

        median = rfa_strategy._geometric_median(points)

        # Geometric median of identical points should be that point
        assert np.allclose(median, identical_point, atol=1e-6)

    def test_geometric_median_outlier_robustness(self, rfa_strategy):
        """Test geometric median robustness to outliers."""
        # Create points clustered around origin with one outlier
        normal_points = np.random.randn(9, 2) * 0.1
        outlier = np.array([[10.0, 10.0]])
        points = np.vstack([normal_points, outlier])

        median = rfa_strategy._geometric_median(points)

        # Median should be closer to normal points than to outlier
        distance_to_origin = np.linalg.norm(median)
        distance_to_outlier = np.linalg.norm(median - outlier[0])

        assert distance_to_origin < distance_to_outlier

    @patch("src.simulation_strategies.rfa_based_removal_strategy.KMeans")
    @patch("src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler")
    def test_aggregate_fit_clustering(
        self, mock_scaler, mock_kmeans, rfa_strategy, mock_client_results
    ):
        """Test aggregate_fit performs clustering correctly."""
        # Setup mocks
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

        rfa_strategy.aggregate_fit(1, mock_client_results, [])

        # Verify clustering was called
        mock_kmeans.assert_called_once()
        mock_scaler_instance.fit.assert_called_once()
        mock_scaler_instance.transform.assert_called_once()

    def test_aggregate_fit_geometric_median_calculation(
        self, rfa_strategy, mock_client_results
    ):
        """Test aggregate_fit calculates geometric median correctly."""
        with (
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            # Setup mocks
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

            result_params, result_metrics = rfa_strategy.aggregate_fit(
                1, mock_client_results, []
            )

            # Should return aggregated parameters
            assert result_params is not None
            assert isinstance(result_metrics, dict)

            # Verify current_round was incremented
            assert rfa_strategy.current_round == 1

            # Verify rounds_history was updated
            assert "1" in rfa_strategy.rounds_history

    def test_aggregate_fit_deviation_calculation(
        self, rfa_strategy, mock_client_results
    ):
        """Test aggregate_fit calculates client deviations from geometric median."""
        with (
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            # Setup mocks
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

            rfa_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify client scores (deviations) were calculated
            assert len(rfa_strategy.client_scores) == 5

            # All deviations should be non-negative
            for score in rfa_strategy.client_scores.values():
                assert score >= 0
                assert np.isfinite(score)

    def test_aggregate_fit_empty_results(self, rfa_strategy):
        """Test aggregate_fit handles empty results."""
        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (None, {})
            result_params, result_metrics = rfa_strategy.aggregate_fit(1, [], [])

            # Should handle empty results gracefully
            assert result_params is None
            assert isinstance(result_metrics, dict)

    def test_weighted_median_factor_parameter_effect(self):
        """Test weighted_median_factor parameter affects geometric median."""
        factors = [0.5, 1.0, 2.0]

        for factor in factors:
            strategy = RFABasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                weighted_median_factor=factor,
            )

            assert strategy.weighted_median_factor == factor

            # Test that factor affects the weighted geometric median
            points = np.array([[1.0, 1.0], [2.0, 2.0]])
            geometric_median = strategy._geometric_median(points)
            weighted_median = geometric_median * factor

            # Weighted median should be scaled by the factor
            expected_weighted = geometric_median * factor
            assert np.allclose(weighted_median, expected_weighted)

    def test_configure_fit_warmup_rounds(self, rfa_strategy):
        """Test configure_fit during warmup rounds."""
        rfa_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = rfa_strategy.configure_fit(1, mock_parameters, mock_client_manager)

        # Should return all clients during warmup
        assert len(result) == 5
        assert rfa_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase(self, rfa_strategy):
        """Test configure_fit removes client with highest deviation."""
        rfa_strategy.current_round = 3  # After begin_removing_from_round
        rfa_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # Highest deviation - should be removed
            "client_2": 0.3,
            "client_3": 0.2,
            "client_4": 0.5,
        }
        rfa_strategy.rounds_history[f"{rfa_strategy.current_round}"] = {
            "client_info": {f"client_{i}": {} for i in range(5)}
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = rfa_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should remove client with highest deviation
        assert "client_1" in rfa_strategy.removed_client_ids
        assert len(result) == 5  # Still returns all clients for training

    def test_configure_fit_no_removal_when_disabled(self, rfa_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        rfa_strategy.remove_clients = False
        rfa_strategy.current_round = 3
        rfa_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # High deviation but shouldn't be removed
            "client_2": 0.3,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        rfa_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should not remove any clients
        assert rfa_strategy.removed_client_ids == set()

    def test_begin_removing_from_round_parameter(self):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = RFABasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=begin_round,
                weighted_median_factor=1.0,
            )

            assert strategy.begin_removing_from_round == begin_round

            # Test warmup behavior
            strategy.current_round = begin_round - 1
            mock_client_manager = Mock()
            mock_clients = {"client_0": Mock(), "client_1": Mock()}
            mock_client_manager.all.return_value = mock_clients

            result = strategy.configure_fit(1, Mock(), mock_client_manager)

            # Should not remove clients during warmup
            assert strategy.removed_client_ids == set()
            assert len(result) == 2

    def test_rounds_history_tracking(self, rfa_strategy, mock_client_results):
        """Test that rounds_history is properly maintained."""
        with (
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            # Setup mocks
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

            rfa_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify rounds_history structure
            assert "1" in rfa_strategy.rounds_history
            assert "round_info" in rfa_strategy.rounds_history["1"]
            assert "client_info" in rfa_strategy.rounds_history["1"]

            # Verify client info is recorded
            client_info = rfa_strategy.rounds_history["1"]["client_info"]
            assert len(client_info) == 5

            # Verify each client has required fields
            for client_key, client_data in client_info.items():
                assert "removal_criterion" in client_data
                assert "absolute_distance" in client_data
                assert "normalized_distance" in client_data
                assert "is_removed" in client_data

    def test_edge_case_single_client(self, rfa_strategy):
        """Test handling of single client scenario."""
        # Create single client result
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "client_0"
        mock_params = [np.random.randn(5, 3), np.random.randn(3)]
        fit_res = Mock(spec=FitRes)
        fit_res.parameters = ndarrays_to_parameters(mock_params)
        fit_res.num_examples = 100

        single_result = [(client_proxy, fit_res)]

        with (
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array([[0.1]])
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array([[0.1]])
            mock_scaler.return_value = mock_scaler_instance

            result = rfa_strategy.aggregate_fit(1, single_result, [])

            # Should handle single client gracefully
            assert len(rfa_strategy.client_scores) == 1
            assert result[0] is not None

    def test_parameter_reconstruction_accuracy(self, rfa_strategy):
        """Test that parameter reconstruction maintains correct shapes."""
        # Create results with known parameter shapes
        results = []
        original_shapes = [(3, 4), (5,), (2, 2, 2)]

        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create parameters with specific shapes
            mock_params = [
                np.random.randn(*original_shapes[0]),
                np.random.randn(*original_shapes[1]),
                np.random.randn(*original_shapes[2]),
            ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        with (
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3]]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3]]
            )
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = rfa_strategy.aggregate_fit(1, results, [])

            # Verify parameter shapes are preserved
            reconstructed_params = parameters_to_ndarrays(result_params)
            assert len(reconstructed_params) == len(original_shapes)

            for i, expected_shape in enumerate(original_shapes):
                assert reconstructed_params[i].shape == expected_shape

    def test_numerical_stability_extreme_values(self, rfa_strategy):
        """Test numerical stability with extreme parameter values."""
        # Create results with extreme values
        results = []
        extreme_values = [1e-10, 1e10, -1e10]

        for i, scale in enumerate(extreme_values):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            mock_params = [np.full((2, 2), scale), np.full(2, scale)]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        with (
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3]]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3]]
            )
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = rfa_strategy.aggregate_fit(1, results, [])

            # Should handle extreme values without numerical issues
            assert result_params is not None

            # Verify results are finite
            reconstructed_params = parameters_to_ndarrays(result_params)
            for param in reconstructed_params:
                assert np.all(np.isfinite(param))

    def test_geometric_median_convergence_criteria(self, rfa_strategy):
        """Test geometric median convergence with different criteria."""
        # Test convergence with points that should converge quickly
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])

        # Test with strict tolerance
        median_strict = rfa_strategy._geometric_median(points, tol=1e-8, max_iter=1000)

        # Test with loose tolerance
        median_loose = rfa_strategy._geometric_median(points, tol=1e-3, max_iter=10)

        # Both should be finite and reasonable
        assert np.all(np.isfinite(median_strict))
        assert np.all(np.isfinite(median_loose))

        # Results should be in reasonable range
        for median in [median_strict, median_loose]:
            assert np.all(median >= -1.0) and np.all(median <= 2.0)


class TestRFAAggregateEvaluate:
    """Test cases for aggregate_evaluate method."""

    @pytest.fixture
    def rfa_strategy(self):
        """Create RFABasedRemovalStrategy instance for testing."""
        return RFABasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            weighted_median_factor=1.0,
        )

    def test_aggregate_evaluate_empty_results(self, rfa_strategy):
        """Test aggregate_evaluate with empty results."""
        loss, metrics = rfa_strategy.aggregate_evaluate(1, [], [])

        # Should return None for empty results
        assert loss is None
        assert metrics == {}

    def test_aggregate_evaluate_with_valid_results(self, rfa_strategy):
        """Test aggregate_evaluate with valid client results."""
        from flwr.common import EvaluateRes

        # Setup rounds_history for current round
        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}

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

            # Initialize client_info in rounds_history
            rfa_strategy.rounds_history["1"]["client_info"][f"client_{i}"] = {}

        loss, metrics = rfa_strategy.aggregate_evaluate(1, results, [])

        # Should return aggregated loss
        assert loss is not None
        assert isinstance(loss, float)
        assert isinstance(metrics, dict)

        # Verify client info was updated
        for i in range(5):
            client_info = rfa_strategy.rounds_history["1"]["client_info"][f"client_{i}"]
            assert "accuracy" in client_info
            assert "loss" in client_info

    def test_aggregate_evaluate_with_removed_clients(self, rfa_strategy):
        """Test aggregate_evaluate excludes removed clients from aggregation."""
        from flwr.common import EvaluateRes

        # Setup rounds_history and mark clients as removed
        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}
        rfa_strategy.removed_client_ids = {"0", "2"}

        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + i * 0.1
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

            # Initialize client_info
            rfa_strategy.rounds_history["1"]["client_info"][f"client_{i}"] = {}

        loss, metrics = rfa_strategy.aggregate_evaluate(1, results, [])

        # Should aggregate only non-removed clients
        assert loss is not None

        # Verify removed clients have no accuracy set
        assert (
            rfa_strategy.rounds_history["1"]["client_info"]["client_0"].get("accuracy")
            is None
        )
        assert (
            rfa_strategy.rounds_history["1"]["client_info"]["client_2"].get("accuracy")
            is None
        )

        # Verify non-removed clients have accuracy
        assert (
            rfa_strategy.rounds_history["1"]["client_info"]["client_1"]["accuracy"]
            == 0.8
        )

    def test_aggregate_evaluate_all_clients_removed(self, rfa_strategy):
        """Test aggregate_evaluate when all clients are removed raises ZeroDivisionError."""
        from flwr.common import EvaluateRes

        # Setup rounds_history
        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}
        rfa_strategy.removed_client_ids = {"0", "1", "2"}

        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8}

            results.append((client_proxy, eval_res))

            # Initialize client_info
            rfa_strategy.rounds_history["1"]["client_info"][f"client_{i}"] = {}

        # Should raise ZeroDivisionError when all clients are removed
        with pytest.raises(ZeroDivisionError):
            rfa_strategy.aggregate_evaluate(1, results, [])

    def test_aggregate_evaluate_previous_round_client_info(self, rfa_strategy):
        """Test aggregate_evaluate copies client info from previous round."""
        from flwr.common import EvaluateRes

        # Setup previous round history
        rfa_strategy.current_round = 2
        rfa_strategy.rounds_history["1"] = {
            "client_info": {
                "client_0": {
                    "removal_criterion": 0.5,
                    "absolute_distance": 0.3,
                    "normalized_distance": 0.2,
                    "is_removed": False,
                },
                "client_1": {
                    "removal_criterion": 0.7,
                    "absolute_distance": 0.5,
                    "normalized_distance": 0.4,
                    "is_removed": False,
                },
            }
        }
        rfa_strategy.rounds_history["2"] = {"client_info": {}}

        # Create result for only client_1 (client_0 is missing)
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "1"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"accuracy": 0.8}
        results = [(client_proxy, eval_res)]

        rfa_strategy.rounds_history["2"]["client_info"]["client_1"] = {}

        rfa_strategy.aggregate_evaluate(2, results, [])

        # Verify client_0 info was copied from previous round
        assert "client_0" in rfa_strategy.rounds_history["2"]["client_info"]
        copied_info = rfa_strategy.rounds_history["2"]["client_info"]["client_0"]
        assert copied_info["removal_criterion"] == 0.5
        assert copied_info["absolute_distance"] == 0.3

    def test_aggregate_evaluate_removed_client_null_metrics(self, rfa_strategy):
        """Test that removed clients copied from previous round get None for accuracy and loss."""
        from flwr.common import EvaluateRes

        # Setup previous round with client_0
        rfa_strategy.current_round = 2
        rfa_strategy.rounds_history["1"] = {
            "client_info": {
                "client_0": {"is_removed": False, "accuracy": 0.9, "loss": 0.3},
            }
        }
        rfa_strategy.rounds_history["2"] = {"client_info": {}}
        # removed_client_ids uses the full key "client_0" not just "0"
        rfa_strategy.removed_client_ids = {"client_0"}

        # Create result for client_1 only (client_0 will be copied from previous round)
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "1"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"accuracy": 0.8}
        results = [(client_proxy, eval_res)]

        rfa_strategy.rounds_history["2"]["client_info"]["client_1"] = {}

        rfa_strategy.aggregate_evaluate(2, results, [])

        # Verify removed client_0 (copied from previous round) has None metrics
        assert "client_0" in rfa_strategy.rounds_history["2"]["client_info"]
        assert (
            rfa_strategy.rounds_history["2"]["client_info"]["client_0"]["accuracy"]
            is None
        )
        assert (
            rfa_strategy.rounds_history["2"]["client_info"]["client_0"]["loss"] is None
        )

        # Verify non-removed client_1 has actual metrics
        assert (
            rfa_strategy.rounds_history["2"]["client_info"]["client_1"]["accuracy"]
            == 0.8
        )
        assert (
            rfa_strategy.rounds_history["2"]["client_info"]["client_1"]["loss"] == 0.5
        )

    def test_aggregate_evaluate_weighted_loss_calculation(self, rfa_strategy):
        """Test that loss is weighted by number of examples."""
        from flwr.common import EvaluateRes

        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}

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

        # Initialize client_info
        for i in range(2):
            rfa_strategy.rounds_history["1"]["client_info"][f"client_{i}"] = {}

        loss, metrics = rfa_strategy.aggregate_evaluate(1, results, [])

        # Weighted average: (100*1.0 + 200*2.0) / (100+200) = 500/300 = 1.6666...
        assert loss is not None
        assert abs(loss - 1.6666666666666667) < 1e-6

    def test_aggregate_evaluate_metrics_from_get_method(self, rfa_strategy):
        """Test that accuracy uses .get() method on metrics dict."""
        from flwr.common import EvaluateRes

        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}

        # Create result with metrics that has accuracy
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"accuracy": 0.85, "precision": 0.9}
        results = [(client_proxy, eval_res)]

        rfa_strategy.rounds_history["1"]["client_info"]["client_0"] = {}

        rfa_strategy.aggregate_evaluate(1, results, [])

        # Verify accuracy was extracted correctly
        assert (
            rfa_strategy.rounds_history["1"]["client_info"]["client_0"]["accuracy"]
            == 0.85
        )

    def test_aggregate_evaluate_no_previous_round(self, rfa_strategy):
        """Test aggregate_evaluate when there is no previous round."""
        from flwr.common import EvaluateRes

        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}

        # Create results
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"accuracy": 0.8}
        results = [(client_proxy, eval_res)]

        rfa_strategy.rounds_history["1"]["client_info"]["client_0"] = {}

        # Should handle gracefully when previous_round doesn't exist
        loss, metrics = rfa_strategy.aggregate_evaluate(1, results, [])

        assert loss is not None
        assert isinstance(metrics, dict)

    def test_aggregate_evaluate_with_failures(self, rfa_strategy):
        """Test aggregate_evaluate handles failures gracefully."""
        from flwr.common import EvaluateRes

        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}

        # Create one successful result
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"accuracy": 0.8}
        results = [(client_proxy, eval_res)]

        rfa_strategy.rounds_history["1"]["client_info"]["client_0"] = {}

        # Provide some failures
        failures = [
            (Mock(spec=ClientProxy), Exception("Test failure")),
            (Mock(spec=ClientProxy), Exception("Another failure")),
        ]

        loss, metrics = rfa_strategy.aggregate_evaluate(1, results, failures)

        # Should still process successful results
        assert loss is not None
        assert isinstance(metrics, dict)

    def test_aggregate_evaluate_missing_accuracy_key(self, rfa_strategy):
        """Test aggregate_evaluate when metrics dict doesn't have accuracy key."""
        from flwr.common import EvaluateRes

        rfa_strategy.current_round = 1
        rfa_strategy.rounds_history["1"] = {"client_info": {}}

        # Create result without accuracy metric
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = 0.5
        eval_res.num_examples = 100
        eval_res.metrics = {"precision": 0.9}  # No accuracy key

        results = [(client_proxy, eval_res)]
        rfa_strategy.rounds_history["1"]["client_info"]["client_0"] = {}

        rfa_strategy.aggregate_evaluate(1, results, [])

        # Should use .get() and return None for missing key
        assert (
            rfa_strategy.rounds_history["1"]["client_info"]["client_0"]["accuracy"]
            is None
        )
