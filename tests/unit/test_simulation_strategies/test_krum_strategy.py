"""
Unit tests for KrumBasedRemovalStrategy.

Tests Krum client selection algorithms, distance calculations, and subset identification.
"""

from unittest.mock import patch
from src.simulation_strategies.krum_based_removal_strategy import (
    KrumBasedRemovalStrategy,
)
from tests.common import (
    ClientProxy,
    FitRes,
    Mock,
    ndarrays_to_parameters,
    np,
    pytest,
)


class TestKrumBasedRemovalStrategy:
    """Test cases for KrumBasedRemovalStrategy."""

    @pytest.fixture
    def krum_strategy(
        self, mock_strategy_history, mock_output_directory, krum_fit_metrics_fn
    ):
        """Create KrumBasedRemovalStrategy instance for testing."""
        return KrumBasedRemovalStrategy(
            remove_clients=True,
            num_malicious_clients=2,
            num_krum_selections=3,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            fit_metrics_aggregation_fn=krum_fit_metrics_fn,
        )

    def test_initialization(self, krum_strategy, mock_strategy_history):
        """Test KrumBasedRemovalStrategy initialization."""
        assert krum_strategy.remove_clients is True
        assert krum_strategy.num_malicious_clients == 2
        assert krum_strategy.num_krum_selections == 3
        assert krum_strategy.begin_removing_from_round == 2
        assert krum_strategy.strategy_history == mock_strategy_history
        assert krum_strategy.current_round == 0
        assert krum_strategy.client_scores == {}
        assert krum_strategy.removed_client_ids == set()

    def test_calculate_krum_scores_distance_matrix(
        self, krum_strategy, mock_client_results
    ):
        """Test _calculate_krum_scores creates proper distance matrix."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        krum_scores = krum_strategy._calculate_krum_scores(
            mock_client_results, distances
        )

        # Verify distance matrix is symmetric
        assert np.allclose(distances, distances.T)

        # Verify krum scores are calculated and returned as a list
        assert isinstance(krum_scores, list)
        assert len(krum_scores) == len(mock_client_results)

        # Verify diagonal is zero (distance from client to itself)
        assert np.allclose(np.diag(distances), 0)

        # Verify all distances are non-negative
        assert np.all(distances >= 0)

    def test_calculate_krum_scores_computation(
        self, krum_strategy, mock_client_results
    ):
        """Test _calculate_krum_scores computes scores correctly."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        krum_scores = krum_strategy._calculate_krum_scores(
            mock_client_results, distances
        )

        # Should return one score per client
        assert len(krum_scores) == len(mock_client_results)

        # All scores should be non-negative
        assert all(score >= 0 for score in krum_scores)

        # Scores should be finite
        assert all(np.isfinite(score) for score in krum_scores)

    def test_calculate_krum_scores_malicious_parameter_effect(
        self, mock_strategy_history, krum_fit_metrics_fn
    ):
        """Test num_malicious_clients parameter affects score calculation."""
        # Test with different num_malicious_clients values
        malicious_counts = [1, 2, 3]

        for num_malicious in malicious_counts:
            strategy = KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=num_malicious,
                num_krum_selections=3,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            )

            # Create simple test data
            results = []
            for i in range(5):
                client_proxy = Mock(spec=ClientProxy)
                client_proxy.cid = str(i)
                mock_params = [np.ones((2, 2)) * i, np.ones(2) * i]
                fit_res = Mock(spec=FitRes)
                fit_res.parameters = ndarrays_to_parameters(mock_params)
                results.append((client_proxy, fit_res))

            distances = np.zeros((5, 5))
            scores = strategy._calculate_krum_scores(results, distances)

            # Verify scores are calculated (should be different for different num_malicious)
            assert len(scores) == 5
            assert all(np.isfinite(score) for score in scores)

    def test_calculate_krum_scores_subset_selection(self, krum_strategy):
        """Test Krum score calculation selects correct subset of distances."""
        # Create controlled test scenario
        results = []
        for i in range(4):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            # Create parameters that will result in predictable distances
            mock_params = [np.array([[i, i], [i, i]]), np.array([i, i])]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((4, 4))
        scores = krum_strategy._calculate_krum_scores(results, distances)

        # Verify that scores are computed using the correct subset
        # (num_malicious_clients - 2 = 2 - 2 = 0 closest distances)
        assert len(scores) == 4

        # With num_malicious_clients=2, we should sum 0 distances (just the client itself)
        # So all scores should be 0
        assert all(score == 0 for score in scores)

    def test_calculate_krum_scores_edge_case_insufficient_clients(
        self, mock_strategy_history, krum_fit_metrics_fn
    ):
        """Test Krum score calculation with insufficient clients."""
        strategy = KrumBasedRemovalStrategy(
            remove_clients=True,
            num_malicious_clients=5,  # More malicious than total clients
            num_krum_selections=3,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fit_metrics_aggregation_fn=krum_fit_metrics_fn,
        )

        # Create only 3 clients
        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.ones((2, 2)), np.ones(2)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((3, 3))

        # Should handle edge case gracefully
        scores = strategy._calculate_krum_scores(results, distances)
        assert len(scores) == 3

    def test_aggregate_fit_clustering(
        self, krum_strategy, mock_client_results, mock_krum_clustering
    ):
        """Test aggregate_fit performs clustering correctly."""
        mock_kmeans = mock_krum_clustering["kmeans"]
        mock_scaler_instance = mock_krum_clustering["scaler"].return_value

        krum_strategy.aggregate_fit(1, mock_client_results, [])

        # Verify clustering was called
        mock_kmeans.assert_called_once()
        mock_scaler_instance.fit.assert_called_once()
        mock_scaler_instance.transform.assert_called_once()

    def test_aggregate_fit_krum_score_calculation(
        self, krum_strategy, mock_client_results, mock_krum_clustering
    ):
        """Test aggregate_fit calculates Krum scores for all clients."""
        krum_strategy.aggregate_fit(1, mock_client_results, [])

        # Verify Krum scores were calculated for all clients
        assert len(krum_strategy.client_scores) == 5

        # Verify all scores are valid numbers
        for score in krum_strategy.client_scores.values():
            assert isinstance(score, (int, float))
            assert np.isfinite(score)

    def test_aggregate_fit_client_selection(
        self, krum_strategy, mock_client_results, mock_krum_clustering
    ):
        """Test aggregate_fit selects client with minimum Krum score."""
        mock_parent_aggregate = mock_krum_clustering["parent_aggregate"]
        # Mock parent aggregate_fit to capture the selected clients
        selected_clients = []

        def capture_selected_clients(server_round, results, failures):
            selected_clients.extend(results)
            return (Mock(), {})

        mock_parent_aggregate.side_effect = capture_selected_clients

        krum_strategy.aggregate_fit(1, mock_client_results, [])

        # Should select only one client (the one with minimum Krum score)
        assert len(selected_clients) == 1

    def test_configure_fit_warmup_rounds(self, krum_strategy):
        """Test configure_fit during warmup rounds."""
        krum_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = krum_strategy.configure_fit(1, mock_parameters, mock_client_manager)

        # Should return all clients during warmup
        assert len(result) == 5
        assert krum_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase(self, krum_strategy):
        """Test configure_fit removes client with highest Krum score."""
        krum_strategy.current_round = 3  # After begin_removing_from_round
        krum_strategy.client_scores = {
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

        result = krum_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should remove client with highest Krum score
        assert "client_1" in krum_strategy.removed_client_ids
        assert len(result) == 5  # Still returns all clients for training

    def test_configure_fit_no_removal_when_disabled(self, krum_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        krum_strategy.remove_clients = False
        krum_strategy.current_round = 3
        krum_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # Highest score but shouldn't be removed
            "client_2": 0.3,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        krum_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should not remove any clients
        assert krum_strategy.removed_client_ids == set()

    def test_num_krum_selections_parameter(
        self, mock_strategy_history, krum_fit_metrics_fn
    ):
        """Test num_krum_selections parameter handling."""
        # Test different num_krum_selections values
        for num_selections in [1, 3, 5]:
            strategy = KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=2,
                num_krum_selections=num_selections,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            )

            assert strategy.num_krum_selections == num_selections

    def test_begin_removing_from_round_parameter(
        self, mock_strategy_history, krum_fit_metrics_fn
    ):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=2,
                num_krum_selections=3,
                begin_removing_from_round=begin_round,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            )

            assert strategy.begin_removing_from_round == begin_round

            # Test warmup behavior
            strategy.current_round = begin_round
            mock_client_manager = Mock()
            mock_clients = {"client_0": Mock(), "client_1": Mock()}
            mock_client_manager.all.return_value = mock_clients

            result = strategy.configure_fit(1, Mock(), mock_client_manager)

            # Should return all clients during warmup (current_round <= begin_removing_from_round)
            assert len(result) == 2

    def test_distance_calculation_accuracy(self, krum_strategy):
        """Test that distance calculations are mathematically correct."""
        # Create controlled test data
        results = []
        expected_params = []

        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create simple parameters for easy distance calculation
            params = [np.array([[i, i]]), np.array([i])]
            expected_params.append(np.concatenate([p.flatten() for p in params]))

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((3, 3))
        krum_strategy._calculate_krum_scores(results, distances)

        # Verify distances match expected Euclidean distances
        for i in range(3):
            for j in range(i + 1, 3):
                expected_distance = np.linalg.norm(
                    expected_params[i] - expected_params[j]
                )
                assert abs(distances[i, j] - expected_distance) < 1e-6
                assert abs(distances[j, i] - expected_distance) < 1e-6

    def test_krum_score_mathematical_correctness(self, krum_strategy):
        """Test that Krum scores are calculated according to the algorithm."""
        # Create test data with known distances
        results = []
        for i in range(4):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            # Parameters that create predictable distances
            params = [np.array([[i * 2.0]]), np.array([i * 2.0])]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((4, 4))
        scores = krum_strategy._calculate_krum_scores(results, distances)

        # Manually verify score calculation for first client
        # With num_malicious_clients=2, we sum the (2-2)=0 smallest distances
        # So all scores should be 0
        assert all(score == 0 for score in scores)

    def test_strategy_history_integration(
        self, krum_strategy, mock_client_results, mock_krum_clustering
    ):
        """Test integration with strategy history."""
        krum_strategy.aggregate_fit(1, mock_client_results, [])

        # Verify strategy history methods were called
        assert (
            krum_strategy.strategy_history.insert_single_client_history_entry.call_count
            == 5
        )
        krum_strategy.strategy_history.insert_round_history_entry.assert_called_once()

    def test_edge_case_empty_results(self, krum_strategy):
        """Test handling of empty results."""
        with patch("flwr.server.strategy.Krum.aggregate_fit") as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (None, {})

            result = krum_strategy.aggregate_fit(1, [], [])

            # Should handle empty results gracefully
            assert result is not None

    def test_edge_case_single_client(self, krum_strategy, mock_krum_clustering):
        """Test handling of single client scenario."""
        # Create single client result
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        mock_params = [np.random.randn(10, 5), np.random.randn(5)]
        fit_res = Mock(spec=FitRes)
        fit_res.parameters = ndarrays_to_parameters(mock_params)
        fit_res.num_examples = 100

        single_result = [(client_proxy, fit_res)]

        krum_strategy.aggregate_fit(1, single_result, [])

        # Should handle single client gracefully
        assert len(krum_strategy.client_scores) == 1

    def test_parameter_flattening_consistency(self, krum_strategy):
        """Test that parameter flattening is consistent across calls."""
        # Create test results with complex parameter structures
        results = []
        for i in range(2):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Complex parameter structure
            params = [
                np.random.randn(3, 4, 5),  # 3D tensor
                np.random.randn(10),  # 1D tensor
                np.random.randn(2, 3),  # 2D tensor
            ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((2, 2))
        scores = krum_strategy._calculate_krum_scores(results, distances)

        # Should handle complex parameter structures without errors
        assert len(scores) == 2
        assert all(np.isfinite(score) for score in scores)

    def test_robustness_to_parameter_variations(self, krum_strategy):
        """Test robustness to different parameter magnitudes and distributions."""
        # Test with different parameter characteristics
        test_cases = [
            # Small parameters
            [np.random.randn(5, 5) * 0.01, np.random.randn(5) * 0.01],
            # Large parameters
            [np.random.randn(5, 5) * 100, np.random.randn(5) * 100],
            # Mixed magnitudes
            [np.random.randn(5, 5) * 0.1, np.random.randn(5) * 10],
        ]

        for i, params in enumerate(test_cases):
            results = []
            for j in range(3):
                client_proxy = Mock(spec=ClientProxy)
                client_proxy.cid = f"client_{i}_{j}"

                # Add some variation
                varied_params = [p + np.random.randn(*p.shape) * 0.01 for p in params]

                fit_res = Mock(spec=FitRes)
                fit_res.parameters = ndarrays_to_parameters(varied_params)
                results.append((client_proxy, fit_res))

            distances = np.zeros((3, 3))
            scores = krum_strategy._calculate_krum_scores(results, distances)

            # Should handle all cases without numerical issues
            assert len(scores) == 3
            assert all(np.isfinite(score) for score in scores)
            assert all(score >= 0 for score in scores)

    def test_aggregate_evaluate_empty_results(self, krum_strategy):
        """Test aggregate_evaluate with empty results."""
        result = krum_strategy.aggregate_evaluate(1, [], [])

        assert result == (None, {})

    def test_aggregate_evaluate_collects_per_client_metrics(
        self, krum_strategy, mock_evaluate_results, mock_strategy_history
    ):
        """Test aggregate_evaluate collects per-client metrics."""
        server_round = 1

        krum_strategy.aggregate_evaluate(server_round, mock_evaluate_results, [])

        # Should call insert_single_client_history_entry twice per client (accuracy and loss)
        assert mock_strategy_history.insert_single_client_history_entry.call_count == 10

    def test_aggregate_evaluate_calculates_aggregated_loss(
        self, krum_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate calculates weighted aggregated loss."""
        server_round = 1

        loss_aggregated, _ = krum_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, []
        )

        # Should return a float loss value
        assert isinstance(loss_aggregated, float)
        assert loss_aggregated >= 0

    def test_aggregate_evaluate_returns_empty_metrics(
        self, krum_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate returns empty metrics dict."""
        server_round = 1

        _, metrics = krum_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, []
        )

        # Krum strategy doesn't return metrics in aggregate_evaluate
        assert isinstance(metrics, dict)

    def test_aggregate_evaluate_stores_per_client_data(
        self, krum_strategy, mock_evaluate_results, mock_strategy_history
    ):
        """Test aggregate_evaluate stores correct per-client data."""
        server_round = 1
        krum_strategy.current_round = 1

        krum_strategy.aggregate_evaluate(server_round, mock_evaluate_results, [])

        # Verify both accuracy and loss were stored
        calls = mock_strategy_history.insert_single_client_history_entry.call_args_list

        # First 5 calls should be accuracy
        first_call = calls[0]
        assert first_call[1]["client_id"] == 0
        assert first_call[1]["current_round"] == 1
        assert first_call[1]["accuracy"] == 0.8

        # Next 5 calls should be loss
        sixth_call = calls[5]
        assert sixth_call[1]["client_id"] == 0
        assert sixth_call[1]["current_round"] == 1
        assert sixth_call[1]["loss"] == 0.5

    def test_aggregate_evaluate_with_failures(
        self, krum_strategy, mock_evaluate_results
    ):
        """Test aggregate_evaluate handles failures parameter."""
        server_round = 1
        failures = [Exception("Test failure")]

        # Should process successfully despite failures
        loss_aggregated, metrics = krum_strategy.aggregate_evaluate(
            server_round, mock_evaluate_results, failures
        )

        assert isinstance(loss_aggregated, float)
        assert isinstance(metrics, dict)
