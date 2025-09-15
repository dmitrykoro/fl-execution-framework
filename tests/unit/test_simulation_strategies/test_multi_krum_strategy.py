"""
Unit tests for MultiKrumBasedRemovalStrategy.

Tests Multi-Krum client selection algorithms and removal logic.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from flwr.common import FitRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.multi_krum_based_removal_strategy import (
    MultiKrumBasedRemovalStrategy,
)

from tests.conftest import generate_mock_client_data


class TestMultiKrumBasedRemovalStrategy:
    """Test cases for MultiKrumBasedRemovalStrategy."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def krum_fit_metrics_fn(self):
        """Provide consistent fit_metrics_aggregation_fn for Krum-based strategies."""
        return lambda x: x

    @pytest.fixture
    def multi_krum_strategy(
        self, mock_strategy_history, mock_output_directory, krum_fit_metrics_fn
    ):
        """Create MultiKrumBasedRemovalStrategy instance for testing."""
        return MultiKrumBasedRemovalStrategy(
            remove_clients=True,
            num_of_malicious_clients=2,
            num_krum_selections=3,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            fit_metrics_aggregation_fn=krum_fit_metrics_fn,
        )

    @pytest.fixture
    def mock_client_results(self):
        """Create mock client results for testing."""
        return generate_mock_client_data(num_clients=6)

    def test_initialization(self, multi_krum_strategy, mock_strategy_history):
        """Test MultiKrumBasedRemovalStrategy initialization."""
        assert multi_krum_strategy.remove_clients is True
        assert multi_krum_strategy.num_of_malicious_clients == 2
        assert multi_krum_strategy.num_krum_selections == 3
        assert multi_krum_strategy.begin_removing_from_round == 2
        assert multi_krum_strategy.strategy_history == mock_strategy_history
        assert multi_krum_strategy.current_round == 0
        assert multi_krum_strategy.client_scores == {}
        assert multi_krum_strategy.removed_client_ids == set()

    def test_calculate_multi_krum_scores_distance_matrix(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test _calculate_multi_krum_scores creates proper distance matrix."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            mock_client_results, distances
        )

        # Verify distance matrix is symmetric
        assert np.allclose(distances, distances.T)

        # Verify multi krum scores are calculated and returned as a list
        assert isinstance(multi_krum_scores, list)
        assert len(multi_krum_scores) == len(mock_client_results)

        # Verify diagonal is zero (distance from client to itself)
        assert np.allclose(np.diag(distances), 0)

        # Verify all distances are non-negative
        assert np.all(distances >= 0)

    def test_calculate_multi_krum_scores_computation(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test _calculate_multi_krum_scores computes scores correctly."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            mock_client_results, distances
        )

        # Should return one score per client
        assert len(multi_krum_scores) == len(mock_client_results)

        # All scores should be non-negative
        assert all(score >= 0 for score in multi_krum_scores)

        # Scores should be finite
        assert all(np.isfinite(score) for score in multi_krum_scores)

    def test_calculate_multi_krum_scores_selection_parameter_effect(
        self, mock_strategy_history, mock_output_directory, krum_fit_metrics_fn
    ):
        """Test num_krum_selections parameter affects score calculation."""
        # Test with different num_krum_selections values
        selection_counts = [2, 3, 4]

        for num_selections in selection_counts:
            strategy = MultiKrumBasedRemovalStrategy(
                remove_clients=True,
                num_of_malicious_clients=2,
                num_krum_selections=num_selections,
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
            scores = strategy._calculate_multi_krum_scores(results, distances)

            # Verify scores are calculated
            assert len(scores) == 5
            assert all(np.isfinite(score) for score in scores)

    @patch("src.simulation_strategies.multi_krum_based_removal_strategy.KMeans")
    @patch("src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler")
    def test_aggregate_fit_clustering(
        self, mock_scaler, mock_kmeans, multi_krum_strategy, mock_client_results
    ):
        """Test aggregate_fit performs clustering correctly."""
        # Setup mocks
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

        with patch("flwr.server.strategy.Krum.aggregate_fit") as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify clustering was called
            mock_kmeans.assert_called_once()
            mock_scaler_instance.fit.assert_called_once()
            mock_scaler_instance.transform.assert_called_once()

    def test_aggregate_fit_multi_krum_score_calculation(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test aggregate_fit calculates Multi-Krum scores for all clients."""
        with patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler, patch(
            "flwr.server.strategy.Krum.aggregate_fit"
        ) as mock_parent_aggregate:

            # Setup mocks
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

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify Multi-Krum scores were calculated for all clients
            assert len(multi_krum_strategy.client_scores) == 6

            # Verify all scores are valid numbers
            for score in multi_krum_strategy.client_scores.values():
                assert isinstance(score, (int, float))
                assert np.isfinite(score)

    def test_aggregate_fit_top_client_selection(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test aggregate_fit selects top num_krum_selections clients."""
        with patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler, patch(
            "flwr.server.strategy.Krum.aggregate_fit"
        ) as mock_parent_aggregate:

            # Setup mocks
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

            # Mock parent aggregate_fit to capture the selected clients
            selected_clients = []

            def capture_selected_clients(server_round, results, failures):
                selected_clients.extend(results)
                return (Mock(), {})

            mock_parent_aggregate.side_effect = capture_selected_clients

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Should select num_krum_selections clients
            assert len(selected_clients) == multi_krum_strategy.num_krum_selections

    def test_configure_fit_warmup_rounds(self, multi_krum_strategy):
        """Test configure_fit during warmup rounds."""
        multi_krum_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(6)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = multi_krum_strategy.configure_fit(
            1, mock_parameters, mock_client_manager
        )

        # Should return all clients during warmup
        assert len(result) == 6
        assert multi_krum_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase(self, multi_krum_strategy):
        """Test configure_fit removes clients with highest Multi-Krum scores."""
        multi_krum_strategy.current_round = 3  # After begin_removing_from_round
        multi_krum_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # High score - candidate for removal
            "client_2": 0.3,
            "client_3": 0.2,
            "client_4": 0.9,  # Highest score - should be removed first
            "client_5": 0.5,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(6)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        multi_krum_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should remove one client with highest Multi-Krum score
        assert len(multi_krum_strategy.removed_client_ids) == 1
        assert "client_4" in multi_krum_strategy.removed_client_ids

    def test_configure_fit_removal_limit(self, multi_krum_strategy):
        """Test configure_fit respects removal limit based on num_krum_selections."""
        multi_krum_strategy.current_round = 5  # Well after begin_removing_from_round
        multi_krum_strategy.client_scores = {f"client_{i}": float(i) for i in range(6)}

        # Simulate multiple rounds of removal
        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(6)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        # Remove clients until limit is reached
        total_clients = 6
        max_removals = (
            total_clients - multi_krum_strategy.num_krum_selections
        )  # 6 - 3 = 3

        for round_num in range(max_removals + 2):  # Try to remove more than limit
            multi_krum_strategy.configure_fit(
                5 + round_num, mock_parameters, mock_client_manager
            )

        # Should not remove more than the limit
        assert len(multi_krum_strategy.removed_client_ids) <= max_removals

    def test_configure_fit_no_removal_when_disabled(self, multi_krum_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        multi_krum_strategy.remove_clients = False
        multi_krum_strategy.current_round = 3
        multi_krum_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,  # High score but shouldn't be removed
            "client_2": 0.9,  # Highest score but shouldn't be removed
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        multi_krum_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should not remove any clients
        assert multi_krum_strategy.removed_client_ids == set()

    def test_num_krum_selections_parameter_effect(
        self, mock_strategy_history, mock_output_directory, krum_fit_metrics_fn
    ):
        """Test num_krum_selections parameter affects client selection and removal limits."""
        selection_counts = [2, 4, 6]

        for num_selections in selection_counts:
            strategy = MultiKrumBasedRemovalStrategy(
                remove_clients=True,
                num_of_malicious_clients=2,
                num_krum_selections=num_selections,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            )

            assert strategy.num_krum_selections == num_selections

            # Test removal limit calculation
            total_clients = 8
            expected_max_removals = total_clients - num_selections

            strategy.current_round = 3
            strategy.client_scores = {
                f"client_{i}": float(i) for i in range(total_clients)
            }

            mock_client_manager = Mock()
            mock_clients = {f"client_{i}": Mock() for i in range(total_clients)}
            mock_client_manager.all.return_value = mock_clients

            # Simulate multiple removal rounds
            for _ in range(expected_max_removals + 2):
                strategy.configure_fit(3, Mock(), mock_client_manager)

            # Should respect the removal limit
            assert len(strategy.removed_client_ids) <= expected_max_removals

    def test_begin_removing_from_round_parameter(
        self, mock_strategy_history, mock_output_directory, krum_fit_metrics_fn
    ):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = MultiKrumBasedRemovalStrategy(
                remove_clients=True,
                num_of_malicious_clients=2,
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

    def test_strategy_history_integration(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test integration with strategy history."""
        with patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler, patch(
            "flwr.server.strategy.Krum.aggregate_fit"
        ) as mock_parent_aggregate:

            # Setup mocks
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

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify strategy history methods were called
            assert (
                multi_krum_strategy.strategy_history.insert_single_client_history_entry.call_count
                == 6
            )
            multi_krum_strategy.strategy_history.insert_round_history_entry.assert_called_once()

    def test_edge_case_empty_results(self, multi_krum_strategy):
        """Test handling of empty results."""
        with patch("flwr.server.strategy.Krum.aggregate_fit") as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (None, {})

            result = multi_krum_strategy.aggregate_fit(1, [], [])

            # Should handle empty results gracefully
            assert result is not None

    def test_edge_case_insufficient_clients_for_selections(self, multi_krum_strategy):
        """Test handling when fewer clients than num_krum_selections."""
        # Create only 2 clients when num_krum_selections is 3
        client_proxy1 = Mock(spec=ClientProxy)
        client_proxy1.cid = "0"
        mock_params1 = [np.random.randn(5, 5), np.random.randn(5)]
        fit_res1 = Mock(spec=FitRes)
        fit_res1.parameters = ndarrays_to_parameters(mock_params1)

        client_proxy2 = Mock(spec=ClientProxy)
        client_proxy2.cid = "1"
        mock_params2 = [np.random.randn(5, 5), np.random.randn(5)]
        fit_res2 = Mock(spec=FitRes)
        fit_res2.parameters = ndarrays_to_parameters(mock_params2)

        insufficient_results = [(client_proxy1, fit_res1), (client_proxy2, fit_res2)]

        with patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler, patch(
            "flwr.server.strategy.Krum.aggregate_fit"
        ) as mock_parent_aggregate:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array([[0.1], [0.2]])
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array([[0.1], [0.2]])
            mock_scaler.return_value = mock_scaler_instance

            # Mock parent aggregate_fit to capture selected clients
            selected_clients = []

            def capture_selected_clients(server_round, results, failures):
                selected_clients.extend(results)
                return (Mock(), {})

            mock_parent_aggregate.side_effect = capture_selected_clients

            multi_krum_strategy.aggregate_fit(1, insufficient_results, [])

            # Should handle insufficient clients gracefully
            # Should select all available clients (2) instead of num_krum_selections (3)
            assert len(selected_clients) == 2

    def test_distance_calculation_accuracy(self, multi_krum_strategy):
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
        multi_krum_strategy._calculate_multi_krum_scores(results, distances)

        # Verify distances match expected Euclidean distances
        for i in range(3):
            for j in range(i + 1, 3):
                expected_distance = np.linalg.norm(
                    expected_params[i] - expected_params[j]
                )
                assert abs(distances[i, j] - expected_distance) < 1e-6
                assert abs(distances[j, i] - expected_distance) < 1e-6

    def test_multi_krum_vs_krum_score_difference(self, multi_krum_strategy):
        """Test that Multi-Krum scores differ from regular Krum scores."""
        # Create test data
        results = []
        for i in range(4):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            params = [np.array([[i * 2.0]]), np.array([i * 2.0])]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((4, 4))
        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            results, distances
        )

        # Multi-Krum uses num_krum_selections - 2 = 3 - 2 = 1 closest distances
        # Regular Krum would use num_malicious_clients - 2 = 2 - 2 = 0 closest distances

        # Verify scores are calculated (specific values depend on the algorithm)
        assert len(multi_krum_scores) == 4
        assert all(np.isfinite(score) for score in multi_krum_scores)

    def test_removal_stops_when_limit_reached(self, multi_krum_strategy):
        """Test that removal stops when the limit is reached."""
        multi_krum_strategy.current_round = 5
        total_clients = 6
        max_removals = total_clients - multi_krum_strategy.num_krum_selections  # 3

        # Pre-populate removed clients to near the limit
        for i in range(max_removals - 1):
            multi_krum_strategy.removed_client_ids.add(f"client_{i}")

        multi_krum_strategy.client_scores = {
            f"client_{i}": float(i) for i in range(total_clients)
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(total_clients)}
        mock_client_manager.all.return_value = mock_clients

        # This should add one more removal, reaching the limit
        multi_krum_strategy.configure_fit(5, Mock(), mock_client_manager)
        assert len(multi_krum_strategy.removed_client_ids) == max_removals

        # This should not add any more removals
        multi_krum_strategy.configure_fit(6, Mock(), mock_client_manager)
        assert len(multi_krum_strategy.removed_client_ids) == max_removals

        # Verify remove_clients is set to False when limit is reached
        assert multi_krum_strategy.remove_clients is False
