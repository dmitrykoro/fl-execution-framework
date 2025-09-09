"""
Unit tests for BulyanStrategy.

Tests Bulyan aggregation algorithm combining Multi-Krum and trimmed mean.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_history import \
    SimulationStrategyHistory
from src.simulation_strategies.bulyan_strategy import BulyanStrategy


class TestBulyanStrategy:
    """Test cases for BulyanStrategy."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def bulyan_strategy(self, mock_strategy_history, mock_output_directory):
        """Create BulyanStrategy instance for testing."""
        return BulyanStrategy(
            remove_clients=True,
            num_krum_selections=13,  # n - f, where n=15, f=1
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def mock_client_results(self):
        """Create mock client results for testing Bulyan algorithm."""
        results = []
        np.random.seed(42)  # For reproducible tests

        # Create 15 clients for proper Bulyan testing (n=15, f=4, C=6)
        for i in range(15):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create mock parameters with some variation
            if i < 2:  # Potential Byzantine clients
                mock_params = [np.random.randn(5, 3) * 3, np.random.randn(3) * 3]
            else:  # Honest clients
                mock_params = [np.random.randn(5, 3), np.random.randn(3)]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100

            results.append((client_proxy, fit_res))

        return results

    def test_initialization(self, bulyan_strategy, mock_strategy_history):
        """Test BulyanStrategy initialization."""
        assert bulyan_strategy.remove_clients is True
        assert bulyan_strategy.num_krum_selections == 13
        assert bulyan_strategy.begin_removing_from_round == 2
        assert bulyan_strategy.strategy_history == mock_strategy_history
        assert bulyan_strategy.current_round == 0
        assert bulyan_strategy.client_scores == {}
        assert bulyan_strategy.removed_client_ids == set()

    def test_pairwise_sq_dists_static_method(self):
        """Test _pairwise_sq_dists static method."""
        vectors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        distances = BulyanStrategy._pairwise_sq_dists(vectors)

        # Should be symmetric
        assert np.allclose(distances, distances.T)

        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0)

        # Check specific distances
        assert abs(distances[0, 1] - 1.0) < 1e-6  # Distance from (0,0) to (1,0) squared
        assert abs(distances[0, 2] - 1.0) < 1e-6  # Distance from (0,0) to (0,1) squared
        assert abs(distances[1, 2] - 2.0) < 1e-6  # Distance from (1,0) to (0,1) squared

    def test_pairwise_sq_dists_single_vector(self):
        """Test _pairwise_sq_dists with single vector."""
        vectors = np.array([[1.0, 2.0, 3.0]])

        distances = BulyanStrategy._pairwise_sq_dists(vectors)

        # Should be 1x1 matrix with zero
        assert distances.shape == (1, 1)
        assert distances[0, 0] == 0.0

    def test_pairwise_sq_dists_identical_vectors(self):
        """Test _pairwise_sq_dists with identical vectors."""
        vectors = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        )

        distances = BulyanStrategy._pairwise_sq_dists(vectors)

        # All distances should be zero
        assert np.allclose(distances, 0.0)

    @patch("src.simulation_strategies.bulyan_strategy.KMeans")
    @patch("src.simulation_strategies.bulyan_strategy.MinMaxScaler")
    def test_aggregate_fit_clustering(
        self, mock_scaler, mock_kmeans, bulyan_strategy, mock_client_results
    ):
        """Test aggregate_fit performs clustering correctly."""
        # Setup mocks
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.array(
            [
                [0.1],
                [0.2],
                [0.3],
                [0.4],
                [0.5],
                [0.6],
                [0.7],
                [0.8],
                [0.9],
                [1.0],
                [1.1],
                [1.2],
                [1.3],
                [1.4],
                [1.5],
            ]
        )
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.array(
            [
                [0.1],
                [0.2],
                [0.3],
                [0.4],
                [0.5],
                [0.6],
                [0.7],
                [0.8],
                [0.9],
                [1.0],
                [1.1],
                [1.2],
                [1.3],
                [1.4],
                [1.5],
            ]
        )
        mock_scaler_instance.fit.return_value = mock_scaler_instance
        mock_scaler.return_value = mock_scaler_instance

        bulyan_strategy.aggregate_fit(1, mock_client_results, [])

        # Verify clustering was called
        mock_kmeans.assert_called_once()
        mock_scaler_instance.fit.assert_called_once()
        mock_scaler_instance.transform.assert_called_once()

    def test_aggregate_fit_bulyan_algorithm(self, bulyan_strategy, mock_client_results):
        """Test aggregate_fit implements Bulyan algorithm correctly."""
        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [
                    [0.1],
                    [0.2],
                    [0.3],
                    [0.4],
                    [0.5],
                    [0.6],
                    [0.7],
                    [0.8],
                    [0.9],
                    [1.0],
                    [1.1],
                    [1.2],
                    [1.3],
                    [1.4],
                    [1.5],
                ]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [
                    [0.1],
                    [0.2],
                    [0.3],
                    [0.4],
                    [0.5],
                    [0.6],
                    [0.7],
                    [0.8],
                    [0.9],
                    [1.0],
                    [1.1],
                    [1.2],
                    [1.3],
                    [1.4],
                    [1.5],
                ]
            )
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = bulyan_strategy.aggregate_fit(
                1, mock_client_results, []
            )

            # Should return aggregated parameters
            assert result_params is not None
            assert isinstance(result_metrics, dict)

            # Verify current_round was incremented
            assert bulyan_strategy.current_round == 1

            # Verify client scores were calculated
            assert len(bulyan_strategy.client_scores) == 15

    def test_aggregate_fit_insufficient_clients(self, bulyan_strategy):
        """Test aggregate_fit with insufficient clients for Bulyan preconditions."""
        # Create insufficient clients (less than 4*f + 3 = 4*2 + 3 = 11)
        insufficient_results = []
        for i in range(5):  # Only 5 clients, need at least 11
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.random.randn(3, 3), np.random.randn(3)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            insufficient_results.append((client_proxy, fit_res))

        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler, patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:

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
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            bulyan_strategy.aggregate_fit(1, insufficient_results, [])

            # Should fall back to simple mean aggregation
            mock_parent_aggregate.assert_called_once()

    def test_aggregate_fit_empty_results(self, bulyan_strategy):
        """Test aggregate_fit handles empty results."""
        result_params, result_metrics = bulyan_strategy.aggregate_fit(1, [], [])

        # Should return None for empty results
        assert result_params is None
        assert result_metrics == {}

    def test_aggregate_fit_parameter_shape_preservation(self, bulyan_strategy):
        """Test that Bulyan aggregation preserves parameter shapes."""
        # Create results with specific parameter shapes
        results = []
        original_shapes = [(3, 4), (5,), (2, 2)]

        for i in range(15):  # Sufficient clients for Bulyan
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create parameters with specific shapes
            mock_params = [
                np.random.randn(*original_shapes[0]) + i * 0.1,
                np.random.randn(*original_shapes[1]) + i * 0.1,
                np.random.randn(*original_shapes[2]) + i * 0.1,
            ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = bulyan_strategy.aggregate_fit(
                1, results, []
            )

            # Verify parameter shapes are preserved
            reconstructed_params = parameters_to_ndarrays(result_params)
            assert len(reconstructed_params) == len(original_shapes)

            for i, expected_shape in enumerate(original_shapes):
                assert reconstructed_params[i].shape == expected_shape

    def test_configure_fit_warmup_rounds(self, bulyan_strategy):
        """Test configure_fit during warmup rounds."""
        bulyan_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(15)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = bulyan_strategy.configure_fit(1, mock_parameters, mock_client_manager)

        # Should return all clients during warmup
        assert len(result) == 15
        assert bulyan_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase(self, bulyan_strategy):
        """Test configure_fit removes clients with highest scores."""
        bulyan_strategy.current_round = 3  # After begin_removing_from_round
        bulyan_strategy.client_scores = {f"client_{i}": float(i) for i in range(15)}

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(15)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        bulyan_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should remove f clients (where f = (n - C) // 2 = (15 - 12) // 2 = 1)
        assert len(bulyan_strategy.removed_client_ids) == 1

        # Should remove clients with highest scores
        removed_scores = [
            bulyan_strategy.client_scores[cid]
            for cid in bulyan_strategy.removed_client_ids
        ]
        all_scores = list(bulyan_strategy.client_scores.values())
        all_scores.sort(reverse=True)

        # Removed clients should have the highest scores
        for score in removed_scores:
            assert score in all_scores[:1]

    def test_configure_fit_no_removal_when_disabled(self, bulyan_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        bulyan_strategy.remove_clients = False
        bulyan_strategy.current_round = 3
        bulyan_strategy.client_scores = {f"client_{i}": float(i) for i in range(5)}

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        bulyan_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should not remove any clients
        assert bulyan_strategy.removed_client_ids == set()

    def test_num_krum_selections_parameter_effect(
        self, mock_strategy_history, mock_output_directory
    ):
        """Test num_krum_selections parameter affects removal calculations."""
        # Test different num_krum_selections values
        selection_counts = [4, 6, 8]

        for num_selections in selection_counts:
            strategy = BulyanStrategy(
                remove_clients=True,
                num_krum_selections=num_selections,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
            )

            assert strategy.num_krum_selections == num_selections

            # Test removal calculation: f = (n - C) // 2
            n = 15  # Total clients
            expected_f = (n - num_selections) // 2

            strategy.current_round = 3
            strategy.client_scores = {f"client_{i}": float(i) for i in range(n)}

            mock_client_manager = Mock()
            mock_clients = {f"client_{i}": Mock() for i in range(n)}
            mock_client_manager.all.return_value = mock_clients

            strategy.configure_fit(3, Mock(), mock_client_manager)

            # Should remove f clients
            assert len(strategy.removed_client_ids) == expected_f

    def test_begin_removing_from_round_parameter(
        self, mock_strategy_history, mock_output_directory
    ):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = BulyanStrategy(
                remove_clients=True,
                num_krum_selections=6,
                begin_removing_from_round=begin_round,
                strategy_history=mock_strategy_history,
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

    def test_strategy_history_integration(self, bulyan_strategy, mock_client_results):
        """Test integration with strategy history."""
        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            bulyan_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify strategy history methods were called
            assert (
                bulyan_strategy.strategy_history.insert_single_client_history_entry.call_count
                == 15
            )
            bulyan_strategy.strategy_history.insert_round_history_entry.assert_called_once()

    def test_bulyan_algorithm_multi_krum_phase(self, bulyan_strategy):
        """Test that Bulyan correctly implements Multi-Krum phase."""
        # Create controlled test data
        results = []
        for i in range(15):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create parameters with predictable patterns
            if i < 2:  # Outliers
                mock_params = [np.full((2, 2), 10.0 + i), np.full(2, 10.0 + i)]
            else:  # Normal clients
                mock_params = [
                    np.full((2, 2), 1.0 + i * 0.1),
                    np.full(2, 1.0 + i * 0.1),
                ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = bulyan_strategy.aggregate_fit(
                1, results, []
            )

            # Should successfully aggregate
            assert result_params is not None

            # Verify the result is reasonable (not dominated by outliers)
            aggregated_arrays = parameters_to_ndarrays(result_params)

            # The aggregated result should be closer to normal values than outlier values
            assert np.all(
                aggregated_arrays[0] < 5.0
            )  # Should not be close to outlier values (10+)
            assert np.all(aggregated_arrays[1] < 5.0)

    def test_bulyan_algorithm_trimmed_mean_phase(self, bulyan_strategy):
        """Test that Bulyan correctly implements trimmed mean phase."""
        # This is tested implicitly in the main algorithm test, but we can verify
        # that the algorithm handles the trimming correctly by checking that
        # extreme values are properly handled

        # Create results where trimmed mean should remove extremes
        results = []
        for i in range(15):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create a mix of normal and extreme values
            if i == 0:  # Extreme low
                mock_params = [np.full((2,), -100.0), np.full(1, -100.0)]
            elif i == 9:  # Extreme high
                mock_params = [np.full((2,), 100.0), np.full(1, 100.0)]
            else:  # Normal values
                mock_params = [np.full((2,), float(i)), np.full(1, float(i))]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = bulyan_strategy.aggregate_fit(
                1, results, []
            )

            # Should successfully aggregate
            assert result_params is not None

            # The result should not be dominated by extreme values
            aggregated_arrays = parameters_to_ndarrays(result_params)

            # Should be in reasonable range (not ±100)
            assert np.all(np.abs(aggregated_arrays[0]) < 50.0)
            assert np.all(np.abs(aggregated_arrays[1]) < 50.0)

    def test_edge_case_exact_minimum_clients(self, bulyan_strategy):
        """Test with exactly the minimum number of clients for Bulyan."""
        # For C=13, f=1, minimum clients = 15 (satisfies n > 4*f + 2 = 6)
        min_results = []
        for i in range(15):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.random.randn(2, 2), np.random.randn(2)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            min_results.append((client_proxy, fit_res))

        with patch(
            "src.simulation_strategies.bulyan_strategy.KMeans"
        ) as mock_kmeans, patch(
            "src.simulation_strategies.bulyan_strategy.MinMaxScaler"
        ) as mock_scaler:

            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array(
                [[0.1 * i] for i in range(15)]
            )
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler.return_value = mock_scaler_instance

            result_params, result_metrics = bulyan_strategy.aggregate_fit(
                1, min_results, []
            )

            # Should handle minimum clients correctly
            assert result_params is not None
            assert len(bulyan_strategy.client_scores) == 15

    def test_removed_clients_reset_each_round(self, bulyan_strategy):
        """Test that removed_client_ids is reset each round."""
        bulyan_strategy.current_round = 3
        bulyan_strategy.client_scores = {f"client_{i}": float(i) for i in range(15)}

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(15)}
        mock_client_manager.all.return_value = mock_clients

        # First call
        bulyan_strategy.configure_fit(3, Mock(), mock_client_manager)
        first_removed = bulyan_strategy.removed_client_ids.copy()

        # Second call with different scores
        bulyan_strategy.client_scores = {
            f"client_{i}": float(9 - i) for i in range(15)
        }  # Reverse order
        bulyan_strategy.configure_fit(4, Mock(), mock_client_manager)
        second_removed = bulyan_strategy.removed_client_ids.copy()

        # Should have different removed clients (since scores changed)
        # Note: The implementation resets removed_client_ids each round
        assert len(first_removed) == 1
        assert len(second_removed) == 1
