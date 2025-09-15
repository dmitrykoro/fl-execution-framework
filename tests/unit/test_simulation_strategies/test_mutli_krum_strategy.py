"""
Unit tests for MultiKrumStrategy.

Tests the Multi-Krum aggregation strategy implementation.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from flwr.common import FitRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.mutli_krum_strategy import MultiKrumStrategy


class TestMultiKrumStrategy:
    """Test cases for MultiKrumStrategy."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def multi_krum_strategy(self, mock_strategy_history, mock_output_directory):
        """Create MultiKrumStrategy instance for testing."""
        return MultiKrumStrategy(
            remove_clients=True,
            num_of_malicious_clients=2,
            num_krum_selections=3,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def mock_client_results(self):
        """Create mock client results for testing."""
        results = []
        np.random.seed(42)  # For reproducible tests

        for i in range(6):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create mock parameters
            mock_params = [
                np.random.randn(10, 5).astype(np.float32),
                np.random.randn(5).astype(np.float32),
            ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100

            results.append((client_proxy, fit_res))

        return results

    def test_initialization(self, multi_krum_strategy, mock_strategy_history):
        """Test MultiKrumStrategy initialization."""
        assert multi_krum_strategy.client_scores == {}
        assert multi_krum_strategy.removed_client_ids == set()
        assert multi_krum_strategy.remove_clients is True
        assert multi_krum_strategy.num_of_malicious_clients == 2
        assert multi_krum_strategy.num_krum_selections == 3
        assert multi_krum_strategy.begin_removing_from_round == 2
        assert multi_krum_strategy.current_round == 0
        assert multi_krum_strategy.strategy_history == mock_strategy_history

    def test_initialization_creates_logger(self, multi_krum_strategy):
        """Test MultiKrumStrategy creates logger during initialization."""
        assert hasattr(multi_krum_strategy, "logger")
        assert multi_krum_strategy.logger.name == "my_logger"

    def test_calculate_multi_krum_scores_distance_calculation(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test _calculate_multi_krum_scores calculates distances correctly."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            mock_client_results, distances
        )

        # Distance matrix should be symmetric
        assert np.allclose(distances, distances.T)

        # Diagonal should be zero
        assert np.allclose(np.diag(distances), 0)

        # All distances should be non-negative
        assert np.all(distances >= 0)

        # Should return scores for all clients
        assert len(multi_krum_scores) == len(mock_client_results)

    def test_calculate_multi_krum_scores_parameter_flattening(
        self, multi_krum_strategy
    ):
        """Test _calculate_multi_krum_scores flattens parameters correctly."""
        # Create simple test data
        results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Parameters with known shapes
            params = [
                np.ones((2, 2), dtype=np.float32) * i,
                np.ones(2, dtype=np.float32) * i,
            ]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((3, 3))
        multi_krum_strategy._calculate_multi_krum_scores(results, distances)

        # Parameters should be flattened: (2,2) + (2,) = 4 + 2 = 6 elements
        # Distance between client 0 and 1: ||[0,0,0,0,0,0] - [1,1,1,1,1,1]|| = sqrt(6)
        expected_distance_01 = np.sqrt(6.0)
        assert abs(distances[0, 1] - expected_distance_01) < 1e-6

    def test_calculate_multi_krum_scores_selection_parameter(self, multi_krum_strategy):
        """Test _calculate_multi_krum_scores uses num_krum_selections correctly."""
        # Create test data with known distances
        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            params = [np.array([[i]], dtype=np.float32)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((5, 5))
        scores = multi_krum_strategy._calculate_multi_krum_scores(results, distances)

        # For num_krum_selections=3, should use 3-2=1 closest distances
        # For client 2 (middle client), closest distance should be to client 1 or 3
        assert len(scores) == 5
        assert all(score >= 0 for score in scores)

    @patch("src.simulation_strategies.mutli_krum_strategy.KMeans")
    @patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler")
    @patch("src.simulation_strategies.mutli_krum_strategy.time.time_ns")
    def test_aggregate_fit_clustering_setup(
        self,
        mock_time,
        mock_scaler,
        mock_kmeans,
        multi_krum_strategy,
        mock_client_results,
    ):
        """Test aggregate_fit sets up clustering correctly."""
        # Mock time for performance measurement
        mock_time.side_effect = [1000000000, 1000001000]  # 1ms difference

        # Mock KMeans
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.random.rand(6, 1)
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        # Mock MinMaxScaler
        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.random.rand(6, 1)
        mock_scaler.return_value = mock_scaler_instance

        # Mock parent class aggregate_fit method
        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify clustering components were called
            mock_kmeans.assert_called_once_with(n_clusters=1, init="k-means++")
            mock_scaler_instance.fit.assert_called_once()
            mock_scaler_instance.transform.assert_called_once()

    @patch("src.simulation_strategies.mutli_krum_strategy.KMeans")
    @patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler")
    @patch("src.simulation_strategies.mutli_krum_strategy.time.time_ns")
    def test_aggregate_fit_parameter_extraction(
        self,
        mock_time,
        mock_scaler,
        mock_kmeans,
        multi_krum_strategy,
        mock_client_results,
    ):
        """Test aggregate_fit extracts parameters correctly for clustering."""
        mock_time.side_effect = [1000000000, 1000001000]

        # Mock clustering components
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.random.rand(6, 1)
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.random.rand(6, 1)
        mock_scaler.return_value = mock_scaler_instance

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            def create_kmeans_mock(*args, **kwargs):
                instance = Mock()
                instance.transform.return_value = np.random.rand(6, 1)
                instance.fit = Mock(return_value=instance)
                return instance

            mock_kmeans.side_effect = create_kmeans_mock

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Should have extracted parameters from all clients
            mock_kmeans.assert_called_once()

    @patch("src.simulation_strategies.mutli_krum_strategy.KMeans")
    @patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler")
    @patch("src.simulation_strategies.mutli_krum_strategy.time.time_ns")
    def test_aggregate_fit_multi_krum_scoring(
        self,
        mock_time,
        mock_scaler,
        mock_kmeans,
        multi_krum_strategy,
        mock_client_results,
    ):
        """Test aggregate_fit calculates Multi-Krum scores."""
        mock_time.side_effect = [1000000000, 1000001000]

        # Mock clustering
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.random.rand(6, 1)
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.random.rand(6, 1)
        mock_scaler.return_value = mock_scaler_instance

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Should have calculated scores for all clients
            assert len(multi_krum_strategy.client_scores) == len(mock_client_results)

            # All scores should be valid numbers
            for score in multi_krum_strategy.client_scores.values():
                assert isinstance(score, (int, float))
                assert np.isfinite(score)

    @patch("src.simulation_strategies.mutli_krum_strategy.KMeans")
    @patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler")
    @patch("src.simulation_strategies.mutli_krum_strategy.time.time_ns")
    def test_aggregate_fit_client_selection(
        self,
        mock_time,
        mock_scaler,
        mock_kmeans,
        multi_krum_strategy,
        mock_client_results,
    ):
        """Test aggregate_fit selects top clients correctly."""
        mock_time.side_effect = [1000000000, 1000001000]

        # Mock clustering
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.random.rand(6, 1)
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.random.rand(6, 1)
        mock_scaler.return_value = mock_scaler_instance

        # Track which clients are selected for aggregation
        selected_clients = []

        def capture_selected_clients(server_round, results, failures):
            selected_clients.extend(results)
            return (Mock(), {})

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.side_effect = capture_selected_clients

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Should select num_krum_selections clients
            assert len(selected_clients) == multi_krum_strategy.num_krum_selections

    @patch("src.simulation_strategies.mutli_krum_strategy.KMeans")
    @patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler")
    @patch("src.simulation_strategies.mutli_krum_strategy.time.time_ns")
    def test_aggregate_fit_performance_tracking(
        self,
        mock_time,
        mock_scaler,
        mock_kmeans,
        multi_krum_strategy,
        mock_client_results,
    ):
        """Test aggregate_fit tracks performance metrics."""
        # Mock time to return specific values
        start_time = 1000000000
        end_time = 1000001000
        mock_time.side_effect = [start_time, end_time]

        # Mock clustering
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.random.rand(6, 1)
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        mock_scaler_instance = Mock()
        mock_scaler_instance.transform.return_value = np.random.rand(6, 1)
        mock_scaler.return_value = mock_scaler_instance

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Should record performance timing
            multi_krum_strategy.strategy_history.insert_round_history_entry.assert_called_once_with(
                score_calculation_time_nanos=end_time - start_time
            )

    @patch("src.simulation_strategies.mutli_krum_strategy.KMeans")
    @patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler")
    @patch("src.simulation_strategies.mutli_krum_strategy.time.time_ns")
    def test_aggregate_fit_client_history_tracking(
        self,
        mock_time,
        mock_scaler,
        mock_kmeans,
        multi_krum_strategy,
        mock_client_results,
    ):
        """Test aggregate_fit records client history correctly."""
        mock_time.side_effect = [1000000000, 1000001000]

        # Mock clustering
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

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

            # Should record history for all clients
            assert (
                multi_krum_strategy.strategy_history.insert_single_client_history_entry.call_count
                == len(mock_client_results)
            )

            # Check that client history entries have required parameters
            calls = (
                multi_krum_strategy.strategy_history.insert_single_client_history_entry.call_args_list
            )
            for i, call in enumerate(calls):
                args, kwargs = call
                assert "current_round" in kwargs
                assert "client_id" in kwargs
                assert "removal_criterion" in kwargs
                assert "absolute_distance" in kwargs

    def test_aggregate_fit_round_counting(self, multi_krum_strategy):
        """Test aggregate_fit increments round counter correctly."""
        initial_round = multi_krum_strategy.current_round

        with patch("src.simulation_strategies.mutli_krum_strategy.KMeans"):
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.return_value = (Mock(), {})

                        multi_krum_strategy.aggregate_fit(1, [], [])

                        assert multi_krum_strategy.current_round == initial_round + 1

    def test_aggregate_fit_empty_results(self, multi_krum_strategy):
        """Test aggregate_fit handles empty results gracefully."""
        with patch("src.simulation_strategies.mutli_krum_strategy.KMeans"):
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.return_value = (None, {})

                        result = multi_krum_strategy.aggregate_fit(1, [], [])

                        # Should handle empty results without crashing
                        assert result is not None
                        assert multi_krum_strategy.current_round == 1

    def test_aggregate_fit_insufficient_clients(self, multi_krum_strategy):
        """Test aggregate_fit with fewer clients than num_krum_selections."""
        # Create only 2 clients when num_krum_selections is 3
        results = []
        for i in range(2):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            params = [np.array([[i]], dtype=np.float32)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        selected_clients = []

        def capture_selected_clients(server_round, client_results, failures):
            selected_clients.extend(client_results)
            return (Mock(), {})

        with patch("src.simulation_strategies.mutli_krum_strategy.KMeans"):
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.side_effect = capture_selected_clients

                        multi_krum_strategy.aggregate_fit(1, results, [])

                        # Should select all available clients (2) instead of num_krum_selections (3)
                        assert len(selected_clients) == 2

    def test_parameter_tensor_conversion(self, multi_krum_strategy):
        """Test parameter conversion to PyTorch tensors during clustering."""
        # Create results with known parameter shapes
        results = []
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"

        # Parameters that should be converted to tensors and flattened
        params = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # 2x2 matrix
            np.array([5.0, 6.0], dtype=np.float32),  # 1D array
        ]
        fit_res = Mock(spec=FitRes)
        fit_res.parameters = ndarrays_to_parameters(params)
        results.append((client_proxy, fit_res))

        with patch(
            "src.simulation_strategies.mutli_krum_strategy.KMeans"
        ) as mock_kmeans:
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    # Mock KMeans to capture the data passed to it
                    captured_data = None

                    def capture_fit_data(X):
                        nonlocal captured_data
                        captured_data = X
                        instance = Mock()
                        instance.transform.return_value = np.array([[0.1]])
                        return instance

                    mock_kmeans.return_value.fit.side_effect = capture_fit_data

                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.return_value = (Mock(), {})

                        multi_krum_strategy.aggregate_fit(1, results, [])

                        # Should have captured tensor data for clustering
                        assert captured_data is not None
                        assert len(captured_data) == 1  # One client
                        # Flattened: [1,2,3,4,5,6] = 6 elements
                        assert captured_data[0].shape == (6,)

    def test_kmeans_single_cluster_configuration(self, multi_krum_strategy):
        """Test KMeans is configured with single cluster."""
        with patch(
            "src.simulation_strategies.mutli_krum_strategy.KMeans"
        ) as mock_kmeans:
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.return_value = (Mock(), {})

                        multi_krum_strategy.aggregate_fit(1, [], [])

                        # Should configure KMeans with n_clusters=1
                        mock_kmeans.assert_called_once_with(
                            n_clusters=1, init="k-means++"
                        )

    def test_score_selection_uses_argsort(self, multi_krum_strategy):
        """Test client selection uses argsort for lowest scores."""
        # Mock to capture the selection logic
        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            params = [np.array([[i]], dtype=np.float32)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        selected_clients = []

        def capture_selected_clients(server_round, client_results, failures):
            selected_clients.extend(client_results)
            return (Mock(), {})

        with patch("src.simulation_strategies.mutli_krum_strategy.KMeans"):
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.side_effect = capture_selected_clients

                        multi_krum_strategy.aggregate_fit(1, results, [])

                        # Should select exactly num_krum_selections clients
                        assert (
                            len(selected_clients)
                            == multi_krum_strategy.num_krum_selections
                        )

    def test_client_scores_stored_as_float(
        self, multi_krum_strategy, mock_client_results
    ):
        """Test client scores are stored as float values."""
        with patch("src.simulation_strategies.mutli_krum_strategy.KMeans"):
            with patch("src.simulation_strategies.mutli_krum_strategy.MinMaxScaler"):
                with patch(
                    "src.simulation_strategies.mutli_krum_strategy.time.time_ns"
                ):
                    with patch(
                        "flwr.server.strategy.FedAvg.aggregate_fit"
                    ) as mock_parent_aggregate:
                        mock_parent_aggregate.return_value = (Mock(), {})

                        multi_krum_strategy.aggregate_fit(1, mock_client_results, [])

                        # All stored scores should be floats
                        for (
                            client_id,
                            score,
                        ) in multi_krum_strategy.client_scores.items():
                            assert isinstance(client_id, str)
                            assert isinstance(score, float)
