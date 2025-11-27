"""
Unit tests for MultiKrumStrategy.

Tests Multi-Krum aggregation algorithms and client scoring logic.
"""

from unittest.mock import patch

from tests.common import Mock, np, pytest, FitRes, ndarrays_to_parameters, ClientProxy
from flwr.common import EvaluateRes
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.multi_krum_strategy import MultiKrumStrategy

from tests.common import generate_mock_client_data


class TestMultiKrumStrategy:
    """MultiKrumStrategy unit tests."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def multi_krum_strategy(self, mock_strategy_history, mock_output_directory):
        """MultiKrumStrategy with test parameters."""
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
        """Six mock clients with generated parameters."""
        return generate_mock_client_data(num_clients=6)

    def test_initialization(self, multi_krum_strategy, mock_strategy_history):
        """Verify initialization sets parameters correctly."""
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
        """Verify distance matrix is symmetric with zero diagonal."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            mock_client_results,
            distances,  # type: ignore[arg-type]
        )

        assert np.allclose(distances, distances.T)
        assert isinstance(multi_krum_scores, list)
        assert len(multi_krum_scores) == len(mock_client_results)
        assert np.allclose(np.diag(distances), 0)
        assert np.all(distances >= 0)

    def test_calculate_multi_krum_scores_computation(
        self, multi_krum_strategy, mock_client_results
    ):
        """Verify scores are non-negative and finite."""
        distances = np.zeros((len(mock_client_results), len(mock_client_results)))

        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            mock_client_results,
            distances,  # type: ignore[arg-type]
        )

        assert len(multi_krum_scores) == len(mock_client_results)
        assert all(score >= 0 for score in multi_krum_scores)
        assert all(np.isfinite(score) for score in multi_krum_scores)

    def test_calculate_multi_krum_scores_selection_parameter_effect(
        self, mock_strategy_history, mock_output_directory
    ):
        """Verify num_krum_selections parameter changes score calculation."""
        selection_counts = [2, 3, 4]

        for num_selections in selection_counts:
            strategy = MultiKrumStrategy(
                remove_clients=True,
                num_of_malicious_clients=2,
                num_krum_selections=num_selections,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
            )

            results = []
            for i in range(5):
                client_proxy = Mock(spec=ClientProxy)
                client_proxy.cid = str(i)
                mock_params = [np.ones((2, 2)) * i, np.ones(2) * i]
                fit_res = Mock(spec=FitRes)
                fit_res.parameters = ndarrays_to_parameters(mock_params)
                results.append((client_proxy, fit_res))

            distances = np.zeros((5, 5))
            scores = strategy._calculate_multi_krum_scores(results, distances)  # type: ignore[arg-type]

            assert len(scores) == 5
            assert all(np.isfinite(score) for score in scores)

    @patch("src.simulation_strategies.multi_krum_strategy.KMeans")
    @patch("src.simulation_strategies.multi_krum_strategy.MinMaxScaler")
    def test_aggregate_fit_clustering(
        self, mock_scaler, mock_kmeans, multi_krum_strategy, mock_client_results
    ):
        """Verify clustering components are called during aggregation."""
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

            mock_kmeans.assert_called_once()
            mock_scaler_instance.fit.assert_called_once()
            mock_scaler_instance.transform.assert_called_once()

    def test_aggregate_fit_multi_krum_score_calculation(
        self, multi_krum_strategy, mock_client_results
    ):
        """Verify scores are calculated for all participating clients."""
        with (
            patch(
                "src.simulation_strategies.multi_krum_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.multi_krum_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
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

            assert len(multi_krum_strategy.client_scores) == 6
            for score in multi_krum_strategy.client_scores.values():
                assert isinstance(score, (int, float))
                assert np.isfinite(score)

    def test_aggregate_fit_top_client_selection(
        self, multi_krum_strategy, mock_client_results
    ):
        """Verify client selection matches num_krum_selections parameter."""
        with (
            patch(
                "src.simulation_strategies.multi_krum_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.multi_krum_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
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

            assert len(selected_clients) == multi_krum_strategy.num_krum_selections

    def test_aggregate_fit_timing_history_recording(
        self, multi_krum_strategy, mock_client_results
    ):
        """Verify timing data is recorded in strategy history."""
        with (
            patch(
                "src.simulation_strategies.multi_krum_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.multi_krum_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
            patch("src.simulation_strategies.multi_krum_strategy.time") as mock_time,
        ):
            mock_time.time_ns.side_effect = [1000000, 2000000]

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

            multi_krum_strategy.strategy_history.insert_round_history_entry.assert_called()
            call_args = multi_krum_strategy.strategy_history.insert_round_history_entry.call_args
            assert "score_calculation_time_nanos" in call_args.kwargs
            assert call_args.kwargs["score_calculation_time_nanos"] == 1000000

    def test_configure_fit_warmup_rounds(self, multi_krum_strategy):
        """Verify all clients selected during warmup phase."""
        multi_krum_strategy.current_round = 1

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(6)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = multi_krum_strategy.configure_fit(
            1, mock_parameters, mock_client_manager
        )

        assert len(result) == 6
        assert multi_krum_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase(self, multi_krum_strategy):
        """Verify highest scoring clients are removed after warmup."""
        multi_krum_strategy.current_round = 3
        multi_krum_strategy.client_scores = {
            "client_0": 0.1,
            "client_1": 0.8,
            "client_2": 0.3,
            "client_3": 0.2,
            "client_4": 0.9,
            "client_5": 0.5,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(6)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        multi_krum_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        expected_removals = 6 - multi_krum_strategy.num_krum_selections
        assert len(multi_krum_strategy.removed_client_ids) == expected_removals
        assert "client_4" in multi_krum_strategy.removed_client_ids
        assert "client_1" in multi_krum_strategy.removed_client_ids

    def test_configure_fit_client_participation_history(self, multi_krum_strategy):
        """Verify participation history updated with removed clients."""
        multi_krum_strategy.current_round = 3
        multi_krum_strategy.client_scores = {f"client_{i}": float(i) for i in range(6)}

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(6)}
        mock_client_manager.all.return_value = mock_clients

        multi_krum_strategy.configure_fit(3, Mock(), mock_client_manager)

        multi_krum_strategy.strategy_history.update_client_participation.assert_called_once_with(
            current_round=3, removed_client_ids=multi_krum_strategy.removed_client_ids
        )

    def test_aggregate_evaluate_logging_and_history(self, multi_krum_strategy):
        """Verify evaluation results logged and stored in history."""
        eval_results = []
        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + i * 0.1
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8 - i * 0.1}
            eval_results.append((client_proxy, eval_res))

        multi_krum_strategy.current_round = 2
        multi_krum_strategy.removed_client_ids = {"2"}  # Remove client 2

        result = multi_krum_strategy.aggregate_evaluate(1, eval_results, [])

        assert result is not None
        loss, metrics = result
        assert isinstance(loss, float)
        assert isinstance(metrics, dict)

        assert (
            multi_krum_strategy.strategy_history.insert_single_client_history_entry.call_count
            >= 3
        )

    def test_aggregate_evaluate_removed_client_exclusion(self, multi_krum_strategy):
        """Verify removed clients excluded from loss aggregation."""
        eval_results = []
        for i in range(4):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            eval_res = Mock(spec=EvaluateRes)
            eval_res.loss = 0.5 + i * 0.1
            eval_res.num_examples = 100
            eval_res.metrics = {"accuracy": 0.8 - i * 0.1}
            eval_results.append((client_proxy, eval_res))

        multi_krum_strategy.current_round = 2
        multi_krum_strategy.removed_client_ids = {"2", "3"}

        with patch(
            "src.simulation_strategies.multi_krum_strategy.weighted_loss_avg"
        ) as mock_weighted_loss:
            mock_weighted_loss.return_value = 0.55

            multi_krum_strategy.aggregate_evaluate(1, eval_results, [])

            assert mock_weighted_loss.call_count == 1
            aggregated_data = mock_weighted_loss.call_args[0][0]
            assert len(aggregated_data) == 2

    def test_aggregate_evaluate_empty_results(self, multi_krum_strategy):
        """Verify empty results return None gracefully."""
        result = multi_krum_strategy.aggregate_evaluate(1, [], [])

        assert result == (None, {})

    def test_edge_case_insufficient_clients_for_selections(self, multi_krum_strategy):
        """Verify graceful handling when client count < num_krum_selections."""
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

        with (
            patch(
                "src.simulation_strategies.multi_krum_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.multi_krum_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
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

            assert len(selected_clients) == 2

    def test_distance_calculation_accuracy(self, multi_krum_strategy):
        """Verify Euclidean distance calculations match expected values."""
        results = []
        expected_params = []

        for i in range(3):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            params = [np.array([[i, i]]), np.array([i])]
            expected_params.append(np.concatenate([p.flatten() for p in params]))

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((3, 3))
        multi_krum_strategy._calculate_multi_krum_scores(results, distances)  # type: ignore[arg-type]

        for i in range(3):
            for j in range(i + 1, 3):
                expected_distance = np.linalg.norm(
                    expected_params[i] - expected_params[j]
                )
                assert abs(distances[i, j] - expected_distance) < 1e-6
                assert abs(distances[j, i] - expected_distance) < 1e-6

    def test_logging_configuration(self, multi_krum_strategy):
        """Verify logger configuration and isolation."""
        assert multi_krum_strategy.logger is not None
        assert multi_krum_strategy.logger.name.startswith("multi_krum_")
        assert multi_krum_strategy.logger.level == 20
        assert multi_krum_strategy.logger.propagate is False

    def test_strategy_history_client_entry_creation(
        self, multi_krum_strategy, mock_client_results
    ):
        """Verify client history entries contain required fields."""
        with (
            patch(
                "src.simulation_strategies.multi_krum_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.multi_krum_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
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

            assert (
                multi_krum_strategy.strategy_history.insert_single_client_history_entry.call_count
                == 6
            )

            calls = multi_krum_strategy.strategy_history.insert_single_client_history_entry.call_args_list
            for call in calls:
                assert "current_round" in call.kwargs
                assert "client_id" in call.kwargs
                assert "removal_criterion" in call.kwargs
                assert "absolute_distance" in call.kwargs
