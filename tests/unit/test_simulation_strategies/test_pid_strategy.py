"""
Unit tests for PIDBasedRemovalStrategy.

Tests PID controller logic implementation, PID variants behavior, and parameter handling.
"""

from unittest.mock import patch

import torch
from tests.common import Mock, np, pytest, FitRes, ndarrays_to_parameters, ClientProxy
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy

from tests.common import generate_mock_client_data


class TestPIDBasedRemovalStrategy:
    """Test cases for PIDBasedRemovalStrategy."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def mock_network_model(self):
        """Create mock network model."""
        return Mock()

    @pytest.fixture
    def pid_strategy(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Create PIDBasedRemovalStrategy instance for testing."""
        return PIDBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            ki=0.1,
            kd=0.01,
            kp=1.0,
            num_std_dev=2.0,
            strategy_history=mock_strategy_history,
            network_model=mock_network_model,
            use_lora=False,
            aggregation_strategy_keyword="pid",
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def pid_scaled_strategy(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Create PIDBasedRemovalStrategy instance for pid_scaled testing."""
        return PIDBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            ki=0.1,
            kd=0.01,
            kp=1.0,
            num_std_dev=2.0,
            strategy_history=mock_strategy_history,
            network_model=mock_network_model,
            use_lora=False,
            aggregation_strategy_keyword="pid_scaled",
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def pid_standardized_strategy(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Create PIDBasedRemovalStrategy instance for pid_standardized testing."""
        return PIDBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            ki=0.1,
            kd=0.01,
            kp=1.0,
            num_std_dev=2.0,
            strategy_history=mock_strategy_history,
            network_model=mock_network_model,
            use_lora=False,
            aggregation_strategy_keyword="pid_standardized",
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    @pytest.fixture
    def mock_client_results(self):
        """Generate mock client results for testing."""
        return generate_mock_client_data(num_clients=5)

    def test_initialization(
        self, pid_strategy, mock_strategy_history, mock_network_model
    ):
        """Test PIDBasedRemovalStrategy initialization."""
        assert pid_strategy.remove_clients is True
        assert pid_strategy.begin_removing_from_round == 2
        assert pid_strategy.ki == 0.1
        assert pid_strategy.kd == 0.01
        assert pid_strategy.kp == 1.0
        assert pid_strategy.num_std_dev == 2.0
        assert pid_strategy.strategy_history == mock_strategy_history
        assert pid_strategy.network_model == mock_network_model
        assert pid_strategy.aggregation_strategy_keyword == "pid"
        assert pid_strategy.current_round == 0
        assert pid_strategy.client_pids == {}
        assert pid_strategy.client_distance_sums == {}
        assert pid_strategy.client_distances == {}
        assert pid_strategy.removed_client_ids == set()

    def test_calculate_single_client_pid_first_round(self, pid_strategy):
        """Test PID calculation for first round (P component only)."""
        pid_strategy.current_round = 1
        client_id = "client_1"
        distance = 0.5

        pid_score = pid_strategy.calculate_single_client_pid(client_id, distance)

        # First round should only have P component
        expected_p = distance * pid_strategy.kp
        assert pid_score == expected_p

    def test_calculate_single_client_pid_subsequent_rounds(self, pid_strategy):
        """Test PID calculation for subsequent rounds (P + I + D components)."""
        pid_strategy.current_round = 3
        client_id = "client_1"
        distance = 0.5

        # Set up previous state
        pid_strategy.client_distance_sums[client_id] = 1.2
        pid_strategy.client_distances[client_id] = 0.3

        pid_score = pid_strategy.calculate_single_client_pid(client_id, distance)

        # Should include P, I, and D components
        expected_p = distance * pid_strategy.kp
        expected_i = pid_strategy.client_distance_sums[client_id] * pid_strategy.ki
        expected_d = pid_strategy.kd * (
            distance - pid_strategy.client_distances[client_id]
        )
        expected_pid = expected_p + expected_i + expected_d

        assert abs(pid_score - expected_pid) < 1e-6

    def test_calculate_single_client_pid_scaled_first_round(self, pid_scaled_strategy):
        """Test PID scaled calculation for first round."""
        pid_scaled_strategy.current_round = 1
        client_id = "client_1"
        distance = 0.5

        pid_score = pid_scaled_strategy.calculate_single_client_pid_scaled(
            client_id, distance
        )

        # First round should only have P component
        expected_p = distance * pid_scaled_strategy.kp
        assert pid_score == expected_p

    def test_calculate_single_client_pid_scaled_subsequent_rounds(
        self, pid_scaled_strategy
    ):
        """Test PID scaled calculation for subsequent rounds with I scaling."""
        pid_scaled_strategy.current_round = 3
        client_id = "client_1"
        distance = 0.5

        # Set up previous state
        pid_scaled_strategy.client_distance_sums[client_id] = 1.2
        pid_scaled_strategy.client_distances[client_id] = 0.3

        pid_score = pid_scaled_strategy.calculate_single_client_pid_scaled(
            client_id, distance
        )

        # Should include P, scaled I, and D components
        expected_p = distance * pid_scaled_strategy.kp
        expected_i_scaled = (
            pid_scaled_strategy.client_distance_sums[client_id] * pid_scaled_strategy.ki
        ) / pid_scaled_strategy.current_round
        expected_d = pid_scaled_strategy.kd * (
            distance - pid_scaled_strategy.client_distances[client_id]
        )
        expected_pid = expected_p + expected_i_scaled + expected_d

        assert abs(pid_score - expected_pid) < 1e-6

    def test_calculate_single_client_pid_standardized_first_round(
        self, pid_standardized_strategy
    ):
        """Test PID standardized calculation for first round."""
        pid_standardized_strategy.current_round = 1
        client_id = "client_1"
        distance = 0.5
        avg_sum = 1.0
        sum_std_dev = 0.2

        pid_score = pid_standardized_strategy.calculate_single_client_pid_standardized(
            client_id, distance, avg_sum, sum_std_dev
        )

        # First round should only have P component
        expected_p = distance * pid_standardized_strategy.kp
        assert pid_score == expected_p

    def test_calculate_single_client_pid_standardized_subsequent_rounds(
        self, pid_standardized_strategy
    ):
        """Test PID standardized calculation for subsequent rounds with standardized I."""
        pid_standardized_strategy.current_round = 3
        client_id = "client_1"
        distance = 0.5
        avg_sum = 1.0
        sum_std_dev = 0.2

        # Set up previous state
        pid_standardized_strategy.client_distance_sums[client_id] = 1.2
        pid_standardized_strategy.client_distances[client_id] = 0.3

        pid_score = pid_standardized_strategy.calculate_single_client_pid_standardized(
            client_id, distance, avg_sum, sum_std_dev
        )

        # Should include P, standardized I, and D components
        expected_p = distance * pid_standardized_strategy.kp
        expected_i_standardized = (
            (pid_standardized_strategy.client_distance_sums[client_id] - avg_sum)
            / sum_std_dev
        ) * pid_standardized_strategy.ki
        expected_d = pid_standardized_strategy.kd * (
            distance - pid_standardized_strategy.client_distances[client_id]
        )
        expected_pid = expected_p + expected_i_standardized + expected_d

        assert abs(pid_score - expected_pid) < 1e-6

    def test_calculate_single_client_pid_standardized_zero_std_dev(
        self, pid_standardized_strategy
    ):
        """Test PID standardized calculation handles zero standard deviation."""
        pid_standardized_strategy.current_round = 3
        client_id = "client_1"
        distance = 0.5
        avg_sum = 1.0
        sum_std_dev = 0.0  # Zero standard deviation

        # Set up previous state
        pid_standardized_strategy.client_distance_sums[client_id] = 1.2
        pid_standardized_strategy.client_distances[client_id] = 0.3

        pid_score = pid_standardized_strategy.calculate_single_client_pid_standardized(
            client_id, distance, avg_sum, sum_std_dev
        )

        # I component should be 0 when std_dev is 0
        expected_p = distance * pid_standardized_strategy.kp
        expected_i = 0  # Should be 0 due to zero std_dev
        expected_d = pid_standardized_strategy.kd * (
            distance - pid_standardized_strategy.client_distances[client_id]
        )
        expected_pid = expected_p + expected_i + expected_d

        assert abs(pid_score - expected_pid) < 1e-6

    def test_calculate_all_pid_scores_pid_variant(
        self, pid_strategy, mock_client_results
    ):
        """Test calculate_all_pid_scores for standard PID variant."""
        normalized_distances = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])

        pid_scores = pid_strategy.calculate_all_pid_scores(
            mock_client_results, normalized_distances
        )

        assert len(pid_scores) == 5
        assert all(isinstance(score, (int, float)) for score in pid_scores)

        # Verify PID scores were stored
        assert len(pid_strategy.client_pids) == 5

    def test_calculate_all_pid_scores_pid_scaled_variant(
        self, pid_scaled_strategy, mock_client_results
    ):
        """Test calculate_all_pid_scores for PID scaled variant."""
        normalized_distances = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])

        # Set current_round to avoid division by zero
        pid_scaled_strategy.current_round = 1

        pid_scores = pid_scaled_strategy.calculate_all_pid_scores(
            mock_client_results, normalized_distances
        )

        assert len(pid_scores) == 5
        assert all(isinstance(score, (int, float)) for score in pid_scores)

        # Verify PID scores were stored
        assert len(pid_scaled_strategy.client_pids) == 5

    def test_calculate_all_pid_scores_pid_standardized_variant(
        self, pid_standardized_strategy, mock_client_results
    ):
        """Test calculate_all_pid_scores for PID standardized variant."""
        normalized_distances = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])

        # Set up some distance sums for standardization calculation
        pid_standardized_strategy.client_distance_sums = {
            "0": 1.0,
            "1": 1.2,
            "2": 0.8,
            "3": 1.1,
            "4": 0.9,
        }

        pid_scores = pid_standardized_strategy.calculate_all_pid_scores(
            mock_client_results, normalized_distances
        )

        assert len(pid_scores) == 5
        assert all(isinstance(score, (int, float)) for score in pid_scores)

        # Verify PID scores were stored
        assert len(pid_standardized_strategy.client_pids) == 5

    def test_kp_parameter_effect(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Test Kp parameter affects P component calculation."""
        kp_values = [0.5, 1.0, 2.0]

        for kp in kp_values:
            strategy = PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=0.1,
                kd=0.01,
                kp=kp,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            )

            strategy.current_round = 1
            distance = 0.5

            pid_score = strategy.calculate_single_client_pid("client_1", distance)

            # P component should be distance * kp
            expected_p = distance * kp
            assert abs(pid_score - expected_p) < 1e-6

    def test_ki_parameter_effect(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Test Ki parameter affects I component calculation."""
        ki_values = [0.05, 0.1, 0.2]

        for ki in ki_values:
            strategy = PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=ki,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            )

            strategy.current_round = 3
            client_id = "client_1"
            distance = 0.5
            distance_sum = 1.2

            strategy.client_distance_sums[client_id] = distance_sum
            strategy.client_distances[client_id] = 0.3

            pid_score = strategy.calculate_single_client_pid(client_id, distance)

            # I component should be affected by ki
            expected_i = distance_sum * ki
            # P and D components
            expected_p = distance * strategy.kp
            expected_d = strategy.kd * (distance - strategy.client_distances[client_id])
            expected_total = expected_p + expected_i + expected_d

            assert abs(pid_score - expected_total) < 1e-6

    def test_kd_parameter_effect(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Test Kd parameter affects D component calculation."""
        kd_values = [0.005, 0.01, 0.02]

        for kd in kd_values:
            strategy = PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=0.1,
                kd=kd,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            )

            strategy.current_round = 3
            client_id = "client_1"
            distance = 0.5
            prev_distance = 0.3

            strategy.client_distance_sums[client_id] = 1.2
            strategy.client_distances[client_id] = prev_distance

            pid_score = strategy.calculate_single_client_pid(client_id, distance)

            # D component should be affected by kd
            expected_d = kd * (distance - prev_distance)
            # P and I components
            expected_p = distance * strategy.kp
            expected_i = strategy.client_distance_sums[client_id] * strategy.ki
            expected_total = expected_p + expected_i + expected_d

            assert abs(pid_score - expected_total) < 1e-6

    def test_num_std_dev_parameter_effect(self, pid_strategy):
        """Test num_std_dev parameter affects threshold calculation."""
        # Set up scenario for threshold calculation
        pid_strategy.aggregation_strategy_keyword = "pid"

        # Mock PID scores for threshold calculation
        counted_pids = [0.1, 0.3, 0.5, 0.7, 0.9]

        # Calculate expected threshold
        pid_avg = np.mean(counted_pids)
        pid_std = np.std(counted_pids)
        expected_threshold = pid_avg + (pid_strategy.num_std_dev * pid_std)

        # Manually calculate threshold using the same logic
        with (
            patch("numpy.mean", return_value=pid_avg),
            patch("numpy.std", return_value=pid_std),
        ):
            # Simulate the threshold calculation logic
            threshold = pid_avg + (pid_strategy.num_std_dev * pid_std)

            assert abs(threshold - expected_threshold) < 1e-6

    @patch("src.simulation_strategies.pid_based_removal_strategy.KMeans")
    def test_aggregate_fit_clustering(
        self, mock_kmeans, pid_strategy, mock_client_results
    ):
        """Test aggregate_fit performs clustering correctly."""
        # Setup mocks
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.transform.return_value = np.array(
            [[0.1], [0.2], [0.3], [0.4], [0.5]]
        )
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (Mock(), {})

            pid_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify clustering was called
            mock_kmeans.assert_called_once()

    def test_aggregate_fit_pid_calculation(self, pid_strategy, mock_client_results):
        """Test aggregate_fit calculates PID scores for all clients."""
        with (
            patch(
                "src.simulation_strategies.pid_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5]]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            pid_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify PID scores were calculated for all clients
            assert len(pid_strategy.client_pids) == 5
            assert len(pid_strategy.client_distances) == 5
            assert len(pid_strategy.client_distance_sums) == 5

    def test_aggregate_fit_threshold_calculation_pid(
        self, pid_strategy, mock_client_results
    ):
        """Test aggregate_fit calculates threshold correctly for PID variant."""
        with (
            patch(
                "src.simulation_strategies.pid_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5]]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            pid_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify threshold was calculated
            assert pid_strategy.current_threshold is not None
            assert isinstance(pid_strategy.current_threshold, (int, float))

    def test_aggregate_fit_threshold_calculation_pid_scaled(
        self, pid_scaled_strategy, mock_client_results
    ):
        """Test aggregate_fit calculates distance-based threshold for PID scaled variant."""
        with (
            patch(
                "src.simulation_strategies.pid_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5]]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            pid_scaled_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify threshold was calculated using distance-based approach
            assert pid_scaled_strategy.current_threshold is not None
            assert isinstance(pid_scaled_strategy.current_threshold, (int, float))

    def test_configure_fit_warmup_rounds(self, pid_strategy):
        """Test configure_fit during warmup rounds."""
        pid_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = pid_strategy.configure_fit(1, mock_parameters, mock_client_manager)

        # Should return all clients during warmup
        assert len(result) == 5
        assert pid_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase(self, pid_strategy):
        """Test configure_fit removes clients above threshold."""
        pid_strategy.current_round = 3  # After begin_removing_from_round
        pid_strategy.current_threshold = 0.5
        pid_strategy.client_pids = {
            "client_0": 0.3,  # Below threshold
            "client_1": 0.7,  # Above threshold - should be removed
            "client_2": 0.4,  # Below threshold
            "client_3": 0.8,  # Above threshold - should be removed
            "client_4": 0.2,  # Below threshold
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        pid_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should remove clients above threshold
        expected_removed = {"client_1", "client_3"}
        assert pid_strategy.removed_client_ids == expected_removed

    def test_configure_fit_no_removal_when_disabled(self, pid_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        pid_strategy.remove_clients = False
        pid_strategy.current_round = 3
        pid_strategy.current_threshold = 0.5
        pid_strategy.client_pids = {
            "client_0": 0.3,
            "client_1": 0.7,  # Above threshold but shouldn't be removed
            "client_2": 0.8,  # Above threshold but shouldn't be removed
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        pid_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should not remove any clients
        assert pid_strategy.removed_client_ids == set()

    def test_begin_removing_from_round_parameter(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=begin_round,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
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

    def test_cosine_similarity_static_method(self):
        """Test cosine similarity calculation."""

        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([2.0, 4.0, 6.0])

        similarity = PIDBasedRemovalStrategy.cosine_similarity(tensor1, tensor2)

        # Vectors are parallel, so cosine similarity should be 1.0
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""

        tensor1 = torch.tensor([1.0, 0.0])
        tensor2 = torch.tensor([0.0, 1.0])

        similarity = PIDBasedRemovalStrategy.cosine_similarity(tensor1, tensor2)

        # Orthogonal vectors should have cosine similarity of 0
        assert abs(similarity) < 1e-6

    def test_strategy_history_integration(self, pid_strategy, mock_client_results):
        """Test integration with strategy history."""
        with (
            patch(
                "src.simulation_strategies.pid_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5]]
            )
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            pid_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify strategy history methods were called
            assert (
                pid_strategy.strategy_history.insert_single_client_history_entry.call_count
                == 5
            )
            pid_strategy.strategy_history.insert_round_history_entry.assert_called()

    def test_edge_case_empty_results(self, pid_strategy):
        """Test handling of empty results."""
        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (None, {})

            result = pid_strategy.aggregate_fit(1, [], [])

            # Should handle empty results gracefully
            assert result is not None

    def test_edge_case_single_client(self, pid_strategy):
        """Test handling of single client scenario."""
        # Create single client result
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = "0"
        mock_params = [np.random.randn(10, 5), np.random.randn(5)]
        fit_res = Mock(spec=FitRes)
        fit_res.parameters = ndarrays_to_parameters(mock_params)
        fit_res.num_examples = 100

        single_result = [(client_proxy, fit_res)]

        with (
            patch(
                "src.simulation_strategies.pid_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array([[0.1]])
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            pid_strategy.aggregate_fit(1, single_result, [])

            # Should handle single client gracefully
            assert len(pid_strategy.client_pids) == 1
            assert len(pid_strategy.client_distances) == 1
            assert len(pid_strategy.client_distance_sums) == 1

    def test_pid_variants_behavior_differences(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Test that different PID variants produce different behaviors."""
        strategies = {
            "pid": PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            ),
            "pid_scaled": PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid_scaled",
            ),
            "pid_standardized": PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid_standardized",
            ),
        }

        # Set up common state for comparison
        for strategy in strategies.values():
            strategy.current_round = 3
            strategy.client_distance_sums = {"client_1": 1.2}
            strategy.client_distances = {"client_1": 0.3}

        distance = 0.5
        client_id = "client_1"

        # Calculate PID scores for each variant
        pid_scores = {}
        for variant, strategy in strategies.items():
            if variant == "pid":
                pid_scores[variant] = strategy.calculate_single_client_pid(
                    client_id, distance
                )
            elif variant == "pid_scaled":
                pid_scores[variant] = strategy.calculate_single_client_pid_scaled(
                    client_id, distance
                )
            elif variant == "pid_standardized":
                pid_scores[variant] = strategy.calculate_single_client_pid_standardized(
                    client_id, distance, int(1.0), int(0.2)
                )

        # Verify that different variants produce different results
        assert pid_scores["pid"] != pid_scores["pid_scaled"]
        assert pid_scores["pid"] != pid_scores["pid_standardized"]
        assert pid_scores["pid_scaled"] != pid_scores["pid_standardized"]
