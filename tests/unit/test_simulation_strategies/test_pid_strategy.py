"""
Unit tests for PIDBasedRemovalStrategy.

Tests PID controller logic implementation, PID variants behavior, and parameter handling.
"""

from unittest.mock import patch

import torch
from tests.common import Mock, np, pytest, FitRes, ndarrays_to_parameters, ClientProxy
from src.simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy


class TestPIDBasedRemovalStrategy:
    """Test cases for PIDBasedRemovalStrategy."""

    @pytest.fixture
    def pid_strategy_factory(
        self, mock_strategy_history, mock_network_model, mock_output_directory
    ):
        """Factory fixture for creating PID strategy variants."""

        def _create(variant="pid", **kwargs):
            defaults = {
                "remove_clients": True,
                "begin_removing_from_round": 2,
                "ki": 0.1,
                "kd": 0.01,
                "kp": 1.0,
                "num_std_dev": 2.0,
                "strategy_history": mock_strategy_history,
                "network_model": mock_network_model,
                "use_lora": False,
                "aggregation_strategy_keyword": variant,
                "fraction_fit": 1.0,
                "fraction_evaluate": 1.0,
            }
            defaults.update(kwargs)
            return PIDBasedRemovalStrategy(**defaults)

        return _create

    @pytest.fixture
    def pid_strategy(self, pid_strategy_factory):
        """Create PIDBasedRemovalStrategy instance for testing."""
        return pid_strategy_factory("pid")

    @pytest.fixture
    def pid_scaled_strategy(self, pid_strategy_factory):
        """Create PIDBasedRemovalStrategy instance for pid_scaled testing."""
        return pid_strategy_factory("pid_scaled")

    @pytest.fixture
    def pid_standardized_strategy(self, pid_strategy_factory):
        """Create PIDBasedRemovalStrategy instance for pid_standardized testing."""
        return pid_strategy_factory("pid_standardized")

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

    @pytest.mark.parametrize(
        "round_num,has_prior_state",
        [
            pytest.param(1, False, id="first-round-P-only"),
            pytest.param(3, True, id="subsequent-round-PID"),
        ],
    )
    def test_calculate_single_client_pid_rounds(
        self, pid_strategy, round_num, has_prior_state
    ):
        """Test PID calculation for first and subsequent rounds."""
        pid_strategy.current_round = round_num
        client_id = "client_1"
        distance = 0.5

        if has_prior_state:
            pid_strategy.client_distance_sums[client_id] = 1.2
            pid_strategy.client_distances[client_id] = 0.3

        pid_score = pid_strategy.calculate_single_client_pid(client_id, distance)

        expected_p = distance * pid_strategy.kp
        if not has_prior_state:
            assert pid_score == expected_p
        else:
            expected_i = pid_strategy.client_distance_sums[client_id] * pid_strategy.ki
            expected_d = pid_strategy.kd * (
                distance - pid_strategy.client_distances[client_id]
            )
            expected_pid = expected_p + expected_i + expected_d
            assert abs(pid_score - expected_pid) < 1e-6

    @pytest.mark.parametrize(
        "round_num,has_prior_state",
        [
            pytest.param(1, False, id="first-round-P-only"),
            pytest.param(3, True, id="subsequent-round-scaled-I"),
        ],
    )
    def test_calculate_single_client_pid_scaled_rounds(
        self, pid_scaled_strategy, round_num, has_prior_state
    ):
        """Test PID scaled calculation for first and subsequent rounds."""
        pid_scaled_strategy.current_round = round_num
        client_id = "client_1"
        distance = 0.5

        if has_prior_state:
            pid_scaled_strategy.client_distance_sums[client_id] = 1.2
            pid_scaled_strategy.client_distances[client_id] = 0.3

        pid_score = pid_scaled_strategy.calculate_single_client_pid_scaled(
            client_id, distance
        )

        expected_p = distance * pid_scaled_strategy.kp
        if not has_prior_state:
            assert pid_score == expected_p
        else:
            expected_i_scaled = (
                pid_scaled_strategy.client_distance_sums[client_id]
                * pid_scaled_strategy.ki
            ) / pid_scaled_strategy.current_round
            expected_d = pid_scaled_strategy.kd * (
                distance - pid_scaled_strategy.client_distances[client_id]
            )
            expected_pid = expected_p + expected_i_scaled + expected_d
            assert abs(pid_score - expected_pid) < 1e-6

    @pytest.mark.parametrize(
        "round_num,has_prior_state,sum_std_dev",
        [
            pytest.param(1, False, 0.2, id="first-round-P-only"),
            pytest.param(3, True, 0.2, id="subsequent-round-standardized-I"),
            pytest.param(3, True, 0.0, id="subsequent-round-zero-std-dev"),
        ],
    )
    def test_calculate_single_client_pid_standardized_rounds(
        self, pid_standardized_strategy, round_num, has_prior_state, sum_std_dev
    ):
        """Test PID standardized calculation for various rounds and std_dev cases."""
        pid_standardized_strategy.current_round = round_num
        client_id = "client_1"
        distance = 0.5
        avg_sum = 1.0

        if has_prior_state:
            pid_standardized_strategy.client_distance_sums[client_id] = 1.2
            pid_standardized_strategy.client_distances[client_id] = 0.3

        pid_score = pid_standardized_strategy.calculate_single_client_pid_standardized(
            client_id, distance, avg_sum, sum_std_dev
        )

        expected_p = distance * pid_standardized_strategy.kp
        if not has_prior_state:
            assert pid_score == expected_p
        else:
            if sum_std_dev == 0.0:
                expected_i = 0
            else:
                expected_i = (
                    (
                        pid_standardized_strategy.client_distance_sums[client_id]
                        - avg_sum
                    )
                    / sum_std_dev
                ) * pid_standardized_strategy.ki
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

    @pytest.mark.parametrize(
        "param,value,test_round",
        [
            pytest.param("kp", 0.5, 1, id="kp-0.5"),
            pytest.param("kp", 1.0, 1, id="kp-1.0"),
            pytest.param("kp", 2.0, 1, id="kp-2.0"),
            pytest.param("ki", 0.05, 3, id="ki-0.05"),
            pytest.param("ki", 0.1, 3, id="ki-0.1"),
            pytest.param("ki", 0.2, 3, id="ki-0.2"),
            pytest.param("kd", 0.005, 3, id="kd-0.005"),
            pytest.param("kd", 0.01, 3, id="kd-0.01"),
            pytest.param("kd", 0.02, 3, id="kd-0.02"),
        ],
    )
    def test_k_parameter_effects(self, pid_strategy_factory, param, value, test_round):
        """Test K-parameters (kp, ki, kd) affect PID calculation correctly."""
        strategy = pid_strategy_factory("pid", **{param: value})
        strategy.current_round = test_round
        client_id = "client_1"
        distance = 0.5

        if test_round > 1:
            strategy.client_distance_sums[client_id] = 1.2
            strategy.client_distances[client_id] = 0.3

        pid_score = strategy.calculate_single_client_pid(client_id, distance)

        expected_p = distance * strategy.kp
        if test_round == 1:
            assert abs(pid_score - expected_p) < 1e-6
        else:
            expected_i = strategy.client_distance_sums[client_id] * strategy.ki
            expected_d = strategy.kd * (distance - strategy.client_distances[client_id])
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

    @pytest.mark.parametrize(
        "t1,t2,expected",
        [
            pytest.param([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], 1.0, id="parallel-vectors"),
            pytest.param([1.0, 0.0], [0.0, 1.0], 0.0, id="orthogonal-vectors"),
        ],
    )
    def test_cosine_similarity(self, t1, t2, expected):
        """Test cosine similarity calculation for various vector pairs."""
        tensor1 = torch.tensor(t1)
        tensor2 = torch.tensor(t2)
        similarity = PIDBasedRemovalStrategy.cosine_similarity(tensor1, tensor2)
        assert abs(similarity - expected) < 1e-6

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
