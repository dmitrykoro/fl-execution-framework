"""
Unit tests for TrustBasedRemovalStrategy.

Tests trust score calculation algorithms, client removal logic, and threshold behaviors.
"""

from unittest.mock import patch

from tests.common import Mock, np, pytest, FitRes, ndarrays_to_parameters, ClientProxy
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)


class TestTrustBasedRemovalStrategy:
    """Test cases for TrustBasedRemovalStrategy."""

    @pytest.fixture
    def trust_strategy(self, mock_strategy_history, mock_output_directory):
        """Create TrustBasedRemovalStrategy instance for testing."""
        return TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=0.5,
            trust_threshold=0.7,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
        )

    def test_initialization(self, trust_strategy, mock_strategy_history):
        """Test TrustBasedRemovalStrategy initialization."""
        assert trust_strategy.remove_clients is True
        assert trust_strategy.beta_value == 0.5
        assert trust_strategy.trust_threshold == 0.7
        assert trust_strategy.begin_removing_from_round == 2
        assert trust_strategy.strategy_history == mock_strategy_history
        assert trust_strategy.current_round == 0
        assert trust_strategy.client_reputations == {}
        assert trust_strategy.client_trusts == {}
        assert trust_strategy.removed_client_ids == set()

    @pytest.mark.parametrize(
        "round_num,has_prior_rep,prior_rep",
        [
            pytest.param(1, False, None, id="first-round"),
            pytest.param(3, True, 0.6, id="subsequent-round"),
        ],
    )
    def test_calculate_reputation_rounds(
        self, trust_strategy, round_num, has_prior_rep, prior_rep
    ):
        """Test reputation calculation for first and subsequent rounds."""
        trust_strategy.current_round = round_num
        truth_value = np.array([0.8])
        if has_prior_rep:
            trust_strategy.client_reputations["client_1"] = prior_rep

        reputation = trust_strategy.calculate_reputation("client_1", truth_value)

        if not has_prior_rep:
            assert reputation == truth_value
        else:
            assert isinstance(reputation, (float, np.ndarray))

    @pytest.mark.parametrize(
        "prev_rep,truth_val,bound_check",
        [
            pytest.param(0.6, 0.8, "both", id="positive-truth"),
            pytest.param(0.6, 0.3, "both", id="negative-truth"),
            pytest.param(0.9, 0.95, "upper", id="upper-bound"),
            pytest.param(0.1, 0.05, "lower", id="lower-bound"),
        ],
    )
    def test_update_reputation(self, trust_strategy, prev_rep, truth_val, bound_check):
        """Test reputation update with various truth values and bounds."""
        truth_value = np.array([truth_val])
        updated_reputation = trust_strategy.update_reputation(prev_rep, truth_value, 3)

        assert isinstance(updated_reputation, float)
        if bound_check == "both":
            assert 0 <= updated_reputation <= 1
        elif bound_check == "upper":
            assert updated_reputation <= 1.0
        elif bound_check == "lower":
            assert updated_reputation >= 0.0

    @pytest.mark.parametrize(
        "round_num,has_prior_trust,prior_trust",
        [
            pytest.param(1, False, None, id="first-round"),
            pytest.param(3, True, 0.6, id="subsequent-round"),
        ],
    )
    def test_calculate_trust_rounds(
        self, trust_strategy, round_num, has_prior_trust, prior_trust
    ):
        """Test trust calculation for first and subsequent rounds."""
        trust_strategy.current_round = round_num
        if has_prior_trust:
            trust_strategy.client_trusts["client_1"] = prior_trust

        trust = trust_strategy.calculate_trust("client_1", 0.8, 0.7)

        assert isinstance(trust, float)
        assert 0 <= trust <= 1

    def test_update_trust_calculation(self, trust_strategy):
        """Test trust update calculation formula."""
        prev_trust = 0.5
        reputation = 0.8
        d = 0.7

        trust = trust_strategy.update_trust(prev_trust, reputation, d)

        # Trust should be calculated using the mathematical formula
        expected_trust_component = np.sqrt(reputation**2 + d**2) - np.sqrt(
            (1 - reputation) ** 2 + (1 - d) ** 2
        )
        expected_trust = (
            trust_strategy.beta_value * expected_trust_component
            + (1 - trust_strategy.beta_value) * prev_trust
        )

        # Apply bounds
        expected_trust = max(0.0, min(1.0, expected_trust))

        assert abs(trust - expected_trust) < 1e-6

    @pytest.mark.parametrize(
        "prev_trust,reputation,d",
        [
            pytest.param(0.9, 0.95, 0.9, id="high-values"),
            pytest.param(0.1, 0.05, 0.1, id="low-values"),
            pytest.param(0.5, 0.5, 0.5, id="medium-values"),
        ],
    )
    def test_update_trust_bounds(self, trust_strategy, prev_trust, reputation, d):
        """Test trust update respects bounds [0, 1]."""
        trust = trust_strategy.update_trust(prev_trust, reputation, d)
        assert 0 <= trust <= 1

    @patch("src.simulation_strategies.trust_based_removal_strategy.KMeans")
    @patch("src.simulation_strategies.trust_based_removal_strategy.MinMaxScaler")
    def test_aggregate_fit_clustering(
        self, mock_scaler, mock_kmeans, trust_strategy, mock_client_results
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

        with patch.object(
            trust_strategy, "aggregate_fit", wraps=trust_strategy.aggregate_fit
        ):
            # Mock the parent aggregate_fit method
            with patch(
                "flwr.server.strategy.FedAvg.aggregate_fit"
            ) as mock_parent_aggregate:
                mock_parent_aggregate.return_value = (Mock(), {})

                trust_strategy.aggregate_fit(1, mock_client_results, [])

                # Verify clustering was called
                mock_kmeans.assert_called_once()
                mock_scaler_instance.fit.assert_called_once()
                mock_scaler_instance.transform.assert_called_once()

    def test_aggregate_fit_trust_calculation(self, trust_strategy, mock_client_results):
        """Test aggregate_fit calculates trust scores for all clients."""
        with (
            patch(
                "src.simulation_strategies.trust_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.trust_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
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

            mock_parent_aggregate.return_value = (Mock(), {})

            trust_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify trust scores were calculated for all clients
            assert len(trust_strategy.client_reputations) == 5
            assert len(trust_strategy.client_trusts) == 5

            # Verify all trust scores are within bounds
            for trust in trust_strategy.client_trusts.values():
                assert 0 <= trust <= 1

    def test_configure_fit_warmup_rounds(self, trust_strategy):
        """Test configure_fit during warmup rounds."""
        trust_strategy.current_round = 1  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = trust_strategy.configure_fit(1, mock_parameters, mock_client_manager)

        # Should return all clients during warmup
        assert len(result) == 5
        assert trust_strategy.removed_client_ids == set()

    def test_configure_fit_removal_phase_first_round(self, trust_strategy):
        """Test configure_fit in first round after warmup."""
        trust_strategy.current_round = 2  # Equal to begin_removing_from_round
        trust_strategy.client_trusts = {
            "client_0": 0.9,
            "client_1": 0.3,  # Lowest trust
            "client_2": 0.7,
            "client_3": 0.8,
            "client_4": 0.6,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        result = trust_strategy.configure_fit(2, mock_parameters, mock_client_manager)

        # Should remove client with lowest trust
        assert "client_1" in trust_strategy.removed_client_ids
        assert len(result) == 5  # Still returns all clients for training

    def test_configure_fit_removal_phase_threshold(self, trust_strategy):
        """Test configure_fit removes clients below threshold."""
        trust_strategy.current_round = 3  # After begin_removing_from_round
        trust_strategy.client_trusts = {
            "client_0": 0.9,
            "client_1": 0.3,  # Below threshold
            "client_2": 0.5,  # Below threshold
            "client_3": 0.8,
            "client_4": 0.6,  # Below threshold
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        trust_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should remove clients below threshold (0.7)
        expected_removed = {"client_1", "client_2", "client_4"}
        assert trust_strategy.removed_client_ids == expected_removed

    def test_configure_fit_no_removal_when_disabled(self, trust_strategy):
        """Test configure_fit doesn't remove clients when removal is disabled."""
        trust_strategy.remove_clients = False
        trust_strategy.current_round = 3
        trust_strategy.client_trusts = {
            "client_0": 0.9,
            "client_1": 0.3,  # Below threshold but shouldn't be removed
            "client_2": 0.5,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        mock_parameters = Mock()

        trust_strategy.configure_fit(3, mock_parameters, mock_client_manager)

        # Should not remove any clients
        assert trust_strategy.removed_client_ids == set()

    @pytest.mark.parametrize(
        "begin_round",
        [
            pytest.param(1, id="round-1"),
            pytest.param(3, id="round-3"),
            pytest.param(5, id="round-5"),
        ],
    )
    def test_begin_removing_from_round_parameter(
        self, mock_strategy_history, begin_round
    ):
        """Test begin_removing_from_round parameter handling."""
        strategy = TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=0.5,
            trust_threshold=0.7,
            begin_removing_from_round=begin_round,
            strategy_history=mock_strategy_history,
        )

        assert strategy.begin_removing_from_round == begin_round

        strategy.current_round = begin_round - 1
        mock_client_manager = Mock()
        mock_clients = {"client_0": Mock(), "client_1": Mock()}
        mock_client_manager.all.return_value = mock_clients

        result = strategy.configure_fit(1, Mock(), mock_client_manager)

        # Should not remove clients during warmup
        assert strategy.removed_client_ids == set()
        assert len(result) == 2

    @pytest.mark.parametrize(
        "beta",
        [
            pytest.param(0.1, id="beta-0.1"),
            pytest.param(0.5, id="beta-0.5"),
            pytest.param(0.9, id="beta-0.9"),
        ],
    )
    def test_beta_value_parameter_effect(self, mock_strategy_history, beta):
        """Test beta_value parameter affects trust and reputation calculations."""
        strategy = TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=beta,
            trust_threshold=0.7,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
        )

        reputation = strategy.update_reputation(0.6, np.array([0.8]), 3)
        assert 0 <= reputation <= 1

        trust = strategy.update_trust(0.5, 0.8, 0.7)
        assert 0 <= trust <= 1

    @pytest.mark.parametrize(
        "threshold,expected_removed",
        [
            pytest.param(0.3, {"client_2"}, id="threshold-0.3"),
            pytest.param(0.7, {"client_1", "client_2"}, id="threshold-0.7"),
            pytest.param(0.9, {"client_0", "client_1", "client_2"}, id="threshold-0.9"),
        ],
    )
    def test_trust_threshold_parameter_effect(
        self, mock_strategy_history, threshold, expected_removed
    ):
        """Test trust_threshold parameter affects client removal."""
        strategy = TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=0.5,
            trust_threshold=threshold,
            begin_removing_from_round=1,
            strategy_history=mock_strategy_history,
        )

        strategy.current_round = 2
        strategy.client_trusts = {
            "client_0": 0.8,
            "client_1": 0.5,
            "client_2": 0.2,
        }

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        strategy.configure_fit(2, Mock(), mock_client_manager)

        assert strategy.removed_client_ids == expected_removed

    def test_strategy_history_integration(self, trust_strategy, mock_client_results):
        """Test integration with strategy history."""
        with (
            patch(
                "src.simulation_strategies.trust_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.trust_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
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

            mock_parent_aggregate.return_value = (Mock(), {})

            trust_strategy.aggregate_fit(1, mock_client_results, [])

            # Verify strategy history methods were called
            assert (
                trust_strategy.strategy_history.insert_single_client_history_entry.call_count
                == 5
            )
            trust_strategy.strategy_history.insert_round_history_entry.assert_called_once()

    def test_edge_case_empty_results(self, trust_strategy):
        """Test handling of empty results."""
        with patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:
            mock_parent_aggregate.return_value = (None, {})

            result = trust_strategy.aggregate_fit(1, [], [])

            # Should handle empty results gracefully
            assert result is not None

    def test_edge_case_single_client(self, trust_strategy):
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
                "src.simulation_strategies.trust_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.trust_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
            patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent_aggregate,
        ):
            # Setup mocks
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array([[0.1]])
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.transform.return_value = np.array([[0.1]])
            mock_scaler.return_value = mock_scaler_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            trust_strategy.aggregate_fit(1, single_result, [])

            # Should handle single client gracefully
            assert len(trust_strategy.client_trusts) == 1
            assert len(trust_strategy.client_reputations) == 1
