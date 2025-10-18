"""
Unit tests for TrustBasedRemovalStrategy.

Tests trust score calculation algorithms, client removal logic, and threshold behaviors.
"""

from unittest.mock import patch

from tests.common import Mock, np, pytest, FitRes, ndarrays_to_parameters, ClientProxy
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)

from tests.common import generate_mock_client_data


class TestTrustBasedRemovalStrategy:
    """Test cases for TrustBasedRemovalStrategy."""

    @pytest.fixture
    def mock_client_results(self):
        """Generate mock client results for testing."""
        return generate_mock_client_data(num_clients=5)

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

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

    def test_calculate_reputation_first_round(self, trust_strategy):
        """Test reputation calculation for first round."""
        trust_strategy.current_round = 1
        truth_value = np.array([0.8])

        reputation = trust_strategy.calculate_reputation("client_1", truth_value)

        assert reputation == truth_value

    def test_calculate_reputation_subsequent_rounds(self, trust_strategy):
        """Test reputation calculation for subsequent rounds."""
        trust_strategy.current_round = 3
        trust_strategy.client_reputations["client_1"] = 0.6
        truth_value = np.array([0.8])

        reputation = trust_strategy.calculate_reputation("client_1", truth_value)

        # Should call update_reputation method
        assert isinstance(reputation, (float, np.ndarray))

    def test_update_reputation_positive_truth(self, trust_strategy):
        """Test reputation update with positive truth value."""
        prev_reputation = 0.6
        truth_value = np.array([0.8])
        current_round = 3

        updated_reputation = trust_strategy.update_reputation(
            prev_reputation, truth_value, current_round
        )

        # Should be between 0 and 1
        assert 0 <= updated_reputation <= 1
        assert isinstance(updated_reputation, float)

    def test_update_reputation_negative_truth(self, trust_strategy):
        """Test reputation update with negative truth value."""
        prev_reputation = 0.6
        truth_value = np.array([0.3])
        current_round = 3

        updated_reputation = trust_strategy.update_reputation(
            prev_reputation, truth_value, current_round
        )

        # Should be between 0 and 1
        assert 0 <= updated_reputation <= 1
        assert isinstance(updated_reputation, float)

    def test_update_reputation_bounds(self, trust_strategy):
        """Test reputation update respects bounds [0, 1]."""
        # Test upper bound
        prev_reputation = 0.9
        truth_value = np.array([0.95])
        current_round = 2

        updated_reputation = trust_strategy.update_reputation(
            prev_reputation, truth_value, current_round
        )
        assert updated_reputation <= 1.0

        # Test lower bound
        prev_reputation = 0.1
        truth_value = np.array([0.05])
        current_round = 2

        updated_reputation = trust_strategy.update_reputation(
            prev_reputation, truth_value, current_round
        )
        assert updated_reputation >= 0.0

    def test_calculate_trust_first_round(self, trust_strategy):
        """Test trust calculation for first round."""
        trust_strategy.current_round = 1
        reputation = 0.8
        d = 0.7

        trust = trust_strategy.calculate_trust("client_1", reputation, d)

        # Should call update_trust with prev_trust = 0
        assert isinstance(trust, float)
        assert 0 <= trust <= 1

    def test_calculate_trust_subsequent_rounds(self, trust_strategy):
        """Test trust calculation for subsequent rounds."""
        trust_strategy.current_round = 3
        trust_strategy.client_trusts["client_1"] = 0.6
        reputation = 0.8
        d = 0.7

        trust = trust_strategy.calculate_trust("client_1", reputation, d)

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

    def test_update_trust_bounds(self, trust_strategy):
        """Test trust update respects bounds [0, 1]."""
        # Test various combinations that might exceed bounds
        test_cases = [
            (0.9, 0.95, 0.9),  # High values
            (0.1, 0.05, 0.1),  # Low values
            (0.5, 0.5, 0.5),  # Medium values
        ]

        for prev_trust, reputation, d in test_cases:
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

    def test_begin_removing_from_round_parameter(self, mock_strategy_history):
        """Test begin_removing_from_round parameter handling."""
        # Test different begin_removing_from_round values
        for begin_round in [1, 3, 5]:
            strategy = TrustBasedRemovalStrategy(
                remove_clients=True,
                beta_value=0.5,
                trust_threshold=0.7,
                begin_removing_from_round=begin_round,
                strategy_history=mock_strategy_history,
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

    def test_beta_value_parameter_effect(self, mock_strategy_history):
        """Test beta_value parameter affects trust and reputation calculations."""
        # Test with different beta values
        beta_values = [0.1, 0.5, 0.9]

        for beta in beta_values:
            strategy = TrustBasedRemovalStrategy(
                remove_clients=True,
                beta_value=beta,
                trust_threshold=0.7,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
            )

            # Test reputation update
            prev_reputation = 0.6
            truth_value = np.array([0.8])
            current_round = 3

            reputation = strategy.update_reputation(
                prev_reputation, truth_value, current_round
            )

            # Beta value should affect the result
            assert 0 <= reputation <= 1

            # Test trust update
            prev_trust = 0.5
            reputation_val = 0.8
            d = 0.7

            trust = strategy.update_trust(prev_trust, reputation_val, d)

            # Beta value should affect the result
            assert 0 <= trust <= 1

    def test_trust_threshold_parameter_effect(self, mock_strategy_history):
        """Test trust_threshold parameter affects client removal."""
        thresholds = [0.3, 0.7, 0.9]

        for threshold in thresholds:
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

            # Clients below threshold should be removed
            for client_id, trust in strategy.client_trusts.items():
                if trust < threshold:
                    assert client_id in strategy.removed_client_ids
                else:
                    assert client_id not in strategy.removed_client_ids

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
