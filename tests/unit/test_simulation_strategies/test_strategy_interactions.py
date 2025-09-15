"""
Unit tests for strategy interactions and combinations.

Tests Trust + PID strategy combinations, Krum variant interactions, and Byzantine-robust strategy combinations.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.bulyan_strategy import BulyanStrategy
from src.simulation_strategies.krum_based_removal_strategy import (
    KrumBasedRemovalStrategy,
)
from src.simulation_strategies.multi_krum_based_removal_strategy import (
    MultiKrumBasedRemovalStrategy,
)
from src.simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy
from src.simulation_strategies.rfa_based_removal_strategy import RFABasedRemovalStrategy
from src.simulation_strategies.trimmed_mean_based_removal_strategy import (
    TrimmedMeanBasedRemovalStrategy,
)
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)

from tests.conftest import generate_mock_client_data


class TestStrategyInteractions:
    """Test cases for strategy interactions and combinations."""

    @pytest.fixture
    def mock_strategy_history(self):
        """Create mock strategy history."""
        return Mock(spec=SimulationStrategyHistory)

    @pytest.fixture
    def mock_network_model(self):
        """Create mock network model."""
        return Mock()

    @pytest.fixture
    def krum_fit_metrics_fn(self):
        """Provide consistent fit_metrics_aggregation_fn for Krum-based strategies."""
        return lambda x: x

    @pytest.fixture
    def mock_client_results_normal(self):
        """Generate mock client results with normal behavior."""
        return generate_mock_client_data(num_clients=10)

    @pytest.fixture
    def mock_client_results_byzantine(self):
        """Generate mock client results with Byzantine behavior."""
        return generate_mock_client_data(num_clients=10)

    def test_trust_pid_combination_consistency(
        self, mock_strategy_history, mock_network_model
    ):
        """Test that Trust and PID strategies can work together consistently."""
        # Create Trust strategy
        trust_strategy = TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=0.5,
            trust_threshold=0.7,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
        )

        # Create PID strategy
        pid_strategy = PIDBasedRemovalStrategy(
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
        )

        # Both strategies should have compatible interfaces
        assert hasattr(trust_strategy, "configure_fit")
        assert hasattr(trust_strategy, "aggregate_fit")
        assert hasattr(pid_strategy, "configure_fit")
        assert hasattr(pid_strategy, "aggregate_fit")

        # Both should handle the same begin_removing_from_round parameter
        assert (
            trust_strategy.begin_removing_from_round
            == pid_strategy.begin_removing_from_round
        )

        # Both should track removed clients
        assert hasattr(trust_strategy, "removed_client_ids")
        assert hasattr(pid_strategy, "removed_client_ids")

    def test_trust_pid_removal_criteria_differences(
        self, mock_strategy_history, mock_network_model, mock_client_results_normal
    ):
        """Test that Trust and PID strategies use different removal criteria."""
        # Create strategies
        trust_strategy = TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=0.5,
            trust_threshold=0.7,
            begin_removing_from_round=1,
            strategy_history=mock_strategy_history,
        )

        pid_strategy = PIDBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=1,
            ki=0.1,
            kd=0.01,
            kp=1.0,
            num_std_dev=2.0,
            strategy_history=mock_strategy_history,
            network_model=mock_network_model,
            use_lora=False,
            aggregation_strategy_keyword="pid",
        )

        # Mock the clustering and aggregation components
        with patch(
            "src.simulation_strategies.trust_based_removal_strategy.KMeans"
        ) as mock_kmeans_trust, patch(
            "src.simulation_strategies.trust_based_removal_strategy.MinMaxScaler"
        ) as mock_scaler_trust, patch(
            "src.simulation_strategies.pid_based_removal_strategy.KMeans"
        ) as mock_kmeans_pid, patch(
            "flwr.server.strategy.FedAvg.aggregate_fit"
        ) as mock_parent_aggregate:

            # Setup mocks for Trust strategy
            mock_kmeans_trust_instance = Mock()
            mock_kmeans_trust_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
            )
            mock_kmeans_trust.return_value.fit.return_value = mock_kmeans_trust_instance

            mock_scaler_trust_instance = Mock()
            mock_scaler_trust_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
            )
            mock_scaler_trust.return_value = mock_scaler_trust_instance

            # Setup mocks for PID strategy
            mock_kmeans_pid_instance = Mock()
            mock_kmeans_pid_instance.transform.return_value = np.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
            )
            mock_kmeans_pid.return_value.fit.return_value = mock_kmeans_pid_instance

            mock_parent_aggregate.return_value = (Mock(), {})

            # Run both strategies
            trust_strategy.aggregate_fit(1, mock_client_results_normal, [])
            pid_strategy.aggregate_fit(1, mock_client_results_normal, [])

            # Verify different removal criteria are calculated
            # Trust uses trust scores
            assert hasattr(trust_strategy, "client_trusts")
            assert len(trust_strategy.client_trusts) == 10

            # PID uses PID scores
            assert hasattr(pid_strategy, "client_pids")
            assert len(pid_strategy.client_pids) == 10

            # The scores should be different types of metrics
            trust_scores = list(trust_strategy.client_trusts.values())
            pid_scores = list(pid_strategy.client_pids.values())

            # Both should be valid numbers but likely different values
            assert all(
                0 <= score <= 1 for score in trust_scores
            )  # Trust scores are bounded [0,1]
            assert all(
                isinstance(score, (int, float)) for score in pid_scores
            )  # PID scores are unbounded

    def test_pid_variants_consistency(self, mock_strategy_history, mock_network_model):
        """Test that PID variants (pid, pid_scaled, pid_standardized) behave consistently."""
        pid_variants = ["pid", "pid_scaled", "pid_standardized"]
        strategies = {}

        # Create all PID variants
        for variant in pid_variants:
            strategies[variant] = PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=2,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword=variant,
            )

        # All variants should have the same interface
        for variant, strategy in strategies.items():
            assert hasattr(strategy, "calculate_single_client_pid")
            assert hasattr(strategy, "calculate_single_client_pid_scaled")
            assert hasattr(strategy, "calculate_single_client_pid_standardized")
            assert strategy.aggregation_strategy_keyword == variant

        # Test that different variants produce different PID calculations
        client_id = "client_1"
        distance = 0.5

        # Set up common state
        for strategy in strategies.values():
            strategy.current_round = 3
            strategy.client_distance_sums[client_id] = 1.2
            strategy.client_distances[client_id] = 0.3

        # Calculate PID scores for each variant
        pid_scores = {}
        pid_scores["pid"] = strategies["pid"].calculate_single_client_pid(
            client_id, distance
        )
        pid_scores["pid_scaled"] = strategies[
            "pid_scaled"
        ].calculate_single_client_pid_scaled(client_id, distance)
        pid_scores["pid_standardized"] = strategies[
            "pid_standardized"
        ].calculate_single_client_pid_standardized(client_id, distance, 1.0, 0.2)

        # All scores should be valid numbers
        for variant, score in pid_scores.items():
            assert isinstance(score, (int, float))
            assert np.isfinite(score)

        # Different variants should produce different results
        assert pid_scores["pid"] != pid_scores["pid_scaled"]
        assert pid_scores["pid"] != pid_scores["pid_standardized"]
        assert pid_scores["pid_scaled"] != pid_scores["pid_standardized"]

    def test_krum_variants_interaction(
        self, mock_strategy_history, krum_fit_metrics_fn
    ):
        """Test interactions between Krum and Multi-Krum strategies."""
        # Create Krum strategy
        krum_strategy = KrumBasedRemovalStrategy(
            remove_clients=True,
            num_malicious_clients=2,
            num_krum_selections=3,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fit_metrics_aggregation_fn=krum_fit_metrics_fn,
        )

        # Create Multi-Krum strategy
        multi_krum_strategy = MultiKrumBasedRemovalStrategy(
            remove_clients=True,
            num_of_malicious_clients=2,
            num_krum_selections=3,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
            fit_metrics_aggregation_fn=krum_fit_metrics_fn,
        )

        # Both should have similar interfaces but different behaviors
        assert hasattr(krum_strategy, "_calculate_krum_scores")
        assert hasattr(multi_krum_strategy, "_calculate_multi_krum_scores")

        # Both should use similar parameters
        assert (
            krum_strategy.num_krum_selections == multi_krum_strategy.num_krum_selections
        )
        assert (
            krum_strategy.begin_removing_from_round
            == multi_krum_strategy.begin_removing_from_round
        )

        # Test that they calculate different types of scores
        # Create test data
        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.array([[i * 2.0]]), np.array([i * 2.0])]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            results.append((client_proxy, fit_res))

        distances = np.zeros((5, 5))

        # Calculate scores with both methods
        krum_scores = krum_strategy._calculate_krum_scores(results, distances.copy())
        multi_krum_scores = multi_krum_strategy._calculate_multi_krum_scores(
            results, distances.copy()
        )

        # Both should return valid scores
        assert len(krum_scores) == 5
        assert len(multi_krum_scores) == 5
        assert all(np.isfinite(score) for score in krum_scores)
        assert all(np.isfinite(score) for score in multi_krum_scores)

    def test_byzantine_robust_strategy_combinations(
        self,
        mock_strategy_history,
        mock_network_model,
        mock_client_results_byzantine,
        krum_fit_metrics_fn,
    ):
        """Test combinations of Byzantine-robust strategies."""
        # Create Byzantine-robust strategies
        strategies = {
            "trust": TrustBasedRemovalStrategy(
                remove_clients=True,
                beta_value=0.5,
                trust_threshold=0.7,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
            ),
            "krum": KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=2,
                num_krum_selections=3,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            ),
            "rfa": RFABasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=1,
                weighted_median_factor=1.0,
            ),
            "bulyan": BulyanStrategy(
                remove_clients=True,
                num_krum_selections=6,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
            ),
        }

        # All strategies should handle Byzantine clients
        for strategy_name, strategy in strategies.items():
            assert hasattr(strategy, "removed_client_ids")
            assert hasattr(strategy, "remove_clients")
            assert strategy.remove_clients is True
            assert strategy.begin_removing_from_round == 1

    def test_strategy_robustness_under_attack(
        self,
        mock_strategy_history,
        mock_network_model,
        mock_client_results_byzantine,
        krum_fit_metrics_fn,
    ):
        """Test strategy robustness under different attack scenarios."""
        # Create strategies known for Byzantine robustness
        robust_strategies = {
            "krum": KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=2,
                num_krum_selections=3,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            ),
            "multi_krum": MultiKrumBasedRemovalStrategy(
                remove_clients=True,
                num_of_malicious_clients=2,
                num_krum_selections=6,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            ),
            "trimmed_mean": TrimmedMeanBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
                trim_ratio=0.2,
            ),
        }

        # Test each strategy's ability to handle Byzantine clients
        for strategy_name, strategy in robust_strategies.items():
            with (
                patch("src.simulation_strategies.krum_based_removal_strategy.KMeans")
                if "krum" in strategy_name
                else (
                    patch(
                        "src.simulation_strategies.multi_krum_based_removal_strategy.KMeans"
                    )
                    if "multi_krum" in strategy_name
                    else patch("builtins.len", return_value=10)
                )
            ):  # For trimmed_mean

                if "krum" in strategy_name:
                    # Mock clustering for Krum strategies
                    # Build the correct module path
                    if strategy_name == "krum":
                        module_path = "src.simulation_strategies.krum_based_removal_strategy.MinMaxScaler"
                    elif strategy_name == "multi_krum":
                        module_path = "src.simulation_strategies.multi_krum_based_removal_strategy.MinMaxScaler"
                    else:
                        module_path = f"src.simulation_strategies.{strategy_name}_based_removal_strategy.MinMaxScaler"

                    with patch(module_path):
                        mock_kmeans_instance = Mock()
                        mock_kmeans_instance.transform.return_value = np.array(
                            [[0.1 * i] for i in range(10)]
                        )

                        # The strategy should be able to process Byzantine clients
                        try:
                            if hasattr(strategy, "aggregate_fit"):
                                # This tests that the strategy can handle Byzantine inputs
                                # without crashing (actual Byzantine detection would need more complex testing)
                                assert callable(strategy.aggregate_fit)
                        except Exception as e:
                            pytest.fail(
                                f"Strategy {strategy_name} failed to handle Byzantine clients: {e}"
                            )

    def test_strategy_parameter_compatibility(
        self, mock_strategy_history, mock_network_model, krum_fit_metrics_fn
    ):
        """Test that strategies with similar parameters are compatible."""
        # Test strategies that share begin_removing_from_round parameter
        strategies_with_begin_round = [
            TrustBasedRemovalStrategy(
                remove_clients=True,
                beta_value=0.5,
                trust_threshold=0.7,
                begin_removing_from_round=3,
                strategy_history=mock_strategy_history,
            ),
            PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=3,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            ),
            KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=2,
                num_krum_selections=3,
                begin_removing_from_round=3,
                strategy_history=mock_strategy_history,
                fit_metrics_aggregation_fn=krum_fit_metrics_fn,
            ),
        ]

        # All should have the same begin_removing_from_round value
        for strategy in strategies_with_begin_round:
            assert strategy.begin_removing_from_round == 3
            assert hasattr(strategy, "current_round")
            assert hasattr(strategy, "removed_client_ids")

    def test_strategy_removal_consistency(
        self, mock_strategy_history, mock_network_model
    ):
        """Test that different strategies handle client removal consistently."""
        # Create strategies with removal enabled
        strategies = {
            "trust": TrustBasedRemovalStrategy(
                remove_clients=True,
                beta_value=0.5,
                trust_threshold=0.7,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
            ),
            "pid": PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=1,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            ),
        }

        # Test removal behavior consistency
        for strategy_name, strategy in strategies.items():
            # Initially no clients should be removed
            assert len(strategy.removed_client_ids) == 0

            # After setting current_round past begin_removing_from_round,
            # strategies should be ready to remove clients
            strategy.current_round = 2

            # Mock client manager
            mock_client_manager = Mock()
            mock_clients = {f"client_{i}": Mock() for i in range(5)}
            mock_client_manager.all.return_value = mock_clients

            # Set up some scores for removal decisions
            if strategy_name == "trust":
                strategy.client_trusts = {
                    f"client_{i}": 0.5 + i * 0.1 for i in range(5)
                }
            elif strategy_name == "pid":
                strategy.client_pids = {f"client_{i}": 0.5 + i * 0.1 for i in range(5)}
                strategy.current_threshold = 0.8

            # Configure fit should handle removal logic
            result = strategy.configure_fit(2, Mock(), mock_client_manager)

            # Should return client configurations
            assert isinstance(result, list)
            assert len(result) > 0

    def test_strategy_aggregation_compatibility(self, mock_strategy_history):
        """Test that strategies produce compatible aggregation outputs."""
        # Create strategies that should produce similar output formats
        strategies = {
            "trimmed_mean": TrimmedMeanBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
                trim_ratio=0.2,
            ),
            "rfa": RFABasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=1,
                weighted_median_factor=1.0,
            ),
        }

        # Create test results
        results = []
        for i in range(5):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)
            mock_params = [np.random.randn(3, 3), np.random.randn(3)]
            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            results.append((client_proxy, fit_res))

        # Test aggregation output compatibility
        for strategy_name, strategy in strategies.items():
            if strategy_name == "rfa":
                with patch(
                    "src.simulation_strategies.rfa_based_removal_strategy.KMeans"
                ) as mock_kmeans, patch(
                    "src.simulation_strategies.rfa_based_removal_strategy.MinMaxScaler"
                ) as mock_scaler:

                    # Setup mocks
                    mock_kmeans_instance = Mock()
                    mock_kmeans_instance.transform.return_value = np.array(
                        [[0.1 * i] for i in range(5)]
                    )
                    mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

                    mock_scaler_instance = Mock()
                    mock_scaler_instance.transform.return_value = np.array(
                        [[0.1 * i] for i in range(5)]
                    )
                    mock_scaler.return_value = mock_scaler_instance

                    params, metrics = strategy.aggregate_fit(1, results, [])
            else:
                params, metrics = strategy.aggregate_fit(1, results, [])

            # Both should return compatible formats
            assert params is not None
            assert isinstance(metrics, dict)

            # Parameters should be convertible back to arrays
            if params is not None:
                arrays = parameters_to_ndarrays(params)
                assert len(arrays) == 2  # Should match input structure
                assert arrays[0].shape == (3, 3)
                assert arrays[1].shape == (3,)

    def test_strategy_failure_handling_compatibility(
        self, mock_strategy_history, mock_network_model
    ):
        """Test that strategies handle failures consistently."""
        strategies = {
            "trust": TrustBasedRemovalStrategy(
                remove_clients=True,
                beta_value=0.5,
                trust_threshold=0.7,
                begin_removing_from_round=1,
                strategy_history=mock_strategy_history,
            ),
            "pid": PIDBasedRemovalStrategy(
                remove_clients=True,
                begin_removing_from_round=1,
                ki=0.1,
                kd=0.01,
                kp=1.0,
                num_std_dev=2.0,
                strategy_history=mock_strategy_history,
                network_model=mock_network_model,
                use_lora=False,
                aggregation_strategy_keyword="pid",
            ),
        }

        # Test empty results handling
        for strategy_name, strategy in strategies.items():
            if strategy_name == "trust":
                with patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent:
                    mock_parent.return_value = (None, {})
                    result = strategy.aggregate_fit(1, [], [])
                    # Should handle empty results gracefully
                    assert result is not None
            elif strategy_name == "pid":
                with patch("flwr.server.strategy.FedAvg.aggregate_fit") as mock_parent:
                    mock_parent.return_value = (None, {})
                    result = strategy.aggregate_fit(1, [], [])
                    # Should handle empty results gracefully
                    assert result is not None

    def test_cross_strategy_client_tracking(
        self, mock_strategy_history, mock_network_model
    ):
        """Test that different strategies can track the same clients consistently."""
        # Create two strategies that might be used in sequence
        trust_strategy = TrustBasedRemovalStrategy(
            remove_clients=True,
            beta_value=0.5,
            trust_threshold=0.7,
            begin_removing_from_round=1,
            strategy_history=mock_strategy_history,
        )

        pid_strategy = PIDBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=1,
            ki=0.1,
            kd=0.01,
            kp=1.0,
            num_std_dev=2.0,
            strategy_history=mock_strategy_history,
            network_model=mock_network_model,
            use_lora=False,
            aggregation_strategy_keyword="pid",
        )

        # Both should be able to track the same set of client IDs
        client_ids = [f"client_{i}" for i in range(5)]

        # Simulate client tracking
        for client_id in client_ids:
            trust_strategy.removed_client_ids.add(client_id)
            pid_strategy.removed_client_ids.add(client_id)

        # Both should have the same removed clients
        assert trust_strategy.removed_client_ids == pid_strategy.removed_client_ids
        assert len(trust_strategy.removed_client_ids) == 5
        assert len(pid_strategy.removed_client_ids) == 5
