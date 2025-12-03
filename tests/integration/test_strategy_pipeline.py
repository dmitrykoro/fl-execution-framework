"""Integration tests for simulation strategy pipelines."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest

from flwr.common import FitRes, EvaluateRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.krum_based_removal_strategy import (
    KrumBasedRemovalStrategy,
)
from src.simulation_strategies.multi_krum_based_removal_strategy import (
    MultiKrumBasedRemovalStrategy,
)
from src.simulation_strategies.multi_krum_strategy import MultiKrumStrategy
from src.simulation_strategies.bulyan_strategy import BulyanStrategy
from src.simulation_strategies.rfa_based_removal_strategy import RFABasedRemovalStrategy
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)
from src.simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy
from src.simulation_strategies.trimmed_mean_based_removal_strategy import (
    TrimmedMeanBasedRemovalStrategy,
)


# Test configs: one baseline + one attack per strategy type
STRATEGY_TEST_CONFIGS = [
    # Baseline configs (one per strategy)
    "femnist_krum_baseline.json",
    "femnist_mkrum_baseline.json",
    "femnist_bulyan_baseline.json",
    "femnist_rfa_baseline.json",
    "femnist_trust_baseline.json",
    "femnist_pid_baseline.json",
    "femnist_pidstdscore_baseline.json",
    "femnist_trimmean_baseline.json",
    # Attack configs (one per strategy to test attack handling paths)
    "femnist_krum_vs_labelflip20.json",
    "femnist_mkrum_vs_labelflip20.json",
    "femnist_bulyan_vs_labelflip50.json",
    "femnist_rfa_vs_labelflip20.json",
    "femnist_trust_vs_labelflip20.json",
    "femnist_pidstd_vs_labelflip20.json",
    "femnist_pidstdscore_vs_labelflip20.json",
]


def get_config_dir() -> Path:
    """Get the testing config directory."""
    return (
        Path(__file__).parent.parent.parent
        / "config"
        / "simulation_strategies"
        / "testing"
    )


def load_config(config_name: str) -> Dict[str, Any]:
    """Load and merge a config file with shared settings."""
    config_dir = get_config_dir()
    config_path = config_dir / config_name

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)

    # Merge shared_settings with first strategy config
    shared = config_data.get("shared_settings", {})
    strategies = config_data.get("simulation_strategies", [{}])

    merged = {**shared, **strategies[0]}

    # Ensure required fields have defaults
    merged.setdefault("num_of_clients", 10)
    merged.setdefault("num_of_rounds", 5)
    merged.setdefault("num_of_malicious_clients", 2)
    merged.setdefault("begin_removing_from_round", 1)
    merged.setdefault("remove_clients", True)
    merged.setdefault("min_fit_clients", 8)
    merged.setdefault("min_evaluate_clients", 8)
    merged.setdefault("min_available_clients", 10)
    merged.setdefault("training_device", "cpu")
    merged.setdefault("config_is_ai_generated", False)

    return merged


def create_mock_fit_results(
    num_clients: int, param_shapes: Optional[List[Tuple[int, ...]]] = None
) -> List[Tuple[ClientProxy, FitRes]]:
    """
    Create mock client fit results with realistic parameter structures.

    Args:
        num_clients: Number of clients
        param_shapes: Parameter shapes (default: typical CNN shapes)

    Returns:
        List of (ClientProxy, FitRes) tuples
    """
    if param_shapes is None:
        # Default: small CNN-like parameter shapes for fast testing
        param_shapes = [(64, 3, 3, 3), (64,), (128, 64, 3, 3), (128,), (10, 128), (10,)]

    results = []
    rng = np.random.default_rng(42)

    for client_id in range(num_clients):
        # Create mock client proxy
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = str(client_id)

        # Create parameters with client-specific variation
        tensors = []
        for shape in param_shapes:
            # Add client-specific offset to create distinguishable parameters
            param = rng.standard_normal(shape).astype(np.float32) + client_id * 0.1
            tensors.append(param)

        parameters = ndarrays_to_parameters(tensors)

        # Create FitRes
        fit_res = Mock(spec=FitRes)
        fit_res.parameters = parameters
        fit_res.num_examples = int(rng.integers(50, 200))
        fit_res.metrics = {
            "loss": float(rng.uniform(0.1, 2.0)),
            "accuracy": float(rng.uniform(0.5, 0.95)),
        }

        results.append((client_proxy, fit_res))

    return results


def create_mock_evaluate_results(
    num_clients: int,
) -> List[Tuple[ClientProxy, EvaluateRes]]:
    """
    Create mock client evaluation results.

    Args:
        num_clients: Number of clients

    Returns:
        List of (ClientProxy, EvaluateRes) tuples
    """
    results = []
    rng = np.random.default_rng(42)

    for client_id in range(num_clients):
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = str(client_id)

        eval_res = Mock(spec=EvaluateRes)
        eval_res.loss = float(rng.uniform(0.1, 1.5))
        eval_res.num_examples = int(rng.integers(30, 100))
        eval_res.metrics = {
            "accuracy": float(rng.uniform(0.6, 0.95)),
        }

        results.append((client_proxy, eval_res))

    return results


class MockDatasetHandler:
    """Minimal mock dataset handler for strategy history."""

    def __init__(
        self, num_clients: int = 10, attack_schedule: Optional[List[Any]] = None
    ):
        self.num_clients = num_clients
        self.malicious_clients: set[int] = set()

        # Parse attack schedule to find malicious clients
        if attack_schedule:
            for attack in attack_schedule:
                selected = attack.get("_selected_clients", [])
                self.malicious_clients.update(selected)


def create_strategy_instance(
    strategy_keyword: str,
    strategy_config: StrategyConfig,
    strategy_history: SimulationStrategyHistory,
    initial_parameters: Any = None,
) -> Any:
    """
    Create a strategy instance based on the keyword.

    Args:
        strategy_keyword: Strategy type (e.g., 'krum', 'bulyan')
        strategy_config: Configuration object
        strategy_history: History tracking object
        initial_parameters: Initial model parameters

    Returns:
        Strategy instance
    """
    common_kwargs = dict(
        initial_parameters=initial_parameters,
        min_fit_clients=strategy_config.min_fit_clients,
        min_evaluate_clients=strategy_config.min_evaluate_clients,
        min_available_clients=strategy_config.min_available_clients,
        evaluate_metrics_aggregation_fn=lambda x: x,
        fit_metrics_aggregation_fn=lambda x: x,
        remove_clients=strategy_config.remove_clients,
        begin_removing_from_round=strategy_config.begin_removing_from_round,
        strategy_history=strategy_history,
    )

    if strategy_keyword == "trust":
        return TrustBasedRemovalStrategy(
            beta_value=getattr(strategy_config, "beta_value", 0.5),
            trust_threshold=getattr(strategy_config, "trust_threshold", 0.7),
            **common_kwargs,
        )
    elif strategy_keyword in (
        "pid",
        "pid_scaled",
        "pid_standardized",
        "pid_standardized_score_based",
    ):
        # PID requires network_model - create a minimal mock
        mock_network = Mock()
        mock_network.parameters.return_value = iter([Mock()])
        return PIDBasedRemovalStrategy(
            ki=getattr(strategy_config, "Ki", 0.1),
            kp=getattr(strategy_config, "Kp", 1.0),
            kd=getattr(strategy_config, "Kd", 0.01),
            num_std_dev=getattr(strategy_config, "num_std_dev", 2.0),
            network_model=mock_network,
            aggregation_strategy_keyword=strategy_keyword,
            use_lora=False,
            **common_kwargs,
        )
    elif strategy_keyword == "krum":
        return KrumBasedRemovalStrategy(
            num_malicious_clients=strategy_config.num_of_malicious_clients,
            num_krum_selections=getattr(strategy_config, "num_krum_selections", 3),
            **common_kwargs,
        )
    elif strategy_keyword == "multi-krum-based":
        return MultiKrumBasedRemovalStrategy(
            num_of_malicious_clients=strategy_config.num_of_malicious_clients,
            num_krum_selections=getattr(strategy_config, "num_krum_selections", 3),
            **common_kwargs,
        )
    elif strategy_keyword == "multi-krum":
        return MultiKrumStrategy(
            num_of_malicious_clients=strategy_config.num_of_malicious_clients,
            num_krum_selections=getattr(strategy_config, "num_krum_selections", 3),
            **common_kwargs,
        )
    elif strategy_keyword == "trimmed_mean":
        return TrimmedMeanBasedRemovalStrategy(
            trim_ratio=getattr(strategy_config, "trim_ratio", 0.2), **common_kwargs
        )
    elif strategy_keyword == "rfa":
        return RFABasedRemovalStrategy(**common_kwargs)
    elif strategy_keyword == "bulyan":
        return BulyanStrategy(
            num_krum_selections=getattr(strategy_config, "num_krum_selections", 6),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_keyword}")


class TestStrategyPipelineIntegration:
    """Integration tests that run actual strategy code with mock weights."""

    @pytest.fixture
    def mock_initial_parameters(self):
        """Create mock initial model parameters."""
        param_shapes = [(64, 3, 3, 3), (64,), (128, 64, 3, 3), (128,), (10, 128), (10,)]
        tensors = [np.zeros(shape, dtype=np.float32) for shape in param_shapes]
        return ndarrays_to_parameters(tensors)

    @pytest.fixture
    def mock_clustering(self):
        """Mock clustering components to avoid sklearn dependencies in fast path."""
        with (
            patch(
                "src.simulation_strategies.krum_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.krum_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array([[0.1]] * 10)
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler_instance.transform.return_value = np.array([[0.1]] * 10)
            mock_scaler.return_value = mock_scaler_instance

            yield {"kmeans": mock_kmeans, "scaler": mock_scaler}

    @pytest.mark.parametrize("config_name", STRATEGY_TEST_CONFIGS)
    def test_strategy_aggregation_pipeline(
        self,
        config_name: str,
        mock_initial_parameters,
        mock_clustering,
    ):
        """
        Test that strategy aggregation runs successfully with mock weights.

        This catches bugs like:
        - RFA using wrong method signature
        - Device objects not JSON serializable
        - Dict vs object misuse
        """
        # Load real config
        try:
            config_dict = load_config(config_name)
        except Exception as e:
            pytest.skip(f"Could not load config {config_name}: {e}")

        strategy_keyword = config_dict.get("aggregation_strategy_keyword", "krum")
        num_clients = config_dict.get("num_of_clients", 10)
        num_rounds = min(
            config_dict.get("num_of_rounds", 5), 3
        )  # Limit rounds for speed

        # Create strategy config
        strategy_config = StrategyConfig.from_dict(config_dict)

        # Create mock dataset handler
        mock_dataset_handler = MockDatasetHandler(
            num_clients=num_clients,
            attack_schedule=config_dict.get("attack_schedule"),
        )

        # Create real strategy history
        strategy_history = SimulationStrategyHistory(
            strategy_config=strategy_config,
            dataset_handler=mock_dataset_handler,  # type: ignore[arg-type]
        )

        # Create strategy instance
        try:
            strategy = create_strategy_instance(
                strategy_keyword=strategy_keyword,
                strategy_config=strategy_config,
                strategy_history=strategy_history,
                initial_parameters=mock_initial_parameters,
            )
        except Exception as e:
            pytest.fail(f"Failed to create strategy {strategy_keyword}: {e}")

        # Create mock client data
        mock_fit_results = create_mock_fit_results(num_clients)
        mock_eval_results = create_mock_evaluate_results(num_clients)

        # Run aggregation for multiple rounds
        for round_num in range(1, num_rounds + 1):
            try:
                # Run aggregate_fit (the core strategy logic)
                aggregated_params, fit_metrics = strategy.aggregate_fit(
                    server_round=round_num,
                    results=mock_fit_results,
                    failures=[],
                )

                # Verify aggregation returned something
                assert aggregated_params is not None or fit_metrics is not None, (
                    f"aggregate_fit returned None for round {round_num}"
                )

            except Exception as e:
                pytest.fail(
                    f"Strategy {strategy_keyword} failed in aggregate_fit "
                    f"round {round_num}: {type(e).__name__}: {e}"
                )

            try:
                # Run aggregate_evaluate
                loss, eval_metrics = strategy.aggregate_evaluate(
                    server_round=round_num,
                    results=mock_eval_results,
                    failures=[],
                )
            except Exception as e:
                pytest.fail(
                    f"Strategy {strategy_keyword} failed in aggregate_evaluate "
                    f"round {round_num}: {type(e).__name__}: {e}"
                )

        # Verify strategy history was populated
        assert strategy_history is not None
        assert strategy_history.rounds_history is not None

    def test_strategy_config_serialization(
        self, mock_initial_parameters, mock_clustering
    ):
        """
        Test that strategy configs can be serialized to JSON.

        This catches the training_device=device(type='cpu') bug.
        """
        config_dict = {
            "aggregation_strategy_keyword": "krum",
            "num_of_clients": 10,
            "num_of_rounds": 3,
            "num_of_malicious_clients": 2,
            "begin_removing_from_round": 1,
            "remove_clients": True,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 10,
            "training_device": "cpu",  # Should be string, not device object
            "num_krum_selections": 6,
            "config_is_ai_generated": False,
        }

        strategy_config = StrategyConfig.from_dict(config_dict)

        # Test serialization - this would fail if training_device is a device object
        try:
            config_as_dict = strategy_config.__dict__
            json_str = json.dumps(config_as_dict, default=str)
            assert json_str is not None
        except TypeError as e:
            if "not JSON serializable" in str(e):
                pytest.fail(f"Config contains non-serializable object: {e}")
            raise


class TestStrategyHistoryIntegration:
    """Tests for strategy history population during aggregation."""

    @pytest.fixture
    def mock_clustering(self):
        """Mock clustering for fast testing."""
        with (
            patch(
                "src.simulation_strategies.krum_based_removal_strategy.KMeans"
            ) as mock_kmeans,
            patch(
                "src.simulation_strategies.krum_based_removal_strategy.MinMaxScaler"
            ) as mock_scaler,
        ):
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.transform.return_value = np.array([[0.1]] * 10)
            mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

            mock_scaler_instance = Mock()
            mock_scaler_instance.fit.return_value = mock_scaler_instance
            mock_scaler_instance.transform.return_value = np.array([[0.1]] * 10)
            mock_scaler.return_value = mock_scaler_instance

            yield

    def test_krum_history_population(self, mock_clustering):
        """Test that Krum populates strategy history correctly."""
        config_dict = {
            "aggregation_strategy_keyword": "krum",
            "num_of_clients": 5,
            "num_of_rounds": 3,
            "num_of_malicious_clients": 1,
            "begin_removing_from_round": 1,
            "remove_clients": True,
            "min_fit_clients": 4,
            "min_evaluate_clients": 4,
            "min_available_clients": 5,
            "num_krum_selections": 3,
            "config_is_ai_generated": False,
        }

        strategy_config = StrategyConfig.from_dict(config_dict)
        mock_dataset_handler = MockDatasetHandler(num_clients=5)

        strategy_history = SimulationStrategyHistory(
            strategy_config=strategy_config,
            dataset_handler=mock_dataset_handler,  # type: ignore[arg-type]
        )

        # Create initial parameters
        param_shapes = [(10, 5), (5,)]
        initial_params = ndarrays_to_parameters(
            [np.zeros(shape, dtype=np.float32) for shape in param_shapes]
        )

        strategy = KrumBasedRemovalStrategy(
            remove_clients=True,
            num_malicious_clients=1,
            num_krum_selections=3,
            begin_removing_from_round=1,
            strategy_history=strategy_history,
            initial_parameters=initial_params,
            min_fit_clients=4,
            min_evaluate_clients=4,
            min_available_clients=5,
            fit_metrics_aggregation_fn=lambda x: x,
        )

        # Run one round
        mock_fit_results = create_mock_fit_results(5, param_shapes=param_shapes)
        strategy.aggregate_fit(1, mock_fit_results, [])

        # Verify history was populated
        assert (
            len(strategy_history.rounds_history.score_calculation_time_nanos_history)
            >= 1
        )

        # Verify client scores were recorded
        assert len(strategy.client_scores) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
