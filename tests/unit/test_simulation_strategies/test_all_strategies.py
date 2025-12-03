"""Consolidated parametrized tests for all simulation strategies."""

from unittest.mock import Mock

import pytest

STRATEGY_IMPORTS = {
    "trust": "src.simulation_strategies.trust_based_removal_strategy.TrustBasedRemovalStrategy",
    "krum": "src.simulation_strategies.krum_based_removal_strategy.KrumBasedRemovalStrategy",
    "multi_krum": "src.simulation_strategies.multi_krum_strategy.MultiKrumStrategy",
    "multi_krum_removal": "src.simulation_strategies.multi_krum_based_removal_strategy.MultiKrumBasedRemovalStrategy",
    "bulyan": "src.simulation_strategies.bulyan_strategy.BulyanStrategy",
    "pid": "src.simulation_strategies.pid_based_removal_strategy.PIDBasedRemovalStrategy",
    "trimmed_mean": "src.simulation_strategies.trimmed_mean_based_removal_strategy.TrimmedMeanBasedRemovalStrategy",
    "rfa": "src.simulation_strategies.rfa_based_removal_strategy.RFABasedRemovalStrategy",
}


def get_strategy_class(name: str):
    """Dynamically import and return strategy class."""
    import importlib

    module_path, class_name = STRATEGY_IMPORTS[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# Strategy initialization parameters (tuples for immutability)
STRATEGY_INIT_CONFIGS = (
    pytest.param(
        "trust",
        {
            "remove_clients": True,
            "beta_value": 0.5,
            "trust_threshold": 0.7,
            "begin_removing_from_round": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "removed_client_ids"},
        id="trust-strategy",
    ),
    pytest.param(
        "krum",
        {
            "remove_clients": True,
            "num_malicious_clients": 2,
            "num_krum_selections": 3,
            "begin_removing_from_round": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "fit_metrics_aggregation_fn": lambda x: x,
        },
        {"remove_clients", "current_round", "client_scores", "removed_client_ids"},
        id="krum-strategy",
    ),
    pytest.param(
        "multi_krum",
        {
            "remove_clients": True,
            "num_of_malicious_clients": 2,
            "num_krum_selections": 3,
            "begin_removing_from_round": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "client_scores", "removed_client_ids"},
        id="multi-krum-strategy",
    ),
    pytest.param(
        "bulyan",
        {
            "remove_clients": True,
            "num_krum_selections": 13,
            "begin_removing_from_round": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "client_scores", "removed_client_ids"},
        id="bulyan-strategy",
    ),
    pytest.param(
        "trimmed_mean",
        {
            "remove_clients": True,
            "begin_removing_from_round": 2,
            "trim_ratio": 0.2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "client_scores"},
        id="trimmed-mean-strategy",
    ),
    pytest.param(
        "rfa",
        {
            "remove_clients": True,
            "begin_removing_from_round": 2,
            "weighted_median_factor": 1.0,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "removed_client_ids", "client_scores"},
        id="rfa-strategy",
    ),
)


# PID strategy requires additional network_model parameter
PID_INIT_CONFIGS = (
    pytest.param(
        "pid",
        {
            "remove_clients": True,
            "begin_removing_from_round": 2,
            "ki": 0.1,
            "kd": 0.01,
            "kp": 1.0,
            "num_std_dev": 2.0,
            "use_lora": False,
            "aggregation_strategy_keyword": "pid",
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "client_pids", "removed_client_ids"},
        id="pid-strategy",
    ),
    pytest.param(
        "pid",
        {
            "remove_clients": True,
            "begin_removing_from_round": 2,
            "ki": 0.1,
            "kd": 0.01,
            "kp": 1.0,
            "num_std_dev": 2.0,
            "use_lora": False,
            "aggregation_strategy_keyword": "pid_scaled",
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "client_pids", "removed_client_ids"},
        id="pid-scaled-strategy",
    ),
    pytest.param(
        "pid",
        {
            "remove_clients": True,
            "begin_removing_from_round": 2,
            "ki": 0.1,
            "kd": 0.01,
            "kp": 1.0,
            "num_std_dev": 2.0,
            "use_lora": False,
            "aggregation_strategy_keyword": "pid_standardized",
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
        },
        {"remove_clients", "current_round", "client_pids", "removed_client_ids"},
        id="pid-standardized-strategy",
    ),
)


# Begin removing from round test values
BEGIN_REMOVING_ROUNDS = (1, 2, 3, 5)


class TestAllStrategiesInitialization:
    """Consolidated initialization tests for all strategies using parameterization."""

    @pytest.mark.parametrize(
        "strategy_name,init_params,expected_attrs", STRATEGY_INIT_CONFIGS
    )
    def test_strategy_common_initialization(
        self,
        strategy_name,
        init_params,
        expected_attrs,
        mock_strategy_history,
        mock_output_directory,
    ):
        """Test common initialization attributes across all strategies."""
        strategy_class = get_strategy_class(strategy_name)
        strategy = strategy_class(strategy_history=mock_strategy_history, **init_params)

        # Common assertions for all strategies
        assert strategy.remove_clients == init_params["remove_clients"]
        assert strategy.current_round == 0

        # Verify expected attributes exist
        for attr in expected_attrs:
            assert hasattr(strategy, attr), (
                f"Strategy {strategy_name} missing attribute: {attr}"
            )

        # Verify removed_client_ids is empty set if present
        if hasattr(strategy, "removed_client_ids"):
            assert strategy.removed_client_ids == set()

    @pytest.mark.parametrize(
        "strategy_name,init_params,expected_attrs", PID_INIT_CONFIGS
    )
    def test_pid_strategy_variants_initialization(
        self,
        strategy_name,
        init_params,
        expected_attrs,
        mock_strategy_history,
        mock_network_model,
        mock_output_directory,
    ):
        """Test PID strategy variants initialization (requires network_model)."""
        strategy_class = get_strategy_class(strategy_name)
        strategy = strategy_class(
            strategy_history=mock_strategy_history,
            network_model=mock_network_model,
            **init_params,
        )

        assert strategy.remove_clients == init_params["remove_clients"]
        assert strategy.current_round == 0
        assert (
            strategy.aggregation_strategy_keyword
            == init_params["aggregation_strategy_keyword"]
        )

        # Verify expected attributes exist
        for attr in expected_attrs:
            assert hasattr(strategy, attr), f"PID strategy missing attribute: {attr}"


class TestAllStrategiesWarmupBehavior:
    """Consolidated warmup behavior tests using parameterization."""

    @pytest.mark.parametrize("strategy_name,init_params,_", STRATEGY_INIT_CONFIGS)
    @pytest.mark.parametrize("warmup_round", [1])
    def test_configure_fit_warmup_returns_all_clients(
        self,
        strategy_name,
        init_params,
        _,
        warmup_round,
        mock_strategy_history,
        mock_output_directory,
    ):
        """Test that warmup rounds return all clients without removal."""
        strategy_class = get_strategy_class(strategy_name)
        strategy = strategy_class(strategy_history=mock_strategy_history, **init_params)
        strategy.current_round = warmup_round  # Before begin_removing_from_round

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(5)}
        mock_client_manager.all.return_value = mock_clients

        result = strategy.configure_fit(warmup_round, Mock(), mock_client_manager)

        # Should return all clients during warmup
        assert len(result) == 5
        # Not all strategies have removed_client_ids (e.g., trimmed_mean)
        if hasattr(strategy, "removed_client_ids"):
            assert strategy.removed_client_ids == set()


class TestAllStrategiesRemovalDisabled:
    """Consolidated tests for removal-disabled behavior."""

    @pytest.mark.parametrize("strategy_name,init_params,_", STRATEGY_INIT_CONFIGS)
    def test_no_removal_when_disabled(
        self,
        strategy_name,
        init_params,
        _,
        mock_strategy_history,
        mock_output_directory,
    ):
        """Test that no clients are removed when remove_clients=False."""
        strategy_class = get_strategy_class(strategy_name)

        # Create strategy with removal disabled
        disabled_params = {**init_params, "remove_clients": False}
        strategy = strategy_class(
            strategy_history=mock_strategy_history, **disabled_params
        )
        strategy.current_round = 5  # After begin_removing_from_round

        # Set up mock scores (different strategies use different score attributes)
        if hasattr(strategy, "client_scores"):
            strategy.client_scores = {f"client_{i}": float(i) for i in range(3)}
        if hasattr(strategy, "client_trusts"):
            strategy.client_trusts = {f"client_{i}": 0.3 + i * 0.1 for i in range(3)}
        if hasattr(strategy, "client_pids"):
            strategy.client_pids = {f"client_{i}": float(i) for i in range(3)}

        mock_client_manager = Mock()
        mock_clients = {f"client_{i}": Mock() for i in range(3)}
        mock_client_manager.all.return_value = mock_clients

        strategy.configure_fit(5, Mock(), mock_client_manager)

        # Should not remove any clients when disabled
        # Not all strategies have removed_client_ids (e.g., trimmed_mean)
        if hasattr(strategy, "removed_client_ids"):
            assert strategy.removed_client_ids == set()


class TestAllStrategiesBeginRemovingRound:
    """Consolidated tests for begin_removing_from_round parameter."""

    @pytest.mark.parametrize(
        "strategy_name,init_params,_", STRATEGY_INIT_CONFIGS[:4]
    )  # Subset for faster tests
    @pytest.mark.parametrize("begin_round", BEGIN_REMOVING_ROUNDS)
    def test_begin_removing_round_parameter_handling(
        self,
        strategy_name,
        init_params,
        _,
        begin_round,
        mock_strategy_history,
        mock_output_directory,
    ):
        """Test begin_removing_from_round parameter is respected."""
        strategy_class = get_strategy_class(strategy_name)

        params = {**init_params, "begin_removing_from_round": begin_round}
        strategy = strategy_class(strategy_history=mock_strategy_history, **params)

        assert strategy.begin_removing_from_round == begin_round

        # During warmup (before begin_round), no clients should be removed
        strategy.current_round = begin_round - 1

        mock_client_manager = Mock()
        mock_clients = {"client_0": Mock(), "client_1": Mock()}
        mock_client_manager.all.return_value = mock_clients

        result = strategy.configure_fit(begin_round - 1, Mock(), mock_client_manager)

        assert len(result) == 2
        assert strategy.removed_client_ids == set()
