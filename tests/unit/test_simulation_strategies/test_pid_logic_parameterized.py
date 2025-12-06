import pytest
from unittest.mock import Mock
from src.simulation_strategies.pid_based_removal_strategy import PIDBasedRemovalStrategy


class TestPIDLogicParameterized:
    """Tests PID logic calculation across multiple strategy variants."""

    @pytest.fixture
    def mock_deps(self):
        """Creates common mocks for all strategy instances.

        Returns:
            dict: Dictionary containing mock objects.
        """
        return {"strategy_history": Mock(), "network_model": Mock()}

    @pytest.mark.parametrize(
        "variant, round_num, distance, prev_state, extra_args, expected_calc",
        [
            ("pid", 1, 0.5, {}, {}, lambda p, d: d * p.kp),
            ("pid_scaled", 1, 0.5, {}, {}, lambda p, d: d * p.kp),
            ("pid_standardized", 1, 0.5, {}, {}, lambda p, d: d * p.kp),
            (
                "pid",
                2,
                0.5,
                {"sum": 1.0, "dist": 0.2},
                {},
                lambda p, d: (d * p.kp) + (1.0 * p.ki) + (p.kd * (d - 0.2)),
            ),
            (
                "pid_scaled",
                2,
                0.5,
                {"sum": 1.0, "dist": 0.2},
                {},
                lambda p, d: (d * p.kp) + ((1.0 * p.ki) / 2) + (p.kd * (d - 0.2)),
            ),
            (
                "pid_standardized",
                2,
                0.5,
                {"sum": 1.5, "dist": 0.2},
                {"avg_sum": 1.0, "sum_std_dev": 0.5},
                lambda p, d: (d * p.kp)
                + (((1.5 - 1.0) / 0.5) * p.ki)
                + (p.kd * (d - 0.2)),
            ),
            (
                "pid_standardized",
                2,
                0.5,
                {"sum": 1.5, "dist": 0.2},
                {"avg_sum": 1.0, "sum_std_dev": 0.0},
                lambda p, d: (d * p.kp) + 0 + (p.kd * (d - 0.2)),
            ),
        ],
    )
    def test_calculate_pid_logic(
        self,
        mock_deps,
        variant,
        round_num,
        distance,
        prev_state,
        extra_args,
        expected_calc,
    ):
        """Verifies PID calculation logic across variants.

        Args:
            mock_deps: Fixture containing mock dependencies.
            variant: The PID variant strategy.
            round_num: The current simulation round.
            distance: The distance metric.
            prev_state: Previous state dictionary.
            extra_args: Additional arguments for standardized PID.
            expected_calc: Lambda function for expected result calculation.
        """
        strategy = PIDBasedRemovalStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            ki=0.1,
            kd=0.01,
            kp=1.0,
            num_std_dev=2.0,
            strategy_history=mock_deps["strategy_history"],
            network_model=mock_deps["network_model"],
            use_lora=False,
            aggregation_strategy_keyword=variant,
        )
        strategy.current_round = round_num

        client_id = "test_client"
        if prev_state:
            strategy.client_distance_sums[client_id] = prev_state.get("sum", 0.0)
            strategy.client_distances[client_id] = prev_state.get("dist", 0.0)

        result = 0.0
        if variant == "pid":
            result = strategy.calculate_single_client_pid(client_id, distance)
        elif variant == "pid_scaled":
            result = strategy.calculate_single_client_pid_scaled(client_id, distance)
        elif variant == "pid_standardized":
            result = strategy.calculate_single_client_pid_standardized(
                client_id,
                distance,
                extra_args.get("avg_sum", 0.0),
                extra_args.get("sum_std_dev", 0.0),
            )

        expected_value = expected_calc(strategy, distance)
        assert abs(result - expected_value) < 1e-6, (
            f"Failed for {variant} at round {round_num}. Expected {expected_value}, got {result}"
        )
