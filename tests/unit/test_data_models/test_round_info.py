from tests.common import pytest
from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_config import StrategyConfig


class TestRoundsInfo:
    """Test suite for RoundsInfo data model"""

    def test_init_with_strategy_config(self):
        """Test RoundsInfo initialization with strategy config"""
        config = StrategyConfig(
            aggregation_strategy_keyword="trust", num_of_rounds=5, remove_clients=True
        )

        rounds_info = RoundsInfo(simulation_strategy_config=config)

        assert rounds_info.simulation_strategy_config == config
        assert isinstance(rounds_info.score_calculation_time_nanos_history, list)
        assert len(rounds_info.score_calculation_time_nanos_history) == 0

    def test_post_init_metric_lists_initialization(self):
        """Test that __post_init__ properly initializes metric lists"""
        config = StrategyConfig(remove_clients=True)
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Check plottable metrics
        expected_plottable = [
            "score_calculation_time_nanos_history",
            "aggregated_loss_history",
            "average_accuracy_history",
        ]
        assert rounds_info.plottable_metrics == expected_plottable

        # Check barable metrics
        expected_barable = [
            "removal_accuracy_history",
            "removal_precision_history",
            "removal_recall_history",
            "removal_f1_history",
            "total_fp_and_fn_history",
        ]
        assert rounds_info.barable_metrics == expected_barable

    def test_post_init_savable_metrics_with_removal(self):
        """Test savable_metrics initialization when remove_clients=True"""
        config = StrategyConfig(remove_clients=True)
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        expected_base_savable = [
            "score_calculation_time_nanos_history",
            "removal_threshold_history",
            "aggregated_loss_history",
            "average_accuracy_history",
        ]

        expected_stats_metrics = [
            "average_accuracy_history",
            "tp_history",
            "tn_history",
            "fp_history",
            "fn_history",
            "removal_accuracy_history",
            "removal_precision_history",
            "removal_recall_history",
            "removal_f1_history",
        ]

        # Should include both base and stats metrics
        assert all(
            metric in rounds_info.savable_metrics for metric in expected_base_savable
        )
        assert all(
            metric in rounds_info.savable_metrics for metric in expected_stats_metrics
        )

    def test_post_init_savable_metrics_without_removal(self):
        """Test savable_metrics initialization when remove_clients=False"""
        config = StrategyConfig(remove_clients=False)
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        expected_base_savable = [
            "score_calculation_time_nanos_history",
            "removal_threshold_history",
            "aggregated_loss_history",
            "average_accuracy_history",
            "average_accuracy_std_history",
        ]

        # Should only include base metrics, not stats metrics
        assert rounds_info.savable_metrics == expected_base_savable

    def test_add_history_entry_basic(self):
        """Test add_history_entry with basic parameters"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        rounds_info.add_history_entry(
            score_calculation_time_nanos=1000000,
            removal_threshold=0.5,
            aggregated_loss=0.3,
            average_accuracy=0.85,
        )

        assert rounds_info.score_calculation_time_nanos_history == [1000000]
        assert rounds_info.removal_threshold_history == [0.5]
        assert rounds_info.aggregated_loss_history == [0.3]
        assert rounds_info.average_accuracy_history == [0.85]

    def test_add_history_entry_multiple_rounds(self):
        """Test add_history_entry for multiple rounds"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Add data for 3 rounds
        for i in range(3):
            rounds_info.add_history_entry(
                score_calculation_time_nanos=1000000 * (i + 1),
                removal_threshold=0.5 + 0.1 * i,
                aggregated_loss=0.4 - 0.1 * i,
                average_accuracy=0.8 + 0.05 * i,
            )

        assert rounds_info.score_calculation_time_nanos_history == [
            1000000,
            2000000,
            3000000,
        ]
        assert rounds_info.removal_threshold_history == pytest.approx(
            [0.5, 0.6, 0.7], rel=1e-3
        )
        assert rounds_info.aggregated_loss_history == pytest.approx(
            [0.4, 0.3, 0.2], rel=1e-3
        )
        assert rounds_info.average_accuracy_history == pytest.approx(
            [0.8, 0.85, 0.9], rel=1e-3
        )

    def test_get_metric_by_name_valid_metrics(self):
        """Test get_metric_by_name with valid metric names"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Add some data
        rounds_info.add_history_entry(
            score_calculation_time_nanos=500000,
            removal_threshold=0.7,
            aggregated_loss=0.25,
            average_accuracy=0.9,
        )

        assert rounds_info.get_metric_by_name(
            "score_calculation_time_nanos_history"
        ) == [500000]
        assert rounds_info.get_metric_by_name("removal_threshold_history") == [0.7]
        assert rounds_info.get_metric_by_name("aggregated_loss_history") == [0.25]
        assert rounds_info.get_metric_by_name("average_accuracy_history") == [0.9]

    def test_get_metric_by_name_invalid_metric(self):
        """Test get_metric_by_name with invalid metric name"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        with pytest.raises(AttributeError):
            rounds_info.get_metric_by_name("nonexistent_metric")

    def test_append_tp_tn_fp_fn_basic(self):
        """Test append_tp_tn_fp_fn method"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        rounds_info.append_tp_tn_fp_fn(5, 3, 2, 1)

        assert rounds_info.tp_history == [5]
        assert rounds_info.tn_history == [3]
        assert rounds_info.fp_history == [2]
        assert rounds_info.fn_history == [1]

    def test_append_tp_tn_fp_fn_multiple_rounds(self):
        """Test append_tp_tn_fp_fn for multiple rounds"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Add data for 3 rounds
        test_data = [(5, 3, 2, 1), (4, 4, 1, 2), (6, 2, 3, 0)]

        for tp, tn, fp, fn in test_data:
            rounds_info.append_tp_tn_fp_fn(tp, tn, fp, fn)

        assert rounds_info.tp_history == [5, 4, 6]
        assert rounds_info.tn_history == [3, 4, 2]
        assert rounds_info.fp_history == [2, 1, 3]
        assert rounds_info.fn_history == [1, 2, 0]

    def test_calculate_additional_metrics_basic(self):
        """Test calculate_additional_metrics method"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Add test data: tp=4, tn=3, fp=2, fn=1
        rounds_info.append_tp_tn_fp_fn(4, 3, 2, 1)
        rounds_info.calculate_additional_metrics()

        # Expected calculations:
        # accuracy = (tp + tn) / (tp + tn + fp + fn) = (4 + 3) / (4 + 3 + 2 + 1) = 7/10 = 0.7
        # precision = tp / (tp + fp) = 4 / (4 + 2) = 4/6 = 0.6667
        # recall = tp / (tp + fn) = 4 / (4 + 1) = 4/5 = 0.8
        # f1 = 2 * precision * recall / (precision + recall) = 2 * 0.6667 * 0.8 / (0.6667 + 0.8) = 0.7273

        assert len(rounds_info.removal_accuracy_history) == 1
        assert rounds_info.removal_accuracy_history[0] == pytest.approx(0.7, rel=1e-3)
        assert rounds_info.removal_precision_history[0] == pytest.approx(
            0.6667, rel=1e-3
        )
        assert rounds_info.removal_recall_history[0] == pytest.approx(0.8, rel=1e-3)
        assert rounds_info.removal_f1_history[0] == pytest.approx(0.7273, rel=1e-3)
        assert rounds_info.total_fp_and_fn_history[0] == 3  # fp + fn = 2 + 1

    def test_calculate_additional_metrics_multiple_rounds(self):
        """Test calculate_additional_metrics for multiple rounds"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Add test data for 2 rounds
        rounds_info.append_tp_tn_fp_fn(4, 3, 2, 1)  # Round 1
        rounds_info.append_tp_tn_fp_fn(6, 2, 1, 1)  # Round 2
        rounds_info.calculate_additional_metrics()

        # Round 1: accuracy = 7/10 = 0.7
        # Round 2: accuracy = 8/10 = 0.8
        assert len(rounds_info.removal_accuracy_history) == 2
        assert rounds_info.removal_accuracy_history[0] == pytest.approx(0.7, rel=1e-3)
        assert rounds_info.removal_accuracy_history[1] == pytest.approx(0.8, rel=1e-3)

        # Check that all metrics have correct length
        assert len(rounds_info.removal_precision_history) == 2
        assert len(rounds_info.removal_recall_history) == 2
        assert len(rounds_info.removal_f1_history) == 2
        assert len(rounds_info.total_fp_and_fn_history) == 2

    def test_calculate_additional_metrics_edge_cases(self):
        """Test calculate_additional_metrics with edge cases"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Edge case: no false positives (perfect precision)
        rounds_info.append_tp_tn_fp_fn(5, 3, 0, 2)
        rounds_info.calculate_additional_metrics()

        # precision = tp / (tp + fp) = 5 / (5 + 0) = 1.0
        # recall = tp / (tp + fn) = 5 / (5 + 2) = 5/7 ≈ 0.714
        assert rounds_info.removal_precision_history[0] == pytest.approx(1.0, rel=1e-3)
        assert rounds_info.removal_recall_history[0] == pytest.approx(0.714, rel=1e-3)

    def test_data_consistency_across_operations(self):
        """Test data consistency across multiple operations"""
        config = StrategyConfig(remove_clients=True)
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # Add round data and tp/tn/fp/fn data
        rounds_info.add_history_entry(
            score_calculation_time_nanos=1500000,
            removal_threshold=0.6,
            aggregated_loss=0.35,
            average_accuracy=0.88,
        )

        rounds_info.append_tp_tn_fp_fn(7, 2, 1, 0)

        # Verify basic data
        assert rounds_info.score_calculation_time_nanos_history == [1500000]
        assert rounds_info.removal_threshold_history == [0.6]
        assert rounds_info.aggregated_loss_history == [0.35]
        assert rounds_info.average_accuracy_history == [0.88]

        # Verify tp/tn/fp/fn data
        assert rounds_info.tp_history == [7]
        assert rounds_info.tn_history == [2]
        assert rounds_info.fp_history == [1]
        assert rounds_info.fn_history == [0]

        # Calculate additional metrics and verify
        rounds_info.calculate_additional_metrics()

        # accuracy = (7 + 2) / (7 + 2 + 1 + 0) = 9/10 = 0.9
        assert rounds_info.removal_accuracy_history[0] == pytest.approx(0.9, rel=1e-3)

    def test_empty_initialization(self):
        """Test RoundsInfo with empty initialization"""
        config = StrategyConfig()
        rounds_info = RoundsInfo(simulation_strategy_config=config)

        # All history lists should be empty initially
        assert len(rounds_info.score_calculation_time_nanos_history) == 0
        assert len(rounds_info.removal_threshold_history) == 0
        assert len(rounds_info.aggregated_loss_history) == 0
        assert len(rounds_info.average_accuracy_history) == 0
        assert len(rounds_info.tp_history) == 0
        assert len(rounds_info.tn_history) == 0
        assert len(rounds_info.fp_history) == 0
        assert len(rounds_info.fn_history) == 0
