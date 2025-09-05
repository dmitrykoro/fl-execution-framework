import pytest

from src.data_models.client_info import ClientInfo


class TestClientInfo:
    """Test suite for ClientInfo data model"""

    def test_init_basic_parameters(self):
        """Test ClientInfo initialization with basic parameters"""
        client = ClientInfo(client_id=1, num_of_rounds=5)

        assert client.client_id == 1
        assert client.num_of_rounds == 5
        assert client.is_malicious is None

    def test_init_with_malicious_flag(self):
        """Test ClientInfo initialization with malicious flag"""
        benign_client = ClientInfo(client_id=1, num_of_rounds=3, is_malicious=False)
        malicious_client = ClientInfo(client_id=2, num_of_rounds=3, is_malicious=True)

        assert benign_client.is_malicious is False
        assert malicious_client.is_malicious is True

    def test_post_init_history_lists_initialization(self):
        """Test that __post_init__ properly initializes all history lists"""
        client = ClientInfo(client_id=1, num_of_rounds=4)

        # Check that all history lists are initialized with correct length
        assert len(client.removal_criterion_history) == 4
        assert len(client.absolute_distance_history) == 4
        assert len(client.loss_history) == 4
        assert len(client.accuracy_history) == 4
        assert len(client.aggregation_participation_history) == 4

        # Check that most lists are initialized with None values
        assert all(val is None for val in client.removal_criterion_history)
        assert all(val is None for val in client.absolute_distance_history)
        assert all(val is None for val in client.loss_history)
        assert all(val is None for val in client.accuracy_history)

        # Check that aggregation_participation_history is initialized with 1s
        assert all(val == 1 for val in client.aggregation_participation_history)

    def test_post_init_rounds_list(self):
        """Test that __post_init__ properly initializes rounds list"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        assert client.rounds == [1, 2, 3]

        client_5_rounds = ClientInfo(client_id=2, num_of_rounds=5)
        assert client_5_rounds.rounds == [1, 2, 3, 4, 5]

    def test_add_history_entry_single_metric(self):
        """Test add_history_entry with single metric updates"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        # Add loss for round 1
        client.add_history_entry(current_round=1, loss=0.5)
        assert client.loss_history[0] == 0.5
        assert client.loss_history[1] is None
        assert client.loss_history[2] is None

        # Add accuracy for round 2
        client.add_history_entry(current_round=2, accuracy=0.8)
        assert client.accuracy_history[1] == 0.8
        assert client.accuracy_history[0] is None
        assert client.accuracy_history[2] is None

    def test_add_history_entry_multiple_metrics(self):
        """Test add_history_entry with multiple metrics in single call"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        client.add_history_entry(
            current_round=2,
            removal_criterion=0.7,
            absolute_distance=1.2,
            loss=0.3,
            accuracy=0.9,
            aggregation_participation=0,
        )

        # Check that all metrics were updated for round 2 (index 1)
        assert client.removal_criterion_history[1] == 0.7
        assert client.absolute_distance_history[1] == 1.2
        assert client.loss_history[1] == 0.3
        assert client.accuracy_history[1] == 0.9
        assert client.aggregation_participation_history[1] == 0

        # Check that other rounds remain unchanged
        assert client.removal_criterion_history[0] is None
        assert client.removal_criterion_history[2] is None

    def test_add_history_entry_partial_updates(self):
        """Test add_history_entry with partial metric updates"""
        client = ClientInfo(client_id=1, num_of_rounds=4)

        # Update only some metrics for round 1
        client.add_history_entry(current_round=1, loss=0.4, accuracy=0.85)

        # Update different metrics for round 3
        client.add_history_entry(
            current_round=3, removal_criterion=0.6, aggregation_participation=0
        )

        # Check round 1 updates
        assert client.loss_history[0] == 0.4
        assert client.accuracy_history[0] == 0.85
        assert client.removal_criterion_history[0] is None

        # Check round 3 updates
        assert client.removal_criterion_history[2] == 0.6
        assert client.aggregation_participation_history[2] == 0
        assert client.loss_history[2] is None

    def test_add_history_entry_round_indexing(self):
        """Test that add_history_entry correctly maps round numbers to list indices"""
        client = ClientInfo(client_id=1, num_of_rounds=5)

        # Round numbers are 1-indexed, but lists are 0-indexed
        client.add_history_entry(current_round=1, loss=0.1)  # Should go to index 0
        client.add_history_entry(current_round=3, loss=0.3)  # Should go to index 2
        client.add_history_entry(current_round=5, loss=0.5)  # Should go to index 4

        assert client.loss_history[0] == 0.1
        assert client.loss_history[1] is None
        assert client.loss_history[2] == 0.3
        assert client.loss_history[3] is None
        assert client.loss_history[4] == 0.5

    def test_add_history_entry_none_values(self):
        """Test add_history_entry with None values (should not update)"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        # Set initial value
        client.add_history_entry(current_round=1, loss=0.5)
        assert client.loss_history[0] == 0.5

        # Try to update with None (should not change)
        client.add_history_entry(current_round=1, loss=None)
        assert client.loss_history[0] == 0.5  # Should remain unchanged

    def test_get_metric_by_name_valid_metrics(self):
        """Test get_metric_by_name with valid metric names"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        # Update some metrics
        client.add_history_entry(current_round=1, loss=0.4, accuracy=0.8)
        client.add_history_entry(current_round=2, removal_criterion=0.6)

        # Test getting different metrics
        loss_history = client.get_metric_by_name("loss_history")
        accuracy_history = client.get_metric_by_name("accuracy_history")
        removal_history = client.get_metric_by_name("removal_criterion_history")

        assert loss_history == [0.4, None, None]
        assert accuracy_history == [0.8, None, None]
        assert removal_history == [None, 0.6, None]

    def test_get_metric_by_name_all_metrics(self):
        """Test get_metric_by_name for all available metrics"""
        client = ClientInfo(client_id=1, num_of_rounds=2)

        # Test all metric types
        metrics_to_test = [
            "removal_criterion_history",
            "absolute_distance_history",
            "loss_history",
            "accuracy_history",
            "aggregation_participation_history",
        ]

        for metric_name in metrics_to_test:
            metric_data = client.get_metric_by_name(metric_name)
            assert isinstance(metric_data, list)
            assert len(metric_data) == 2

    def test_get_metric_by_name_invalid_metric(self):
        """Test get_metric_by_name with invalid metric name"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        with pytest.raises(AttributeError):
            client.get_metric_by_name("nonexistent_metric")

    def test_plottable_metrics_class_attribute(self):
        """Test that plottable_metrics class attribute contains expected metrics"""
        expected_plottable = [
            "removal_criterion_history",
            "absolute_distance_history",
            "loss_history",
            "accuracy_history",
        ]

        assert ClientInfo.plottable_metrics == expected_plottable

    def test_savable_metrics_class_attribute(self):
        """Test that savable_metrics class attribute contains expected metrics"""
        expected_savable = [
            "removal_criterion_history",
            "absolute_distance_history",
            "loss_history",
            "accuracy_history",
            "aggregation_participation_history",
        ]

        assert ClientInfo.savable_metrics == expected_savable

    def test_edge_case_single_round(self):
        """Test ClientInfo with single round"""
        client = ClientInfo(client_id=1, num_of_rounds=1)

        assert len(client.loss_history) == 1
        assert client.rounds == [1]

        client.add_history_entry(current_round=1, loss=0.2, accuracy=0.95)
        assert client.loss_history[0] == 0.2
        assert client.accuracy_history[0] == 0.95

    def test_edge_case_zero_rounds(self):
        """Test ClientInfo with zero rounds"""
        client = ClientInfo(client_id=1, num_of_rounds=0)

        assert len(client.loss_history) == 0
        assert len(client.accuracy_history) == 0
        assert client.rounds == []

    def test_multiple_updates_same_round(self):
        """Test multiple updates to the same round"""
        client = ClientInfo(client_id=1, num_of_rounds=3)

        # First update
        client.add_history_entry(current_round=2, loss=0.5, accuracy=0.7)
        assert client.loss_history[1] == 0.5
        assert client.accuracy_history[1] == 0.7

        # Second update to same round (should overwrite)
        client.add_history_entry(current_round=2, loss=0.3)
        assert client.loss_history[1] == 0.3  # Updated
        assert client.accuracy_history[1] == 0.7  # Unchanged

        # Third update with different metric
        client.add_history_entry(current_round=2, accuracy=0.9)
        assert client.loss_history[1] == 0.3  # Unchanged
        assert client.accuracy_history[1] == 0.9  # Updated

    def test_data_consistency_across_operations(self):
        """Test data consistency across multiple operations"""
        client = ClientInfo(client_id=5, num_of_rounds=4, is_malicious=True)

        # Verify initial state
        assert client.client_id == 5
        assert client.is_malicious is True
        assert len(client.loss_history) == 4

        # Add data for all rounds
        for round_num in range(1, 5):
            client.add_history_entry(
                current_round=round_num,
                loss=0.1 * round_num,
                accuracy=0.8 + 0.05 * round_num,
                aggregation_participation=1 if round_num <= 2 else 0,
            )

        # Verify data consistency
        expected_loss = [0.1, 0.2, 0.3, 0.4]
        expected_accuracy = [0.85, 0.9, 0.95, 1.0]
        expected_participation = [1, 1, 0, 0]

        assert client.get_metric_by_name("loss_history") == pytest.approx(
            expected_loss, rel=1e-3
        )
        assert client.get_metric_by_name("accuracy_history") == pytest.approx(
            expected_accuracy, rel=1e-3
        )
        assert (
            client.get_metric_by_name("aggregation_participation_history")
            == expected_participation
        )
