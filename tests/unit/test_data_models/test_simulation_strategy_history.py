import sys
from unittest.mock import MagicMock, Mock

import pytest
from src.data_models.client_info import ClientInfo
from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.dataset_handlers.dataset_handler import DatasetHandler

# Mock cv2 before importing modules that depend on it
sys.modules["cv2"] = MagicMock()


class TestSimulationStrategyHistory:
    """Test suite for SimulationStrategyHistory data model"""

    def test_init_basic(self):
        """Test SimulationStrategyHistory initialization with basic parameters"""
        # Create mock dataset handler
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {1, 3}

        config = StrategyConfig(
            aggregation_strategy_keyword="trust",
            num_of_rounds=3,
            num_of_clients=5,
            remove_clients=True,
        )

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )

        assert history.strategy_config == config
        assert history.dataset_handler == mock_dataset_handler
        assert isinstance(history.rounds_history, RoundsInfo)
        assert isinstance(history._clients_dict, dict)

    def test_post_init_rounds_history_creation(self):
        """Test that __post_init__ creates RoundsInfo correctly"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(
            aggregation_strategy_keyword="pid", num_of_rounds=4, num_of_clients=6
        )

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Verify RoundsInfo was created with correct config
        assert isinstance(history.rounds_history, RoundsInfo)
        assert history.rounds_history is not None
        assert history.rounds_history.simulation_strategy_config == config

    def test_post_init_clients_dict_creation(self):
        """Test that __post_init__ creates client dictionary correctly"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {0, 2}

        config = StrategyConfig(num_of_rounds=3, num_of_clients=4)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Verify correct number of clients created
        assert len(history._clients_dict) == 4

        # Verify client IDs are correct
        for i in range(4):
            assert i in history._clients_dict
            client = history._clients_dict[i]
            assert isinstance(client, ClientInfo)
            assert client.client_id == i
            assert client.num_of_rounds == 3

    def test_post_init_malicious_client_marking(self):
        """Test that malicious clients are correctly marked"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {1, 3}

        config = StrategyConfig(num_of_rounds=2, num_of_clients=5)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Check malicious client marking
        assert history._clients_dict[0].is_malicious is False
        assert history._clients_dict[1].is_malicious is True
        assert history._clients_dict[2].is_malicious is False
        assert history._clients_dict[3].is_malicious is True
        assert history._clients_dict[4].is_malicious is False

    def test_get_all_clients(self):
        """Test get_all_clients method returns correct list"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {1}

        config = StrategyConfig(num_of_rounds=2, num_of_clients=3)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        all_clients = history.get_all_clients()

        assert len(all_clients) == 3
        assert all(isinstance(client, ClientInfo) for client in all_clients)

        # Verify client IDs are present (order may vary)
        client_ids = {client.client_id for client in all_clients}
        assert client_ids == {0, 1, 2}

    def test_insert_single_client_history_entry_basic(self):
        """Test insert_single_client_history_entry with basic parameters"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=3, num_of_clients=2)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Insert history entry for client 0, round 1
        history.insert_single_client_history_entry(
            client_id=0,
            current_round=1,
            removal_criterion=0.5,
            absolute_distance=0.3,
            loss=0.2,
            accuracy=0.85,
            aggregation_participation=1,
        )

        client = history._clients_dict[0]
        assert client.removal_criterion_history[0] == 0.5
        assert client.absolute_distance_history[0] == 0.3
        assert client.loss_history[0] == 0.2
        assert client.accuracy_history[0] == 0.85
        assert client.aggregation_participation_history[0] == 1

    def test_insert_single_client_history_entry_partial_data(self):
        """Test insert_single_client_history_entry with partial data"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=2, num_of_clients=2)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Insert only some metrics
        history.insert_single_client_history_entry(
            client_id=1, current_round=2, loss=0.4, accuracy=0.75
        )

        client = history._clients_dict[1]
        # Only specified metrics should be updated
        assert client.loss_history[1] == 0.4
        assert client.accuracy_history[1] == 0.75
        # Others should remain as initialized (None or default)
        assert client.removal_criterion_history[1] is None
        assert client.absolute_distance_history[1] is None
        # aggregation_participation should remain default (1)
        assert client.aggregation_participation_history[1] == 1

    def test_insert_round_history_entry_basic(self):
        """Test insert_round_history_entry with all parameters"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=2, num_of_clients=2)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        history.insert_round_history_entry(
            score_calculation_time_nanos=1500000,
            removal_threshold=0.6,
            loss_aggregated=0.25,
        )

        rounds_info = history.rounds_history
        assert rounds_info is not None
        assert rounds_info.score_calculation_time_nanos_history == [1500000]
        assert rounds_info.removal_threshold_history == [0.6]
        assert rounds_info.aggregated_loss_history == [0.25]

    def test_insert_round_history_entry_partial_data(self):
        """Test insert_round_history_entry with partial data"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=2, num_of_clients=2)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Insert only some metrics
        history.insert_round_history_entry(
            score_calculation_time_nanos=2000000, loss_aggregated=0.35
        )

        rounds_info = history.rounds_history
        assert rounds_info is not None
        assert rounds_info.score_calculation_time_nanos_history == [2000000]
        assert rounds_info.aggregated_loss_history == [0.35]
        # removal_threshold should not be updated
        assert len(rounds_info.removal_threshold_history) == 0

    def test_update_client_participation(self):
        """Test update_client_participation method"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=3, num_of_clients=5)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Remove clients 1 and 3 in round 2
        removed_client_ids = {1, 3}
        history.update_client_participation(
            current_round=2, removed_client_ids=removed_client_ids
        )

        # Check that removed clients have participation = 0 for round 2 (index 1)
        assert history._clients_dict[1].aggregation_participation_history[1] == 0
        assert history._clients_dict[3].aggregation_participation_history[1] == 0

        # Check that non-removed clients still have participation = 1
        assert history._clients_dict[0].aggregation_participation_history[1] == 1
        assert history._clients_dict[2].aggregation_participation_history[1] == 1
        assert history._clients_dict[4].aggregation_participation_history[1] == 1

    def test_calculate_additional_rounds_data_basic_scenario(self):
        """Test calculate_additional_rounds_data with basic scenario"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {
            1,
            3,
        }  # Clients 1 and 3 are malicious

        config = StrategyConfig(num_of_rounds=2, num_of_clients=4, remove_clients=True)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Set up client data for round 1
        # Client 0 (benign): aggregated, accuracy 0.8
        history.insert_single_client_history_entry(
            0, 1, accuracy=0.8, aggregation_participation=1
        )
        # Client 1 (malicious): not aggregated
        history.insert_single_client_history_entry(
            1, 1, accuracy=0.6, aggregation_participation=0
        )
        # Client 2 (benign): aggregated, accuracy 0.9
        history.insert_single_client_history_entry(
            2, 1, accuracy=0.9, aggregation_participation=1
        )
        # Client 3 (malicious): aggregated (false negative)
        history.insert_single_client_history_entry(
            3, 1, accuracy=0.7, aggregation_participation=1
        )

        # Set up client data for round 2
        history.insert_single_client_history_entry(
            0, 2, accuracy=0.85, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            1, 2, accuracy=0.65, aggregation_participation=0
        )
        history.insert_single_client_history_entry(
            2, 2, accuracy=0.95, aggregation_participation=0
        )  # False positive
        history.insert_single_client_history_entry(
            3, 2, accuracy=0.75, aggregation_participation=0
        )

        history.calculate_additional_rounds_data()

        rounds_info = history.rounds_history
        assert rounds_info is not None

        # Round 1: TP=2 (clients 0,2), TN=1 (client 1), FP=0, FN=1 (client 3)
        # Round 2: TP=1 (client 0), TN=2 (clients 1,3), FP=1 (client 2), FN=0
        assert rounds_info.tp_history == [2, 1]
        assert rounds_info.tn_history == [1, 2]
        assert rounds_info.fp_history == [0, 1]
        assert rounds_info.fn_history == [1, 0]

        # Average accuracy should be calculated for benign aggregated clients
        # Round 1: (0.8 + 0.9) / 2 = 0.85
        # Round 2: 0.85 / 1 = 0.85
        assert rounds_info.average_accuracy_history == pytest.approx(
            [0.85, 0.85], rel=1e-3
        )

    def test_calculate_additional_rounds_data_no_removal(self):
        """Test calculate_additional_rounds_data when remove_clients=False"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {1}

        config = StrategyConfig(num_of_rounds=2, num_of_clients=3, remove_clients=False)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Set up client data (all clients participate when no removal)
        history.insert_single_client_history_entry(
            0, 1, accuracy=0.8, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            1, 1, accuracy=0.6, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            2, 1, accuracy=0.9, aggregation_participation=1
        )

        history.insert_single_client_history_entry(
            0, 2, accuracy=0.85, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            1, 2, accuracy=0.65, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            2, 2, accuracy=0.95, aggregation_participation=1
        )

        history.calculate_additional_rounds_data()

        rounds_info = history.rounds_history
        assert rounds_info is not None

        # When remove_clients=False, TP/TN/FP/FN should still be calculated but all zeros
        # since no removal logic is applied
        assert len(rounds_info.tp_history) == 2
        assert len(rounds_info.tn_history) == 2
        assert len(rounds_info.fp_history) == 2
        assert len(rounds_info.fn_history) == 2

        # Average accuracy should include only benign clients (0 and 2)
        # Round 1: (0.8 + 0.9) / 2 = 0.85
        # Round 2: (0.85 + 0.95) / 2 = 0.9
        assert rounds_info.average_accuracy_history == pytest.approx(
            [0.85, 0.9], rel=1e-3
        )

    def test_calculate_additional_rounds_data_calls_additional_metrics(self):
        """Test that calculate_additional_rounds_data calls calculate_additional_metrics when remove_clients=True"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=1, num_of_clients=2, remove_clients=True)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Set up minimal client data
        history.insert_single_client_history_entry(
            0, 1, accuracy=0.8, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            1, 1, accuracy=0.9, aggregation_participation=1
        )

        # Mock the calculate_additional_metrics method to verify it's called
        assert history.rounds_history is not None
        original_method = history.rounds_history.calculate_additional_metrics
        history.rounds_history.calculate_additional_metrics = Mock()

        history.calculate_additional_rounds_data()

        # Verify calculate_additional_metrics was called
        history.rounds_history.calculate_additional_metrics.assert_called_once()

        # Restore original method
        history.rounds_history.calculate_additional_metrics = original_method

    def test_data_consistency_across_operations(self):
        """Test data consistency across multiple operations"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {2}

        config = StrategyConfig(
            num_of_rounds=1,  # Use only 1 round to avoid None values
            num_of_clients=3,
            remove_clients=True,
        )

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Insert client history entries for round 1
        history.insert_single_client_history_entry(
            0, 1, loss=0.3, accuracy=0.8, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            1, 1, loss=0.25, accuracy=0.85, aggregation_participation=1
        )
        history.insert_single_client_history_entry(
            2, 1, loss=0.4, accuracy=0.7, aggregation_participation=0
        )

        # Insert round history entry
        history.insert_round_history_entry(
            score_calculation_time_nanos=1000000,
            removal_threshold=0.5,
            loss_aggregated=0.275,
        )

        # Update client participation
        history.update_client_participation(1, {2})

        # Calculate additional data
        history.calculate_additional_rounds_data()

        # Verify data consistency
        assert len(history._clients_dict) == 3
        assert history._clients_dict[0].loss_history[0] == 0.3
        assert history._clients_dict[1].accuracy_history[0] == 0.85
        assert history._clients_dict[2].aggregation_participation_history[0] == 0

        assert history.rounds_history is not None
        assert history.rounds_history.score_calculation_time_nanos_history == [1000000]
        assert history.rounds_history.removal_threshold_history == [0.5]
        assert history.rounds_history.aggregated_loss_history == [0.275]

        # Average accuracy should be (0.8 + 0.85) / 2 = 0.825 for benign aggregated clients
        assert history.rounds_history.average_accuracy_history[0] == pytest.approx(
            0.825, rel=1e-3
        )

    def test_edge_case_no_clients(self):
        """Test edge case with zero clients"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=1, num_of_clients=0)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        assert len(history._clients_dict) == 0
        assert len(history.get_all_clients()) == 0

    def test_edge_case_all_clients_malicious(self):
        """Test edge case where all clients are malicious"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = {0, 1, 2}

        config = StrategyConfig(num_of_rounds=1, num_of_clients=3)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # All clients should be marked as malicious
        for client in history.get_all_clients():
            assert client.is_malicious is True

    def test_edge_case_single_round_single_client(self):
        """Test edge case with single round and single client"""
        mock_dataset_handler = Mock(spec=DatasetHandler)
        mock_dataset_handler.poisoned_client_ids = set()

        config = StrategyConfig(num_of_rounds=1, num_of_clients=1, remove_clients=True)

        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=mock_dataset_handler,
            rounds_history=None,
        )

        # Insert data and calculate
        history.insert_single_client_history_entry(
            0, 1, accuracy=0.9, aggregation_participation=1
        )
        history.calculate_additional_rounds_data()

        # Should work without errors
        assert len(history._clients_dict) == 1
        assert history.rounds_history is not None
        assert history.rounds_history.average_accuracy_history[0] == 0.9
