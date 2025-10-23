"""Unit tests for dynamic client.is_malicious updates based on attack_schedule."""

from unittest.mock import Mock
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.data_models.simulation_strategy_config import StrategyConfig


class TestDynamicClientMaliciousUpdate:
    """Test suite for dynamic client.is_malicious updates."""

    def test_update_client_malicious_status_with_attack_schedule(self):
        """Test that client.is_malicious is updated correctly based on attack_schedule."""
        # Create a minimal strategy config with attack_schedule
        strategy_config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "fedavg",
                "num_of_rounds": 10,
                "num_of_clients": 5,
                "dataset_keyword": "femnist_iid",
                "attack_schedule": [
                    {
                        "start_round": 3,
                        "end_round": 8,
                        "attack_type": "label_flipping",
                        "flip_fraction": 0.7,
                        "selection_strategy": "specific",
                        "malicious_client_ids": [0, 1, 2],
                    }
                ],
            }
        )

        # Create mock dataset handler (no filesystem poisoning needed)
        dataset_handler = Mock()
        dataset_handler.poisoned_client_ids = []  # No static poisoning

        # Create strategy history
        strategy_history = SimulationStrategyHistory(
            strategy_config=strategy_config, dataset_handler=dataset_handler
        )

        # Initially, all clients should be non-malicious (no static attack)
        for client_id in range(5):
            assert strategy_history._clients_dict[client_id].is_malicious is False

        # Round 2: Before attack starts - all should remain non-malicious
        strategy_history.update_client_malicious_status(current_round=2)
        assert strategy_history._clients_dict[0].is_malicious is False
        assert strategy_history._clients_dict[1].is_malicious is False
        assert strategy_history._clients_dict[2].is_malicious is False
        assert strategy_history._clients_dict[3].is_malicious is False
        assert strategy_history._clients_dict[4].is_malicious is False

        # Round 3: Attack starts - clients 0,1,2 should be malicious
        strategy_history.update_client_malicious_status(current_round=3)
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is True
        assert strategy_history._clients_dict[3].is_malicious is False
        assert strategy_history._clients_dict[4].is_malicious is False

        # Round 5: During attack - same pattern
        strategy_history.update_client_malicious_status(current_round=5)
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is True
        assert strategy_history._clients_dict[3].is_malicious is False
        assert strategy_history._clients_dict[4].is_malicious is False

        # Round 8: Last round of attack - same pattern
        strategy_history.update_client_malicious_status(current_round=8)
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is True
        assert strategy_history._clients_dict[3].is_malicious is False
        assert strategy_history._clients_dict[4].is_malicious is False

        # Round 9: After attack ends - all should be non-malicious again
        strategy_history.update_client_malicious_status(current_round=9)
        assert strategy_history._clients_dict[0].is_malicious is False
        assert strategy_history._clients_dict[1].is_malicious is False
        assert strategy_history._clients_dict[2].is_malicious is False
        assert strategy_history._clients_dict[3].is_malicious is False
        assert strategy_history._clients_dict[4].is_malicious is False

    def test_update_client_malicious_status_without_attack_schedule(self):
        """Test that update is skipped when no attack_schedule is present."""
        # Create config without attack_schedule
        strategy_config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "fedavg",
                "num_of_rounds": 10,
                "num_of_clients": 5,
                "dataset_keyword": "femnist_iid",
            }
        )

        dataset_handler = Mock()
        dataset_handler.poisoned_client_ids = [0, 1]  # Static poisoning

        strategy_history = SimulationStrategyHistory(
            strategy_config=strategy_config, dataset_handler=dataset_handler
        )

        # Clients 0,1 should be malicious (static)
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is False

        # Update should not change anything (no attack_schedule)
        strategy_history.update_client_malicious_status(current_round=5)

        # Should remain the same
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is False

    def test_update_client_malicious_status_with_stacked_attacks(self):
        """Test that malicious status is correct with overlapping attacks."""
        strategy_config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "fedavg",
                "num_of_rounds": 12,
                "num_of_clients": 5,
                "dataset_keyword": "femnist_iid",
                "attack_schedule": [
                    {
                        "start_round": 3,
                        "end_round": 8,
                        "attack_type": "label_flipping",
                        "flip_fraction": 0.7,
                        "selection_strategy": "specific",
                        "malicious_client_ids": [0, 1, 2],
                    },
                    {
                        "start_round": 5,
                        "end_round": 10,
                        "attack_type": "gaussian_noise",
                        "target_noise_snr": 10.0,
                        "selection_strategy": "specific",
                        "malicious_client_ids": [0, 1],
                    },
                ],
            }
        )

        dataset_handler = Mock()
        dataset_handler.poisoned_client_ids = []

        strategy_history = SimulationStrategyHistory(
            strategy_config=strategy_config, dataset_handler=dataset_handler
        )

        # Round 4: Only label_flipping active
        strategy_history.update_client_malicious_status(current_round=4)
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is True
        assert strategy_history._clients_dict[3].is_malicious is False

        # Round 6: Both attacks overlap - clients 0,1 should still be malicious
        # Client 2 should also be malicious (only label_flipping)
        strategy_history.update_client_malicious_status(current_round=6)
        assert strategy_history._clients_dict[0].is_malicious is True  # Both attacks
        assert strategy_history._clients_dict[1].is_malicious is True  # Both attacks
        assert (
            strategy_history._clients_dict[2].is_malicious is True
        )  # Only label_flipping
        assert strategy_history._clients_dict[3].is_malicious is False

        # Round 9: Only gaussian_noise active
        strategy_history.update_client_malicious_status(current_round=9)
        assert strategy_history._clients_dict[0].is_malicious is True
        assert strategy_history._clients_dict[1].is_malicious is True
        assert strategy_history._clients_dict[2].is_malicious is False
        assert strategy_history._clients_dict[3].is_malicious is False

        # Round 11: No attacks
        strategy_history.update_client_malicious_status(current_round=11)
        assert strategy_history._clients_dict[0].is_malicious is False
        assert strategy_history._clients_dict[1].is_malicious is False
        assert strategy_history._clients_dict[2].is_malicious is False
        assert strategy_history._clients_dict[3].is_malicious is False
