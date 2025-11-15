"""Unit tests for attack stacking functionality in dynamic poisoning."""

from tests.common import pytest
import torch

from src.attack_utils.poisoning import (
    should_poison_this_round,
    apply_poisoning_attack,
)


def _create_attack_schedule_entry(
    start_round: int, end_round: int, attack_type: str, client_ids: list, **kwargs
) -> dict:
    """
    Create a single attack schedule entry (DRY helper).

    Args:
        start_round: Starting round for attack
        end_round: Ending round for attack
        attack_type: Type of attack ("label_flipping" or "gaussian_noise")
        client_ids: List of client IDs to target
        **kwargs: Additional attack-specific parameters

    Returns:
        Attack schedule entry dictionary
    """
    entry = {
        "start_round": start_round,
        "end_round": end_round,
        "attack_type": attack_type,
        "selection_strategy": "specific",
        "malicious_client_ids": client_ids,
    }
    entry.update(kwargs)
    return entry


def _create_overlapping_schedule() -> list:
    """
    Create attack schedule with overlapping rounds (DRY helper).

    Returns:
        List of attack schedule entries with overlaps
    """
    return [
        _create_attack_schedule_entry(
            start_round=3,
            end_round=8,
            attack_type="label_flipping",
            client_ids=[0, 1, 2],
            flip_fraction=0.7,
            target_class=7,
        ),
        _create_attack_schedule_entry(
            start_round=5,
            end_round=10,
            attack_type="gaussian_noise",
            client_ids=[0, 1],
            target_noise_snr=10.0,
            attack_ratio=0.5,
        ),
    ]


def _create_sample_batch(batch_size: int = 10, num_classes: int = 10) -> tuple:
    """
    Create sample batch of images and labels (DRY helper).

    Args:
        batch_size: Number of samples in batch
        num_classes: Number of classes for labels

    Returns:
        Tuple of (images, labels) tensors
    """
    images = torch.rand(batch_size, 1, 28, 28)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


class TestPoisoningStacking:
    """Test suite for attack stacking functionality."""

    @pytest.mark.parametrize(
        "round_num,client_id,expected_count,expected_types",
        [
            # Round 5, client 0: Both attacks overlap
            (5, 0, 2, ["label_flipping", "gaussian_noise"]),
            # Round 5, client 1: Both attacks overlap
            (5, 1, 2, ["label_flipping", "gaussian_noise"]),
            # Round 5, client 2: Only label_flipping
            (5, 2, 1, ["label_flipping"]),
            # Round 3, client 0: Only label_flipping (before gaussian_noise starts)
            (3, 0, 1, ["label_flipping"]),
            # Round 9, client 0: Only gaussian_noise (after label_flipping ends)
            (9, 0, 1, ["gaussian_noise"]),
            # Round 2, client 0: No attacks (before any start)
            (2, 0, 0, []),
            # Round 11, client 0: No attacks (after all end)
            (11, 0, 0, []),
        ],
    )
    def test_overlapping_attack_detection(
        self, round_num, client_id, expected_count, expected_types
    ):
        """Test that overlapping attacks are correctly detected and returned."""
        schedule = _create_overlapping_schedule()

        should_poison, attack_configs = should_poison_this_round(
            round_num, client_id, schedule
        )

        # Verify poisoning status
        assert should_poison == (expected_count > 0), (
            f"Round {round_num}, Client {client_id}: "
            f"Expected should_poison={expected_count > 0}, got {should_poison}"
        )

        # Verify number of attacks
        assert len(attack_configs) == expected_count, (
            f"Round {round_num}, Client {client_id}: "
            f"Expected {expected_count} attacks, got {len(attack_configs)}"
        )

        # Verify attack types
        actual_types = sorted([cfg["attack_type"] for cfg in attack_configs])
        expected_types_sorted = sorted(expected_types)
        assert actual_types == expected_types_sorted, (
            f"Round {round_num}, Client {client_id}: "
            f"Expected types {expected_types_sorted}, got {actual_types}"
        )

    def test_attack_type_deduplication(self):
        """Test that duplicate attack types use first-match-wins strategy."""
        # Schedule with two label_flipping attacks for same client
        schedule = [
            _create_attack_schedule_entry(
                start_round=1,
                end_round=5,
                attack_type="label_flipping",
                client_ids=[0],
                flip_fraction=0.5,
                target_class=5,
            ),
            _create_attack_schedule_entry(
                start_round=3,
                end_round=7,
                attack_type="label_flipping",
                client_ids=[0],
                flip_fraction=0.8,  # Different parameter
                target_class=5,
            ),
        ]

        should_poison, attack_configs = should_poison_this_round(4, 0, schedule)

        # Should only return one label_flipping attack (first match)
        assert len(attack_configs) == 1, (
            f"Expected 1 attack (deduplication), got {len(attack_configs)}"
        )
        assert attack_configs[0]["flip_fraction"] == 0.5, (
            "Should use first match (flip_fraction=0.5)"
        )

    @pytest.mark.parametrize("batch_size", [5, 10, 20])
    def test_sequential_attack_application(self, batch_size):
        """Test that stacked attacks are applied sequentially."""
        images, labels = _create_sample_batch(batch_size=batch_size)
        schedule = _create_overlapping_schedule()

        # Store original values
        original_images = images.clone()
        original_labels = labels.clone()

        # Get attacks for round 5, client 0 (both attacks)
        should_poison, attack_configs = should_poison_this_round(5, 0, schedule)
        assert should_poison and len(attack_configs) == 2

        # Apply attacks sequentially
        for attack_config in attack_configs:
            images, labels = apply_poisoning_attack(images, labels, attack_config)

        # Verify both attacks were applied
        # Labels should be modified by label_flipping
        assert not torch.equal(labels, original_labels), (
            "Labels should be modified by label_flipping"
        )

        # Images should be modified by gaussian_noise
        assert not torch.allclose(images, original_images, rtol=1e-4), (
            "Images should be modified by gaussian_noise"
        )

    def test_empty_schedule_returns_no_attacks(self):
        """Test that empty schedule returns no attacks."""
        should_poison, attack_configs = should_poison_this_round(5, 0, None)
        assert should_poison is False
        assert len(attack_configs) == 0

        should_poison, attack_configs = should_poison_this_round(5, 0, [])
        assert should_poison is False
        assert len(attack_configs) == 0

    def test_client_not_in_schedule(self):
        """Test that clients not in schedule receive no attacks."""
        schedule = _create_overlapping_schedule()

        # Client 5 is not in any attack schedule
        should_poison, attack_configs = should_poison_this_round(5, 5, schedule)

        assert should_poison is False
        assert len(attack_configs) == 0

    @pytest.mark.parametrize(
        "selection_strategy,config_key,config_value",
        [
            ("random", "_selected_clients", [0, 1]),
            ("percentage", "_selected_clients", [0, 2]),
        ],
    )
    def test_selection_strategies(self, selection_strategy, config_key, config_value):
        """Test different client selection strategies."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": selection_strategy,
                "flip_fraction": 0.5,
                "target_class": 7,
                config_key: config_value,  # Pre-selected clients
            }
        ]

        # Test client in selected list
        should_poison, attack_configs = should_poison_this_round(5, 0, schedule)
        assert should_poison is True
        assert len(attack_configs) == 1

        # Test client not in selected list
        should_poison, attack_configs = should_poison_this_round(5, 3, schedule)
        assert should_poison is False
        assert len(attack_configs) == 0

    def test_attack_parameters_preserved(self):
        """Test that attack parameters are correctly preserved in returned configs."""
        schedule = [
            _create_attack_schedule_entry(
                start_round=1,
                end_round=10,
                attack_type="label_flipping",
                client_ids=[0],
                flip_fraction=0.75,
                target_class=5,
            ),
            _create_attack_schedule_entry(
                start_round=1,
                end_round=10,
                attack_type="gaussian_noise",
                client_ids=[0],
                target_noise_snr=15.0,
                attack_ratio=0.8,
            ),
        ]

        should_poison, attack_configs = should_poison_this_round(5, 0, schedule)

        # Find each attack config
        label_flip_cfg = next(
            cfg for cfg in attack_configs if cfg["attack_type"] == "label_flipping"
        )
        gaussian_cfg = next(
            cfg for cfg in attack_configs if cfg["attack_type"] == "gaussian_noise"
        )

        # Verify parameters are preserved
        assert label_flip_cfg["flip_fraction"] == 0.75
        assert label_flip_cfg["target_class"] == 5
        assert gaussian_cfg["target_noise_snr"] == 15.0
        assert gaussian_cfg["attack_ratio"] == 0.8

    def test_stacking_order_consistency(self):
        """Test that attack stacking maintains consistent order."""
        schedule = _create_overlapping_schedule()

        # Get attacks multiple times
        _, configs1 = should_poison_this_round(5, 0, schedule)
        _, configs2 = should_poison_this_round(5, 0, schedule)
        _, configs3 = should_poison_this_round(5, 0, schedule)

        # Extract attack types
        types1 = [cfg["attack_type"] for cfg in configs1]
        types2 = [cfg["attack_type"] for cfg in configs2]
        types3 = [cfg["attack_type"] for cfg in configs3]

        # Order should be consistent across calls
        assert types1 == types2 == types3, (
            "Attack order should be consistent across multiple calls"
        )

    @pytest.mark.parametrize(
        "num_overlaps",
        [2, 3, 4],
    )
    def test_multiple_overlapping_attacks(self, num_overlaps):
        """Test handling of multiple overlapping attacks."""
        # Create schedule with num_overlaps different attack types
        schedule = []
        attack_types = ["label_flipping", "gaussian_noise"]

        for i in range(num_overlaps):
            attack_type = attack_types[i % len(attack_types)]
            # For duplicate types, only first should be returned
            schedule.append(
                _create_attack_schedule_entry(
                    start_round=1,
                    end_round=10,
                    attack_type=attack_type,
                    client_ids=[0],
                    flip_fraction=0.5,
                    target_class=7,
                    target_noise_snr=10.0,
                    attack_ratio=0.5,
                )
            )

        should_poison, attack_configs = should_poison_this_round(5, 0, schedule)

        # Should only return unique attack types
        expected_count = min(num_overlaps, len(attack_types))
        assert len(attack_configs) == expected_count, (
            f"Expected {expected_count} unique attack types, got {len(attack_configs)}"
        )

        # Verify all returned attacks have different types
        attack_type_set = {cfg["attack_type"] for cfg in attack_configs}
        assert len(attack_type_set) == len(attack_configs), (
            "All returned attacks should have unique types"
        )
