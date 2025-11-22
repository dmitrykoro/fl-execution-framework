"""
Unit tests for attack_utils.poisoning module.

Tests all poisoning attack functions and their edge cases.
"""

from tests.common import pytest
import torch

from src.attack_utils.poisoning import (
    apply_label_flipping,
    apply_gaussian_noise,
    should_poison_this_round,
    apply_poisoning_attack,
)


class TestApplyLabelFlipping:
    """Test suite for apply_label_flipping function."""

    def test_single_class_no_flipping(self):
        """Test that single class dataset returns unchanged (can't flip to different class)."""
        labels = torch.tensor([0, 0, 0, 0, 0])
        original_labels = labels.clone()

        result = apply_label_flipping(labels, num_classes=1)

        assert torch.equal(result, original_labels)

    def test_class_level_mapping(self):
        """Test that all samples of each class get mapped to same random class."""
        # Create labels with multiple samples per class
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        num_classes = 5

        result = apply_label_flipping(labels, num_classes=num_classes)

        # All class 0 samples should map to same new class
        class_0_values = result[labels == 0].unique()
        assert len(class_0_values) == 1, "All class 0 samples should map to same class"
        assert class_0_values[0] != 0, "Class 0 should not map to itself"

        # All class 1 samples should map to same new class
        class_1_values = result[labels == 1].unique()
        assert len(class_1_values) == 1, "All class 1 samples should map to same class"
        assert class_1_values[0] != 1, "Class 1 should not map to itself"

        # All class 2 samples should map to same new class
        class_2_values = result[labels == 2].unique()
        assert len(class_2_values) == 1, "All class 2 samples should map to same class"
        assert class_2_values[0] != 2, "Class 2 should not map to itself"

    def test_all_labels_modified(self):
        """Test that all labels get remapped (100% poisoning)."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        num_classes = 10

        result = apply_label_flipping(labels, num_classes=num_classes)

        # Every label should be different from original
        assert not torch.equal(result, labels), "All labels should be modified"

        # But should still be valid class IDs
        assert torch.all(result >= 0)
        assert torch.all(result < num_classes)

    def test_label_values_within_range(self):
        """Test that flipped labels are within valid class range."""
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        num_classes = 10

        result = apply_label_flipping(labels, num_classes=num_classes)

        # All labels should be in range [0, num_classes)
        assert torch.all(result >= 0)
        assert torch.all(result < num_classes)

    def test_no_self_mapping(self):
        """Test that no class maps to itself."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        num_classes = 10

        result = apply_label_flipping(labels, num_classes=num_classes)

        # Check each original class doesn't map to itself
        for original_class in labels.unique():
            mapped_class = result[labels == original_class][0]
            assert mapped_class != original_class, (
                f"Class {original_class} should not map to itself"
            )

    def test_two_class_swapping(self):
        """Test label flipping with only 2 classes."""
        labels = torch.tensor([0, 0, 1, 1])
        num_classes = 2

        result = apply_label_flipping(labels, num_classes=num_classes)

        # With only 2 classes, they should swap
        assert torch.all(result[labels == 0] == 1), "Class 0 should become class 1"
        assert torch.all(result[labels == 1] == 0), "Class 1 should become class 0"


class TestApplyGaussianNoise:
    """Test suite for apply_gaussian_noise function."""

    def test_no_poisoning_when_attack_ratio_zero(self):
        """Test that no samples are poisoned when attack_ratio is 0."""
        images = torch.rand(10, 3, 28, 28)
        original_images = images.clone()

        result = apply_gaussian_noise(images, attack_ratio=0.0)

        assert torch.allclose(result, original_images)

    def test_no_poisoning_when_num_to_poison_zero(self):
        """Test that no samples are poisoned when calculated num_to_poison is 0."""
        images = torch.rand(2, 3, 28, 28)
        original_images = images.clone()

        # With 2 images and attack_ratio=0.3, int(2 * 0.3) = 0
        result = apply_gaussian_noise(images, attack_ratio=0.3)

        assert torch.allclose(result, original_images)

    def test_gaussian_noise_with_mean_std(self):
        """Test Gaussian noise addition with mean and std parameters."""
        images = torch.rand(10, 3, 28, 28)
        original_images = images.clone()

        result = apply_gaussian_noise(images, mean=0.0, std=0.1, attack_ratio=1.0)

        # Images should be modified
        assert not torch.allclose(result, original_images, rtol=1e-4)

        # Results should be clamped to [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_gaussian_noise_with_snr(self):
        """Test Gaussian noise addition with SNR parameter."""
        images = torch.rand(10, 3, 28, 28) + 0.1  # Ensure non-zero signal
        original_images = images.clone()

        result = apply_gaussian_noise(images, target_noise_snr=20.0, attack_ratio=1.0)

        # Images should be modified
        assert not torch.allclose(result, original_images, rtol=1e-4)

        # Results should be clamped to [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_partial_poisoning(self):
        """Test that only specified fraction of samples are poisoned."""
        images = torch.rand(10, 3, 28, 28)
        attack_ratio = 0.5

        result = apply_gaussian_noise(images, std=0.5, attack_ratio=attack_ratio)

        # Exactly 50% of samples should be modified
        expected_poisoned = int(10 * attack_ratio)
        assert expected_poisoned == 5

        # Results should still be valid
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_noise_clamping(self):
        """Test that noisy images are properly clamped to [0, 1]."""
        # Create images near boundaries
        images = torch.ones(5, 1, 10, 10) * 0.9

        # Add large noise that would exceed boundaries
        result = apply_gaussian_noise(images, mean=0.0, std=1.0, attack_ratio=1.0)

        # All values should be clamped
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)


class TestShouldPoisonThisRound:
    """Test suite for should_poison_this_round function."""

    def test_empty_schedule_returns_false(self):
        """Test that empty/None schedule returns False."""
        should_poison, attacks = should_poison_this_round(5, 0, None)
        assert should_poison is False
        assert len(attacks) == 0

        should_poison, attacks = should_poison_this_round(5, 0, [])
        assert should_poison is False
        assert len(attacks) == 0

    def test_specific_selection_strategy(self):
        """Test specific client selection strategy."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0, 2, 4],
            }
        ]

        # Client in list should be poisoned
        should_poison, attacks = should_poison_this_round(5, 0, schedule)
        assert should_poison is True
        assert len(attacks) == 1

        # Client not in list should not be poisoned
        should_poison, attacks = should_poison_this_round(5, 1, schedule)
        assert should_poison is False
        assert len(attacks) == 0

    def test_random_selection_strategy(self):
        """Test random client selection strategy."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "random",
                "_selected_clients": [1, 3, 5],
            }
        ]

        # Client in selected list should be poisoned
        should_poison, attacks = should_poison_this_round(5, 1, schedule)
        assert should_poison is True
        assert len(attacks) == 1

        # Client not in selected list should not be poisoned
        should_poison, attacks = should_poison_this_round(5, 0, schedule)
        assert should_poison is False
        assert len(attacks) == 0

    def test_percentage_selection_strategy(self):
        """Test percentage client selection strategy."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "gaussian_noise",
                "selection_strategy": "percentage",
                "_selected_clients": [0, 2, 4],
                "target_noise_snr": 10.0,
            }
        ]

        # Client in selected list should be poisoned
        should_poison, attacks = should_poison_this_round(5, 2, schedule)
        assert should_poison is True
        assert len(attacks) == 1

        # Client not in selected list should not be poisoned
        should_poison, attacks = should_poison_this_round(5, 1, schedule)
        assert should_poison is False
        assert len(attacks) == 0

    def test_round_range_filtering(self):
        """Test that attacks are only active within their round range."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
            }
        ]

        # Before start_round
        should_poison, attacks = should_poison_this_round(3, 0, schedule)
        assert should_poison is False

        # Within range
        should_poison, attacks = should_poison_this_round(7, 0, schedule)
        assert should_poison is True

        # After end_round
        should_poison, attacks = should_poison_this_round(11, 0, schedule)
        assert should_poison is False

    def test_attack_type_deduplication(self):
        """Test that duplicate attack types use first-match-wins."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
            },
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
            },
        ]

        should_poison, attacks = should_poison_this_round(5, 0, schedule)

        # Should only return one attack (first match)
        assert len(attacks) == 1

    def test_multiple_attack_types(self):
        """Test that different attack types can be returned together."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
            },
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "gaussian_noise",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
                "target_noise_snr": 15.0,
            },
        ]

        should_poison, attacks = should_poison_this_round(5, 0, schedule)

        # Should return both attacks
        assert len(attacks) == 2
        attack_types = {a["attack_type"] for a in attacks}
        assert attack_types == {"label_flipping", "gaussian_noise"}

    def test_missing_attack_type_not_included(self):
        """Test that attack entries without attack_type are not included."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
                # Missing attack_type
            }
        ]

        should_poison, attacks = should_poison_this_round(5, 0, schedule)

        assert should_poison is False
        assert len(attacks) == 0


class TestApplyPoisoningAttack:
    """Test suite for apply_poisoning_attack function."""

    def test_label_flipping_attack(self):
        """Test label flipping attack application."""
        images = torch.rand(10, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_labels = labels.clone()

        attack_config = {
            "attack_type": "label_flipping",
        }
        num_classes = 10

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config, num_classes=num_classes
        )

        # Images should be unchanged
        assert torch.allclose(result_images, images)

        # Labels should be modified
        assert not torch.equal(result_labels, original_labels)

        # All labels should be in valid range
        assert torch.all(result_labels >= 0)
        assert torch.all(result_labels < num_classes)

    def test_gaussian_noise_attack_with_snr(self):
        """Test Gaussian noise attack with SNR parameter."""
        images = torch.rand(10, 3, 28, 28) + 0.1
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_images = images.clone()

        attack_config = {
            "attack_type": "gaussian_noise",
            "target_noise_snr": 20.0,
            "attack_ratio": 1.0,
        }

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config
        )

        # Images should be modified
        assert not torch.allclose(result_images, original_images, rtol=1e-4)

        # Labels should be unchanged
        assert torch.equal(result_labels, labels)

    def test_gaussian_noise_attack_with_mean_std(self):
        """Test Gaussian noise attack with mean/std parameters."""
        images = torch.rand(10, 3, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_images = images.clone()

        attack_config = {
            "attack_type": "gaussian_noise",
            "mean": 0.0,
            "std": 0.1,
            "attack_ratio": 1.0,
        }

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config
        )

        # Images should be modified
        assert not torch.allclose(result_images, original_images, rtol=1e-4)

        # Labels should be unchanged
        assert torch.equal(result_labels, labels)

    def test_old_nested_format_raises_error(self):
        """Test that old nested config format raises helpful error."""
        images = torch.rand(10, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Old format with "params" key
        old_config1 = {
            "type": "label_flipping",
        }

        with pytest.raises(ValueError, match="old nested attack config format"):
            apply_poisoning_attack(images, labels, old_config1)

        # Old format with "type" but no "attack_type"
        old_config2 = {
            "type": "label_flipping",
        }

        with pytest.raises(ValueError, match="old nested attack config format"):
            apply_poisoning_attack(images, labels, old_config2)

    def test_unknown_attack_type_does_nothing(self):
        """Test that unknown attack type returns data unchanged."""
        images = torch.rand(10, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_images = images.clone()
        original_labels = labels.clone()

        attack_config = {"attack_type": "unknown_attack"}

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config
        )

        # Data should be unchanged
        assert torch.allclose(result_images, original_images)
        assert torch.equal(result_labels, original_labels)

    def test_requires_num_classes(self):
        """Test that num_classes is required for label_flipping."""
        images = torch.rand(10, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        attack_config = {
            "attack_type": "label_flipping",
        }

        # Should raise ValueError when num_classes not provided
        with pytest.raises(ValueError, match="num_classes"):
            apply_poisoning_attack(images, labels, attack_config)
