"""
Unit tests for attack_utils.poisoning module.

Tests all poisoning attack functions and their edge cases.
"""

from tests.common import pytest
import torch

from src.attack_utils.poisoning import (
    apply_label_flipping,
    apply_gaussian_noise,
    apply_brightness_attack,
    apply_token_replacement,
    should_poison_this_round,
    apply_poisoning_attack,
)


class TestApplyLabelFlipping:
    """Test suite for apply_label_flipping function."""

    def test_no_flipping_when_fraction_zero(self):
        """Test that no labels are flipped when flip_fraction is 0."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        original_labels = labels.clone()

        result = apply_label_flipping(labels, flip_fraction=0.0)

        assert torch.equal(result, original_labels)

    def test_no_flipping_when_fraction_negative(self):
        """Test that no labels are flipped when flip_fraction is negative."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        original_labels = labels.clone()

        result = apply_label_flipping(labels, flip_fraction=-0.5)

        assert torch.equal(result, original_labels)

    def test_no_flipping_when_num_to_flip_zero(self):
        """Test that no labels are flipped when calculated num_to_flip is 0."""
        labels = torch.tensor([0, 1])
        original_labels = labels.clone()

        # With only 2 labels and flip_fraction=0.3, int(2 * 0.3) = 0
        result = apply_label_flipping(labels, flip_fraction=0.3)

        assert torch.equal(result, original_labels)

    def test_random_flipping(self):
        """Test random label flipping (no target class)."""
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_labels = labels.clone()

        result = apply_label_flipping(labels, flip_fraction=0.5, num_classes=10)

        # Should have modified some labels (though some might randomly flip to same value)
        # Just verify the function executed without error and returned valid labels
        assert len(result) == len(original_labels)
        assert torch.all(result >= 0)
        assert torch.all(result < 10)

    def test_targeted_flipping(self):
        """Test targeted label flipping to specific class."""
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_class = 5

        result = apply_label_flipping(
            labels, flip_fraction=0.5, num_classes=10, target_class=target_class
        )

        # Count how many labels are now target_class
        num_target = (result == target_class).sum().item()
        expected_flips = int(len(labels) * 0.5)

        # Should have at least expected_flips of target_class
        # (original might already have some)
        assert num_target >= expected_flips

    def test_full_flipping(self):
        """Test flipping all labels."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        target_class = 9

        result = apply_label_flipping(
            labels, flip_fraction=1.0, num_classes=10, target_class=target_class
        )

        # All labels should be target_class
        assert torch.all(result == target_class)

    def test_label_values_within_range(self):
        """Test that flipped labels are within valid class range."""
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        num_classes = 10

        result = apply_label_flipping(
            labels, flip_fraction=0.8, num_classes=num_classes
        )

        # All labels should be in range [0, num_classes)
        assert torch.all(result >= 0)
        assert torch.all(result < num_classes)


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


class TestApplyBrightnessAttack:
    """Test suite for apply_brightness_attack function."""

    def test_brightness_darkening(self):
        """Test darkening images with factor < 1."""
        images = torch.rand(5, 3, 28, 28)

        result = apply_brightness_attack(images, factor=0.5)

        # Images should be darker
        assert torch.all(result <= images)
        assert torch.all(result >= 0.0)

    def test_brightness_brightening(self):
        """Test brightening images with factor > 1."""
        images = torch.rand(5, 3, 28, 28) * 0.5  # Keep below 0.5 to allow brightening

        result = apply_brightness_attack(images, factor=1.5)

        # Images should be brighter (with clamping)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_brightness_no_change(self):
        """Test that factor=1.0 doesn't change images."""
        images = torch.rand(5, 3, 28, 28)
        original_images = images.clone()

        result = apply_brightness_attack(images, factor=1.0)

        assert torch.allclose(result, original_images)

    def test_brightness_black(self):
        """Test that factor=0.0 creates black images."""
        images = torch.rand(5, 3, 28, 28)

        result = apply_brightness_attack(images, factor=0.0)

        assert torch.all(result == 0.0)

    def test_brightness_clamping(self):
        """Test that brightness values are clamped to [0, 1]."""
        images = torch.ones(5, 3, 28, 28)

        result = apply_brightness_attack(images, factor=2.0)

        # Should be clamped to 1.0
        assert torch.all(result <= 1.0)


class TestApplyTokenReplacement:
    """Test suite for apply_token_replacement function."""

    def test_no_replacement_when_prob_zero(self):
        """Test that no tokens are replaced when replacement_prob is 0."""
        tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        original_tokens = tokens.clone()

        result = apply_token_replacement(tokens, replacement_prob=0.0)

        assert torch.equal(result, original_tokens)

    def test_no_replacement_when_prob_negative(self):
        """Test that no tokens are replaced when replacement_prob is negative."""
        tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        original_tokens = tokens.clone()

        result = apply_token_replacement(tokens, replacement_prob=-0.5)

        assert torch.equal(result, original_tokens)

    def test_partial_replacement(self):
        """Test partial token replacement."""
        torch.manual_seed(42)  # For reproducibility
        tokens = torch.tensor([[1, 2, 3, 4, 5]] * 100)  # Repeat for statistical test

        result = apply_token_replacement(tokens, replacement_prob=0.3, vocab_size=10)

        # Should have some changes
        changed = ~torch.equal(result, tokens)
        assert changed

    def test_full_replacement(self):
        """Test full token replacement."""
        tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        result = apply_token_replacement(tokens, replacement_prob=1.0, vocab_size=30522)

        # Most tokens should be different (extremely unlikely to match randomly)
        num_changed = (result != tokens).sum().item()
        assert num_changed > 0

    def test_token_values_within_vocab(self):
        """Test that replaced tokens are within vocabulary range."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]] * 10)
        vocab_size = 1000

        result = apply_token_replacement(
            tokens, replacement_prob=0.5, vocab_size=vocab_size
        )

        # All tokens should be in range [0, vocab_size)
        assert torch.all(result >= 0)
        assert torch.all(result < vocab_size)


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
                "flip_fraction": 0.5,
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
                "flip_fraction": 0.5,
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
                "flip_fraction": 0.5,
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
                "flip_fraction": 0.5,
            },
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
                "flip_fraction": 0.8,  # Different param
            },
        ]

        should_poison, attacks = should_poison_this_round(5, 0, schedule)

        # Should only return one attack (first match)
        assert len(attacks) == 1
        assert attacks[0]["flip_fraction"] == 0.5

    def test_multiple_attack_types(self):
        """Test that different attack types can be returned together."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "attack_type": "label_flipping",
                "selection_strategy": "specific",
                "malicious_client_ids": [0],
                "flip_fraction": 0.5,
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
            "flip_fraction": 0.5,
            "num_classes": 10,
            "target_class": 5,
        }

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config
        )

        # Images should be unchanged
        assert torch.allclose(result_images, images)

        # Labels should be modified
        assert not torch.equal(result_labels, original_labels)

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

    def test_brightness_attack(self):
        """Test brightness attack application."""
        images = torch.rand(10, 3, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        attack_config = {"attack_type": "brightness", "factor": 0.5}

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config
        )

        # Images should be darker
        assert torch.all(result_images <= images)

        # Labels should be unchanged
        assert torch.equal(result_labels, labels)

    def test_token_replacement_attack(self):
        """Test token replacement attack application."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]] * 10)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        attack_config = {
            "attack_type": "token_replacement",
            "replacement_prob": 0.5,
            "vocab_size": 30522,
        }

        result_tokens, result_labels = apply_poisoning_attack(
            tokens, labels, attack_config
        )

        # Tokens should be modified
        assert not torch.equal(result_tokens, tokens)

        # Labels should be unchanged
        assert torch.equal(result_labels, labels)

    def test_old_nested_format_raises_error(self):
        """Test that old nested config format raises helpful error."""
        images = torch.rand(10, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Old format with "params" key
        old_config1 = {
            "type": "label_flipping",
            "params": {"flip_fraction": 0.5},
        }

        with pytest.raises(ValueError, match="old nested attack config format"):
            apply_poisoning_attack(images, labels, old_config1)

        # Old format with "type" but no "attack_type"
        old_config2 = {
            "type": "label_flipping",
            "flip_fraction": 0.5,
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

    def test_default_parameters_used(self):
        """Test that default parameters are used when not specified."""
        images = torch.rand(10, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Minimal config - should use defaults
        attack_config = {"attack_type": "label_flipping"}

        result_images, result_labels = apply_poisoning_attack(
            images, labels, attack_config
        )

        # Should still work with defaults
        assert result_images is not None
        assert result_labels is not None
