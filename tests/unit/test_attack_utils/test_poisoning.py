"""
Tests for data poisoning attack utilities.
"""

import torch

from src.attack_utils.poisoning import (
    apply_brightness_attack,
    apply_gaussian_noise,
    apply_label_flipping,
    apply_poisoning_attack,
    apply_token_replacement,
    should_poison_this_round,
)


class TestLabelFlipping:
    """Test label flipping attacks."""

    def test_no_flipping_when_fraction_zero(self):
        """Labels should remain unchanged when flip_fraction is 0."""
        labels = torch.tensor([0, 1, 2, 3, 4])
        result = apply_label_flipping(labels, flip_fraction=0.0)
        assert torch.equal(result, labels)

    def test_targeted_flipping(self):
        """All flipped labels should match target_class."""
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = apply_label_flipping(
            labels, flip_fraction=0.5, num_classes=10, target_class=7
        )
        # Count how many labels are 7
        num_sevens = (result == 7).sum().item()
        # Should have flipped 5 labels to 7 (50% of 10)
        assert num_sevens >= 5

    def test_random_flipping(self):
        """Random flipping should change labels without target."""
        torch.manual_seed(42)
        labels = torch.zeros(100, dtype=torch.long)
        original_labels = labels.clone()
        result = apply_label_flipping(labels, flip_fraction=1.0, num_classes=10)
        # Not all should remain 0
        assert not torch.equal(result, original_labels)
        # All values should be valid class indices
        assert result.min() >= 0
        assert result.max() < 10

    def test_partial_flipping(self):
        """Only specified fraction of labels should change."""
        torch.manual_seed(42)
        labels = torch.zeros(100, dtype=torch.long)
        original_labels = labels.clone()
        result = apply_label_flipping(
            labels, flip_fraction=0.3, num_classes=10, target_class=5
        )
        num_changed = (result != original_labels).sum().item()
        # Should flip approximately 30 labels (30% of 100)
        assert 25 <= num_changed <= 35

    def test_empty_labels(self):
        """Should handle empty label tensor."""
        labels = torch.tensor([], dtype=torch.long)
        result = apply_label_flipping(labels, flip_fraction=0.5)
        assert len(result) == 0


class TestGaussianNoise:
    """Test Gaussian noise attack."""

    def test_noise_added_to_images(self):
        """Images should be modified by noise addition."""
        torch.manual_seed(42)
        images = torch.ones(10, 3, 28, 28) * 0.5
        result = apply_gaussian_noise(images, mean=0.0, std=0.1)
        # Should not be identical
        assert not torch.equal(result, images)

    def test_output_clamped(self):
        """Output should be clamped to [0, 1] range."""
        torch.manual_seed(42)
        images = torch.ones(10, 3, 28, 28) * 0.9
        result = apply_gaussian_noise(images, mean=0.0, std=0.5)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_std_no_change(self):
        """Zero standard deviation should result in no change."""
        images = torch.ones(10, 3, 28, 28) * 0.5
        result = apply_gaussian_noise(images, mean=0.0, std=0.0)
        assert torch.equal(result, images)

    def test_noise_distribution(self):
        """Noise should approximately follow Gaussian distribution."""
        torch.manual_seed(42)
        images = torch.ones(1000, 3, 28, 28) * 0.5
        result = apply_gaussian_noise(images, mean=0.0, std=0.1)
        noise = result - images
        # Mean should be close to 0
        assert abs(noise.mean().item()) < 0.01
        # Std should be close to 0.1 (before clamping)
        assert abs(noise.std().item() - 0.1) < 0.02


class TestBrightnessAttack:
    """Test brightness modification attack."""

    def test_brightness_reduction(self):
        """Factor < 1.0 should darken images."""
        images = torch.ones(10, 3, 28, 28) * 0.8
        result = apply_brightness_attack(images, factor=0.5)
        assert (result < images).all()
        assert torch.allclose(result, images * 0.5)

    def test_brightness_increase(self):
        """Factor > 1.0 should brighten images."""
        images = torch.ones(10, 3, 28, 28) * 0.3
        result = apply_brightness_attack(images, factor=2.0)
        assert (result > images).all()

    def test_output_clamped(self):
        """Output should be clamped to [0, 1] range."""
        images = torch.ones(10, 3, 28, 28) * 0.9
        result = apply_brightness_attack(images, factor=2.0)
        assert result.max() <= 1.0

    def test_zero_factor_makes_black(self):
        """Factor of 0 should result in black images."""
        images = torch.ones(10, 3, 28, 28) * 0.8
        result = apply_brightness_attack(images, factor=0.0)
        assert torch.equal(result, torch.zeros_like(images))

    def test_one_factor_no_change(self):
        """Factor of 1.0 should not change images."""
        images = torch.rand(10, 3, 28, 28)
        result = apply_brightness_attack(images, factor=1.0)
        assert torch.equal(result, images)


class TestTokenReplacement:
    """Test token replacement attack."""

    def test_zero_probability_no_change(self):
        """Zero replacement probability should not change tokens."""
        tokens = torch.randint(0, 1000, (10, 20))
        result = apply_token_replacement(tokens, replacement_prob=0.0)
        assert torch.equal(result, tokens)

    def test_tokens_replaced(self):
        """Some tokens should be replaced with random tokens."""
        torch.manual_seed(42)
        tokens = torch.zeros(100, 20, dtype=torch.long)
        result = apply_token_replacement(tokens, replacement_prob=0.3)
        # Some tokens should have changed
        assert not torch.equal(result, tokens)

    def test_vocab_size_respected(self):
        """Replacement tokens should be within vocab size."""
        torch.manual_seed(42)
        tokens = torch.randint(0, 1000, (10, 20))
        vocab_size = 5000
        result = apply_token_replacement(
            tokens, replacement_prob=0.5, vocab_size=vocab_size
        )
        assert result.min() >= 0
        assert result.max() < vocab_size

    def test_replacement_frequency(self):
        """Replacement frequency should match probability."""
        torch.manual_seed(42)
        tokens = torch.zeros(1000, 100, dtype=torch.long)
        result = apply_token_replacement(tokens, replacement_prob=0.2)
        num_replaced = (result != tokens).sum().item()
        total_tokens = tokens.numel()
        expected = total_tokens * 0.2
        assert abs(num_replaced - expected) < expected * 0.1


class TestShouldPoisonThisRound:
    """Test round-based poisoning scheduling."""

    def test_no_schedule_no_poisoning(self):
        """No attack schedule should result in no poisoning."""
        should_poison, configs = should_poison_this_round(5, 0, None)
        assert not should_poison
        assert configs == []

    def test_empty_schedule_no_poisoning(self):
        """Empty attack schedule should result in no poisoning."""
        should_poison, configs = should_poison_this_round(5, 0, [])
        assert not should_poison
        assert configs == []

    def test_specific_client_selection(self):
        """Specific clients should be poisoned when selected."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "selection_strategy": "specific",
                "client_ids": [0, 2, 5],
                "attack_config": {"type": "label_flipping", "params": {}},
            }
        ]
        should_poison, configs = should_poison_this_round(5, 2, schedule)
        assert should_poison
        assert len(configs) == 1
        assert configs[0] == {"type": "label_flipping", "params": {}}

        should_poison, configs = should_poison_this_round(5, 1, schedule)
        assert not should_poison
        assert configs == []

    def test_round_range_filtering(self):
        """Poisoning should only occur within round range."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 10,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "label_flipping", "params": {}},
            }
        ]
        should_poison, configs = should_poison_this_round(3, 0, schedule)
        assert not should_poison
        assert configs == []

        should_poison, configs = should_poison_this_round(7, 0, schedule)
        assert should_poison
        assert len(configs) == 1

        should_poison, configs = should_poison_this_round(12, 0, schedule)
        assert not should_poison
        assert configs == []

    def test_random_selection_strategy(self):
        """Random selection should use _selected_clients."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "selection_strategy": "random",
                "_selected_clients": [1, 3, 7],
                "attack_config": {"type": "gaussian_noise", "params": {}},
            }
        ]
        should_poison, configs = should_poison_this_round(5, 3, schedule)
        assert should_poison
        assert len(configs) == 1

        should_poison, configs = should_poison_this_round(5, 5, schedule)
        assert not should_poison
        assert configs == []

    def test_percentage_selection_strategy(self):
        """Percentage selection should use _selected_clients."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 10,
                "selection_strategy": "percentage",
                "_selected_clients": [0, 2, 4],
                "attack_config": {"type": "brightness", "params": {}},
            }
        ]
        should_poison, configs = should_poison_this_round(5, 2, schedule)
        assert should_poison
        assert len(configs) == 1

        should_poison, configs = should_poison_this_round(5, 1, schedule)
        assert not should_poison
        assert configs == []

    def test_multiple_attack_phases(self):
        """Should handle multiple non-overlapping attack phases correctly."""
        schedule = [
            {
                "start_round": 1,
                "end_round": 5,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "label_flipping", "params": {}},
            },
            {
                "start_round": 6,
                "end_round": 10,
                "selection_strategy": "specific",
                "client_ids": [1],
                "attack_config": {"type": "gaussian_noise", "params": {}},
            },
        ]
        should_poison, configs = should_poison_this_round(3, 0, schedule)
        assert should_poison
        assert len(configs) == 1
        assert configs[0]["type"] == "label_flipping"

        should_poison, configs = should_poison_this_round(8, 1, schedule)
        assert should_poison
        assert len(configs) == 1
        assert configs[0]["type"] == "gaussian_noise"

    def test_overlapping_schedules_different_attack_types(self):
        """Overlapping schedules with different attack types should stack."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 10,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {
                    "type": "label_flipping",
                    "params": {"flip_fraction": 0.5},
                },
            },
            {
                "start_round": 7,
                "end_round": 12,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "gaussian_noise", "params": {"std": 0.1}},
            },
        ]
        should_poison, configs = should_poison_this_round(8, 0, schedule)
        assert should_poison
        assert len(configs) == 2
        attack_types = {config["type"] for config in configs}
        assert attack_types == {"label_flipping", "gaussian_noise"}

    def test_overlapping_schedules_same_attack_type_deduplication(self):
        """Overlapping schedules with same attack type should deduplicate (first wins)."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 10,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {
                    "type": "label_flipping",
                    "params": {"flip_fraction": 0.3, "num_classes": 10},
                },
            },
            {
                "start_round": 7,
                "end_round": 12,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {
                    "type": "label_flipping",
                    "params": {"flip_fraction": 0.8, "num_classes": 10},
                },
            },
        ]
        should_poison, configs = should_poison_this_round(8, 0, schedule)
        assert should_poison
        assert len(configs) == 1
        assert configs[0]["type"] == "label_flipping"
        assert configs[0]["params"]["flip_fraction"] == 0.3

    def test_three_way_overlap_mixed_types(self):
        """Three overlapping schedules with mixed attack types."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 15,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "label_flipping", "params": {}},
            },
            {
                "start_round": 8,
                "end_round": 12,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "gaussian_noise", "params": {}},
            },
            {
                "start_round": 10,
                "end_round": 20,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "brightness", "params": {}},
            },
        ]
        should_poison, configs = should_poison_this_round(11, 0, schedule)
        assert should_poison
        assert len(configs) == 3
        attack_types = {config["type"] for config in configs}
        assert attack_types == {"label_flipping", "gaussian_noise", "brightness"}

    def test_three_way_overlap_with_duplicates(self):
        """Three overlapping schedules with duplicate attack type."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 15,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {
                    "type": "label_flipping",
                    "params": {"flip_fraction": 0.3},
                },
            },
            {
                "start_round": 8,
                "end_round": 12,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "gaussian_noise", "params": {}},
            },
            {
                "start_round": 10,
                "end_round": 20,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {
                    "type": "label_flipping",
                    "params": {"flip_fraction": 0.9},
                },
            },
        ]
        # Round 11 - all three overlap, but two are same type
        should_poison, configs = should_poison_this_round(11, 0, schedule)
        assert should_poison
        assert len(configs) == 2  # Deduplicated
        attack_types = {config["type"] for config in configs}
        assert attack_types == {"label_flipping", "gaussian_noise"}
        # First label_flipping should win
        label_flip_config = next(c for c in configs if c["type"] == "label_flipping")
        assert label_flip_config["params"]["flip_fraction"] == 0.3

    def test_partial_overlap_transitions(self):
        """Test behavior as rounds transition through overlapping phases."""
        schedule = [
            {
                "start_round": 5,
                "end_round": 10,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "label_flipping", "params": {}},
            },
            {
                "start_round": 8,
                "end_round": 12,
                "selection_strategy": "specific",
                "client_ids": [0],
                "attack_config": {"type": "gaussian_noise", "params": {}},
            },
        ]
        # Round 6 - only first phase
        should_poison, configs = should_poison_this_round(6, 0, schedule)
        assert should_poison
        assert len(configs) == 1
        assert configs[0]["type"] == "label_flipping"

        # Round 9 - both phases overlap
        should_poison, configs = should_poison_this_round(9, 0, schedule)
        assert should_poison
        assert len(configs) == 2

        # Round 11 - only second phase
        should_poison, configs = should_poison_this_round(11, 0, schedule)
        assert should_poison
        assert len(configs) == 1
        assert configs[0]["type"] == "gaussian_noise"


class TestApplyPoisoningAttack:
    """Test unified poisoning attack application."""

    def test_label_flipping_attack(self):
        """Label flipping attack should modify labels."""
        data = torch.rand(10, 3, 28, 28)
        labels = torch.zeros(10, dtype=torch.long)
        config = {
            "type": "label_flipping",
            "params": {"flip_fraction": 1.0, "num_classes": 10, "target_class": 5},
        }
        new_data, new_labels = apply_poisoning_attack(data, labels, config)
        # Data should be unchanged
        assert torch.equal(new_data, data)
        # Labels should all be 5
        assert torch.all(new_labels == 5)

    def test_gaussian_noise_attack(self):
        """Gaussian noise attack should modify data."""
        torch.manual_seed(42)
        data = torch.ones(10, 3, 28, 28) * 0.5
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        config = {"type": "gaussian_noise", "params": {"mean": 0.0, "std": 0.1}}
        new_data, new_labels = apply_poisoning_attack(data, labels, config)
        # Data should be modified
        assert not torch.equal(new_data, data)
        # Labels should be unchanged
        assert torch.equal(new_labels, labels)

    def test_brightness_attack(self):
        """Brightness attack should modify data."""
        data = torch.ones(10, 3, 28, 28) * 0.8
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        config = {"type": "brightness", "params": {"factor": 0.5}}
        new_data, new_labels = apply_poisoning_attack(data, labels, config)
        # Data should be darkened
        assert (new_data < data).all()
        # Labels should be unchanged
        assert torch.equal(new_labels, labels)

    def test_token_replacement_attack(self):
        """Token replacement attack should modify tokens."""
        torch.manual_seed(42)
        data = torch.zeros(10, 20, dtype=torch.long)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        config = {
            "type": "token_replacement",
            "params": {"replacement_prob": 0.5, "vocab_size": 1000},
        }
        new_data, new_labels = apply_poisoning_attack(data, labels, config)
        # Data should be modified
        assert not torch.equal(new_data, data)
        # Labels should be unchanged
        assert torch.equal(new_labels, labels)

    def test_unknown_attack_type(self):
        """Unknown attack type should return unchanged data."""
        data = torch.rand(10, 3, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        config = {"type": "unknown_attack", "params": {}}
        new_data, new_labels = apply_poisoning_attack(data, labels, config)
        # Both should be unchanged
        assert torch.equal(new_data, data)
        assert torch.equal(new_labels, labels)

    def test_missing_params(self):
        """Missing params should use defaults."""
        data = torch.ones(10, 3, 28, 28) * 0.5
        labels = torch.zeros(10, dtype=torch.long)
        config = {"type": "gaussian_noise"}  # No params specified
        new_data, new_labels = apply_poisoning_attack(data, labels, config)
        # Should still apply attack with default params
        assert not torch.equal(new_data, data)
