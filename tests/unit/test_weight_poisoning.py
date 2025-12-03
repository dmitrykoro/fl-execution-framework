"""Unit tests for weight-level poisoning attacks."""

from tests.common import np

from src.attack_utils.weight_poisoning import (
    apply_model_poisoning,
    apply_gradient_scaling,
    apply_byzantine_perturbation,
    apply_weight_poisoning,
    is_weight_attack,
    WEIGHT_ATTACK_TYPES,
)


class TestWeightAttackTypes:
    """Tests for weight attack type constants and helpers."""

    def test_weight_attack_types_contains_expected_types(self):
        """Verify WEIGHT_ATTACK_TYPES includes all expected attack types."""
        expected = {"model_poisoning", "gradient_scaling", "byzantine_perturbation"}
        assert WEIGHT_ATTACK_TYPES == expected

    def test_is_weight_attack_returns_true_for_weight_attacks(self):
        """Verify is_weight_attack returns True for weight attack types."""
        for attack_type in WEIGHT_ATTACK_TYPES:
            assert is_weight_attack(attack_type) is True

    def test_is_weight_attack_returns_false_for_data_attacks(self):
        """Verify is_weight_attack returns False for data poisoning types."""
        data_attacks = ["label_flipping", "gaussian_noise"]
        for attack_type in data_attacks:
            assert is_weight_attack(attack_type) is False

    def test_is_weight_attack_returns_false_for_unknown(self):
        """Verify is_weight_attack returns False for unknown types."""
        assert is_weight_attack("unknown_attack") is False
        assert is_weight_attack("") is False


class TestModelPoisoning:
    """Tests for model poisoning attack."""

    def test_model_poisoning_modifies_parameters(self):
        """Verify model poisoning changes parameter values."""
        # Use non-zero values since poisoning multiplies by magnitude
        params = [np.ones((10, 10)), np.ones((5,))]
        poisoned = apply_model_poisoning(params, poison_ratio=0.1, magnitude=50.0)

        # Check shapes preserved
        assert len(poisoned) == len(params)
        for orig, pois in zip(params, poisoned):
            assert orig.shape == pois.shape

        # Check some values changed (1 * 50 != 1)
        assert not np.allclose(params[0], poisoned[0])

    def test_model_poisoning_with_seed_is_deterministic(self):
        """Verify model poisoning with seed produces consistent results."""
        params = [np.random.randn(100, 100)]

        poisoned1 = apply_model_poisoning(params, seed=42)
        poisoned2 = apply_model_poisoning(params, seed=42)

        assert np.allclose(poisoned1[0], poisoned2[0])

    def test_model_poisoning_different_seeds_differ(self):
        """Verify different seeds produce different results."""
        params = [np.random.randn(100, 100)]

        poisoned1 = apply_model_poisoning(params, seed=42)
        poisoned2 = apply_model_poisoning(params, seed=123)

        assert not np.allclose(poisoned1[0], poisoned2[0])

    def test_model_poisoning_respects_poison_ratio(self):
        """Verify poison_ratio controls fraction of modified weights."""
        params = [np.ones((1000,))]
        poison_ratio = 0.1

        poisoned = apply_model_poisoning(
            params, poison_ratio=poison_ratio, magnitude=100.0, seed=42
        )

        # Count changed values
        changed = np.sum(np.abs(poisoned[0] - params[0]) > 0.01)
        expected_changed = int(1000 * poison_ratio)

        # Allow small tolerance
        assert abs(changed - expected_changed) <= 5

    def test_model_poisoning_magnitude_affects_values(self):
        """Verify magnitude parameter affects poisoned value scale."""
        params = [np.ones((100,))]

        poisoned_small = apply_model_poisoning(params, magnitude=2.0, seed=42)
        poisoned_large = apply_model_poisoning(params, magnitude=100.0, seed=42)

        # Larger magnitude should produce larger deviations
        max_small = np.max(np.abs(poisoned_small[0]))
        max_large = np.max(np.abs(poisoned_large[0]))

        assert max_large > max_small


class TestGradientScaling:
    """Tests for gradient scaling attack."""

    def test_gradient_scaling_scales_all_parameters(self):
        """Verify gradient scaling multiplies all weights."""
        params = [np.ones((10, 10)), np.ones((5,))]
        scale_factor = 3.0

        scaled = apply_gradient_scaling(params, scale_factor=scale_factor)

        for orig, scal in zip(params, scaled):
            assert np.allclose(scal, orig * scale_factor)

    def test_gradient_scaling_preserves_shapes(self):
        """Verify gradient scaling preserves parameter shapes."""
        params = [np.random.randn(10, 20), np.random.randn(5, 5, 5)]

        scaled = apply_gradient_scaling(params, scale_factor=2.0)

        for orig, scal in zip(params, scaled):
            assert orig.shape == scal.shape

    def test_gradient_scaling_with_zero_factor(self):
        """Verify scale_factor=0 produces zero parameters."""
        params = [np.random.randn(10, 10)]

        scaled = apply_gradient_scaling(params, scale_factor=0.0)

        assert np.allclose(scaled[0], 0.0)

    def test_gradient_scaling_with_negative_factor(self):
        """Verify negative scale_factor inverts signs."""
        params = [np.ones((10,))]

        scaled = apply_gradient_scaling(params, scale_factor=-2.0)

        assert np.allclose(scaled[0], -2.0)


class TestByzantinePerturbation:
    """Tests for Byzantine perturbation attack."""

    def test_byzantine_perturbation_replaces_values(self):
        """Verify Byzantine perturbation replaces original values."""
        params = [np.zeros((10, 10)), np.zeros((5,))]

        perturbed = apply_byzantine_perturbation(params, noise_scale=15.0, seed=42)

        # Check shapes preserved
        for orig, pert in zip(params, perturbed):
            assert orig.shape == pert.shape

        # Check values are now non-zero (random noise)
        assert not np.allclose(perturbed[0], 0.0)

    def test_byzantine_perturbation_with_seed_is_deterministic(self):
        """Verify Byzantine perturbation with seed is reproducible."""
        params = [np.zeros((100, 100))]

        perturbed1 = apply_byzantine_perturbation(params, seed=42)
        perturbed2 = apply_byzantine_perturbation(params, seed=42)

        assert np.allclose(perturbed1[0], perturbed2[0])

    def test_byzantine_perturbation_scale_affects_magnitude(self):
        """Verify noise_scale affects perturbation magnitude."""
        params = [np.zeros((1000,))]

        perturbed_small = apply_byzantine_perturbation(params, noise_scale=1.0, seed=42)
        perturbed_large = apply_byzantine_perturbation(
            params, noise_scale=15.0, seed=42
        )

        std_small = np.std(perturbed_small[0])
        std_large = np.std(perturbed_large[0])

        # Larger scale should produce larger standard deviation
        assert std_large > std_small * 10  # 15x scale difference


class TestApplyWeightPoisoning:
    """Tests for the main apply_weight_poisoning dispatcher."""

    def test_apply_weight_poisoning_model_poisoning(self):
        """Verify dispatcher handles model_poisoning correctly."""
        # Use non-zero values since poisoning multiplies by magnitude
        params = [np.ones((10, 10))]
        configs = [{"attack_type": "model_poisoning", "poison_ratio": 0.1}]

        poisoned = apply_weight_poisoning(params, configs)

        assert not np.allclose(poisoned[0], params[0])

    def test_apply_weight_poisoning_gradient_scaling(self):
        """Verify dispatcher handles gradient_scaling correctly."""
        params = [np.ones((10,))]
        configs = [{"attack_type": "gradient_scaling", "scale_factor": 5.0}]

        poisoned = apply_weight_poisoning(params, configs)

        assert np.allclose(poisoned[0], 5.0)

    def test_apply_weight_poisoning_byzantine(self):
        """Verify dispatcher handles byzantine_perturbation correctly."""
        params = [np.zeros((10,))]
        configs = [{"attack_type": "byzantine_perturbation", "seed": 42}]

        poisoned = apply_weight_poisoning(params, configs)

        assert not np.allclose(poisoned[0], 0.0)

    def test_apply_weight_poisoning_skips_data_attacks(self):
        """Verify dispatcher ignores data poisoning attack types."""
        params = [np.ones((10,))]
        configs = [{"attack_type": "label_flipping"}]  # Data attack, not weight

        poisoned = apply_weight_poisoning(params, configs)

        # Should be unchanged since label_flipping is not a weight attack
        assert np.allclose(poisoned[0], params[0])

    def test_apply_weight_poisoning_multiple_attacks(self):
        """Verify dispatcher applies multiple attacks sequentially."""
        params = [np.ones((10,))]
        configs = [
            {"attack_type": "gradient_scaling", "scale_factor": 2.0},
            {"attack_type": "gradient_scaling", "scale_factor": 3.0},
        ]

        poisoned = apply_weight_poisoning(params, configs)

        # Should be scaled by 2 then by 3 = 6x
        assert np.allclose(poisoned[0], 6.0)

    def test_apply_weight_poisoning_empty_configs(self):
        """Verify dispatcher handles empty config list."""
        params = [np.ones((10,))]

        poisoned = apply_weight_poisoning(params, [])

        assert np.allclose(poisoned[0], params[0])

    def test_apply_weight_poisoning_uses_config_params(self):
        """Verify dispatcher extracts parameters from config."""
        params = [np.ones((100,))]
        configs = [
            {
                "attack_type": "model_poisoning",
                "poison_ratio": 0.5,
                "magnitude": 10.0,
                "seed": 123,
            }
        ]

        poisoned = apply_weight_poisoning(params, configs)

        # Should have approximately 50% of values modified
        changed = np.sum(np.abs(poisoned[0] - params[0]) > 0.01)
        assert 40 <= changed <= 60  # Allow tolerance


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_parameter_list(self):
        """Verify attacks handle empty parameter lists."""
        params = []

        assert apply_model_poisoning(params) == []
        assert apply_gradient_scaling(params) == []
        assert apply_byzantine_perturbation(params) == []

    def test_single_element_parameters(self):
        """Verify attacks handle single-element arrays."""
        params = [np.array([1.0])]

        poisoned = apply_model_poisoning(params, poison_ratio=1.0, magnitude=10.0)
        assert poisoned[0].shape == (1,)

    def test_preserves_dtype(self):
        """Verify attacks preserve parameter dtype."""
        params_f32 = [np.ones((10,), dtype=np.float32)]
        params_f64 = [np.ones((10,), dtype=np.float64)]

        poisoned_f32 = apply_byzantine_perturbation(params_f32, seed=42)
        poisoned_f64 = apply_byzantine_perturbation(params_f64, seed=42)

        assert poisoned_f32[0].dtype == np.float32
        assert poisoned_f64[0].dtype == np.float64

    def test_large_parameters(self):
        """Verify attacks handle large parameter arrays."""
        params = [np.random.randn(1000, 1000)]

        poisoned = apply_model_poisoning(params, seed=42)

        assert poisoned[0].shape == (1000, 1000)
