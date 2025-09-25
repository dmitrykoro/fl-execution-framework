"""
Unit tests for strategy client configuration.

Tests automatic client configuration for Byzantine-robust strategies,
validation warnings, and integration with ConfigLoader.
"""

from src.config_loaders.strategy_client_config import (
    analyze_client_requirements,
    apply_client_config,
    validate_client_config,
    CONSISTENT_PARTICIPATION_STRATEGIES,
    FLEXIBLE_PARTICIPATION_STRATEGIES,
)


class TestAnalyzeClientRequirements:
    """Test strategy classification and client requirements analysis."""

    def test_consistent_participation_strategies(self):
        """Test that consistent participation strategies require all clients."""
        for strategy in CONSISTENT_PARTICIPATION_STRATEGIES.keys():
            result = analyze_client_requirements(strategy, num_of_clients=10)
            assert result.min_fit_clients == 10
            assert result.min_evaluate_clients == 10
            assert result.min_available_clients == 10
            assert "auto-configured:" in result.reasoning.lower()
            assert result.warning_message is not None

    def test_flexible_participation_strategies(self):
        """Test that flexible participation strategies use 80% of clients."""
        for strategy in FLEXIBLE_PARTICIPATION_STRATEGIES.keys():
            result = analyze_client_requirements(strategy, num_of_clients=10)
            assert result.min_fit_clients == 8  # 80% of 10
            assert result.min_evaluate_clients == 8
            assert result.min_available_clients == 10

    def test_unknown_strategy_conservative_defaults(self):
        """Test that unknown strategies get conservative 60% defaults."""
        result = analyze_client_requirements("unknown_strategy", num_of_clients=10)
        assert result.min_fit_clients == 6  # 60% of 10
        assert result.min_evaluate_clients == 6
        assert result.min_available_clients == 10
        assert "conservative defaults" in result.reasoning.lower()
        assert result.warning_message is not None
        assert "unknown strategy" in result.warning_message.lower()

    def test_minimum_client_threshold(self):
        """Test that minimum client count is enforced."""
        # Test with small client count
        result = analyze_client_requirements("trimmed_mean", num_of_clients=3)
        assert result.min_fit_clients == 3  # max(3, 80% of 3) = max(3, 2.4) = 3

        result = analyze_client_requirements("unknown_strategy", num_of_clients=4)
        assert result.min_fit_clients == 3  # max(3, 60% of 4) = max(3, 2.4) = 3


class TestApplyClientConfig:
    """Test client configuration application and manual override behavior."""

    def test_auto_configuration_trust_strategy(self):
        """Test auto-configuration for trust strategy."""
        config = {"aggregation_strategy_keyword": "trust", "num_of_clients": 5}

        result = apply_client_config(config, verbose=False)
        assert result["min_fit_clients"] == 5
        assert result["min_evaluate_clients"] == 5
        assert result["min_available_clients"] == 5

    def test_auto_configuration_krum_strategy(self):
        """Test auto-configuration for krum strategy."""
        config = {"aggregation_strategy_keyword": "krum", "num_of_clients": 8}

        result = apply_client_config(config, verbose=False)
        assert result["min_fit_clients"] == 8
        assert result["min_evaluate_clients"] == 8
        assert result["min_available_clients"] == 8

    def test_manual_configuration_preservation(self):
        """Test that manual configuration is preserved."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "num_of_clients": 5,
            "min_fit_clients": 3,
        }

        result = apply_client_config(config, verbose=False)
        assert result["min_fit_clients"] == 3  # Manual override preserved
        # When user configures any parameter, no auto-configuration happens
        assert (
            "min_evaluate_clients" not in result
            or result.get("min_evaluate_clients") is None
        )
        assert (
            "min_available_clients" not in result
            or result.get("min_available_clients") is None
        )

    def test_partial_manual_configuration(self):
        """Test that partial manual configuration works correctly."""
        config = {
            "aggregation_strategy_keyword": "pid",
            "num_of_clients": 6,
            "min_fit_clients": 4,
            "min_evaluate_clients": 4,
        }

        result = apply_client_config(config, verbose=False)
        assert result["min_fit_clients"] == 4  # Manual override
        assert result["min_evaluate_clients"] == 4  # Manual override
        # When user configures any parameter, no auto-configuration happens
        assert (
            "min_available_clients" not in result
            or result.get("min_available_clients") is None
        )

    def test_missing_required_parameters(self):
        """Test behavior with missing required parameters."""
        config = {"aggregation_strategy_keyword": "trust"}  # Missing num_of_clients
        result = apply_client_config(config, verbose=False)
        # Should return original config unchanged
        assert "min_fit_clients" not in result

        config = {"num_of_clients": 5}  # Missing strategy_keyword
        result = apply_client_config(config, verbose=False)
        # Should return original config unchanged
        assert "min_fit_clients" not in result

    def test_flexible_strategy_configuration(self):
        """Test configuration for flexible participation strategies."""
        config = {"aggregation_strategy_keyword": "trimmed_mean", "num_of_clients": 10}

        result = apply_client_config(config, verbose=False)
        assert result["min_fit_clients"] == 8  # 80% of 10
        assert result["min_evaluate_clients"] == 8
        assert result["min_available_clients"] == 10


class TestValidateClientConfig:
    """Test validation warnings and error detection."""

    def test_convergence_risk_warning_trust(self):
        """Test convergence risk warning for trust strategy."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "num_of_clients": 5,
            "min_fit_clients": 3,
            "min_evaluate_clients": 3,
        }

        issues = validate_client_config(config)
        assert any("CONVERGENCE RISK" in issue for issue in issues)
        assert any("trust may not converge properly" in issue for issue in issues)

    def test_convergence_risk_warning_krum(self):
        """Test convergence risk warning for krum strategy."""
        config = {
            "aggregation_strategy_keyword": "krum",
            "num_of_clients": 8,
            "min_fit_clients": 6,
            "min_evaluate_clients": 8,
        }

        issues = validate_client_config(config)
        assert any("CONVERGENCE RISK" in issue for issue in issues)

    def test_statistical_significance_warning(self):
        """Test statistical significance warning for insufficient clients."""
        config = {
            "aggregation_strategy_keyword": "rfa",
            "num_of_clients": 2,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
        }

        issues = validate_client_config(config)
        assert any("STATISTICAL WARNING" in issue for issue in issues)
        assert any(
            "may not provide statistically significant results" in issue
            for issue in issues
        )

    def test_resource_allocation_error(self):
        """Test resource allocation error detection."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "num_of_clients": 5,
            "min_available_clients": 8,  # More than total clients
        }

        issues = validate_client_config(config)
        assert any("CONFIG ERROR" in issue for issue in issues)
        assert any(
            "Cannot require more clients than available" in issue for issue in issues
        )

    def test_no_issues_optimal_config(self):
        """Test that optimal configurations produce no warnings."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "num_of_clients": 10,
            "min_fit_clients": 10,
            "min_evaluate_clients": 10,
            "min_available_clients": 10,
        }

        issues = validate_client_config(config)
        assert len(issues) == 0

    def test_flexible_strategy_no_convergence_warning(self):
        """Test that flexible strategies don't get convergence warnings."""
        config = {
            "aggregation_strategy_keyword": "trimmed_mean",
            "num_of_clients": 10,
            "min_fit_clients": 6,
            "min_evaluate_clients": 6,
        }

        issues = validate_client_config(config)
        convergence_warnings = [
            issue for issue in issues if "CONVERGENCE RISK" in issue
        ]
        assert len(convergence_warnings) == 0


class TestParameterRelationships:
    """Test parameter relationship validation for specific strategies."""

    def test_multi_krum_parameter_validation(self):
        """Test Multi-Krum parameter relationship validation."""
        # This would require extending validate_client_config to check strategy-specific parameters
        # For now, this is a placeholder for future parameter validation
        config = {
            "aggregation_strategy_keyword": "multi-krum",
            "num_of_clients": 10,
            "num_of_malicious_clients": 3,
            "num_krum_selections": 8,  # Should be ≤ (10-3) = 7
        }

        # Note: Current validate_client_config doesn't check this relationship
        # This test documents the intended behavior for future implementation
        validate_client_config(config)
        # Would expect validation of num_krum_selections in future versions

    def test_trust_strategy_parameter_presence(self):
        """Test that trust strategy configurations are properly handled."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "num_of_clients": 5,
            "min_fit_clients": 5,
            "min_evaluate_clients": 5,
            "trust_threshold": 0.8,
            "beta_value": 0.9,
            "begin_removing_from_round": 2,
            "num_of_clusters": 3,
        }

        # Basic validation should pass
        issues = validate_client_config(config)
        config_errors = [issue for issue in issues if "CONFIG ERROR" in issue]
        assert len(config_errors) == 0


class TestStrategyClassification:
    """Test strategy classification constants."""

    def test_consistent_participation_strategies_complete(self):
        """Test that all Byzantine-robust strategies are classified correctly."""
        expected_strategies = {
            "trust",
            "pid",
            "pid_scaled",
            "pid_standardized",
            "krum",
            "multi-krum",
            "multi-krum-based",
            "rfa",
            "bulyan",
        }

        actual_strategies = set(CONSISTENT_PARTICIPATION_STRATEGIES.keys())
        assert actual_strategies == expected_strategies

    def test_flexible_participation_strategies_complete(self):
        """Test that flexible participation strategies are classified correctly."""
        expected_strategies = {"trimmed_mean"}

        actual_strategies = set(FLEXIBLE_PARTICIPATION_STRATEGIES.keys())
        assert actual_strategies == expected_strategies

    def test_no_strategy_overlap(self):
        """Test that no strategy appears in both classification sets."""
        consistent_set = set(CONSISTENT_PARTICIPATION_STRATEGIES.keys())
        flexible_set = set(FLEXIBLE_PARTICIPATION_STRATEGIES.keys())

        overlap = consistent_set.intersection(flexible_set)
        assert len(overlap) == 0

    def test_all_strategies_have_descriptions(self):
        """Test that all classified strategies have meaningful descriptions."""
        for strategy, description in CONSISTENT_PARTICIPATION_STRATEGIES.items():
            assert len(description) > 20  # Reasonable description length
            # Description should explain the reasoning, may contain strategy name
            assert len(description.strip()) > 0

        for strategy, description in FLEXIBLE_PARTICIPATION_STRATEGIES.items():
            assert len(description) > 20
            # Description should explain the reasoning, may contain strategy name
            assert len(description.strip()) > 0


class TestConfigLoaderIntegration:
    """Test integration with ConfigLoader for automatic client configuration."""

    def test_config_loader_auto_applies_strategy_config(self):
        """Test that apply_client_config integrates properly with expected config structure."""
        # Simulate the config structure that ConfigLoader processes
        config = {"aggregation_strategy_keyword": "krum", "num_of_clients": 10}

        # This simulates what happens in config_loader.py line 55
        result = apply_client_config(config, verbose=False)

        # Verify auto-configuration was applied
        assert result["min_fit_clients"] == 10
        assert result["min_evaluate_clients"] == 10
        assert result["min_available_clients"] == 10

    def test_config_loader_preserves_manual_settings(self):
        """Test that manual settings are preserved during config processing."""
        config = {
            "aggregation_strategy_keyword": "trust",
            "num_of_clients": 8,
            "min_fit_clients": 6,  # Manual override
            "min_evaluate_clients": 6,
        }

        result = apply_client_config(config, verbose=False)

        # Manual settings should be preserved
        assert result["min_fit_clients"] == 6
        assert result["min_evaluate_clients"] == 6
        # When user configures any parameter, no auto-configuration happens
        assert (
            "min_available_clients" not in result
            or result.get("min_available_clients") is None
        )

    def test_validation_integration_with_warnings(self):
        """Test that validation warnings are properly generated for integration."""
        config = {
            "aggregation_strategy_keyword": "rfa",
            "num_of_clients": 2,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
        }

        # This simulates what happens in config_loader.py lines 61-64
        issues = validate_client_config(config)

        # Should generate statistical warning for insufficient clients
        assert len(issues) > 0
        assert any("STATISTICAL WARNING" in issue for issue in issues)

    def test_end_to_end_byzantine_strategy_workflow(self):
        """Test complete workflow for Byzantine-robust strategy configuration."""
        # Start with minimal config (as would come from user JSON)
        config = {"aggregation_strategy_keyword": "bulyan", "num_of_clients": 12}

        # Step 1: Apply client configuration (config_loader.py line 55)
        config = apply_client_config(config, verbose=False)
        assert config["min_fit_clients"] == 12
        assert config["min_evaluate_clients"] == 12

        # Step 2: Validate configuration (config_loader.py line 61)
        issues = validate_client_config(config)

        # Should have no issues for optimal Byzantine configuration
        critical_issues = [issue for issue in issues if "ERROR" in issue]
        assert len(critical_issues) == 0

    def test_end_to_end_flexible_strategy_workflow(self):
        """Test complete workflow for flexible participation strategy."""
        config = {"aggregation_strategy_keyword": "trimmed_mean", "num_of_clients": 10}

        # Apply configuration
        config = apply_client_config(config, verbose=False)
        assert config["min_fit_clients"] == 8  # 80% of 10
        assert config["min_evaluate_clients"] == 8

        # Validate configuration
        issues = validate_client_config(config)

        # Should have no convergence warnings for flexible strategies
        convergence_issues = [issue for issue in issues if "CONVERGENCE RISK" in issue]
        assert len(convergence_issues) == 0
