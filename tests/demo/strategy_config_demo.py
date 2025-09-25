#!/usr/bin/env python3
"""
Example demonstrating strategy-based client configuration for federated learning.

This shows how the strategy-based client configuration automatically fixes
common convergence issues by ensuring appropriate client participation
for different aggregation strategies.

USAGE:
------
To run this example:

    PYTHONPATH=. python tests/demo/strategy_config_demo.py

For comprehensive testing patterns and guidance on extending this system,
see the detailed testing guide at tests/docs/testing_guide.md section 4.

QUICK EXPERIMENT:
-----------------
Try modifying the strategy examples below to see different behaviors:
- Change num_of_clients values
- Add research_mode: True to any config
- Test with different strategy names
"""

from src.config_loaders.strategy_client_config import (
    apply_client_config,
    validate_client_config,
    analyze_client_requirements,
)


def demonstrate_strategy_config():
    """Demonstrate strategy-based client configuration with different strategies."""

    print("STRATEGY-BASED CLIENT CONFIGURATION DEMO")
    print("=" * 50)

    # Example 1: Trust-based strategy (NEEDS consistent participation)
    print("\nEXAMPLE 1: Trust-based Strategy")
    trust_config = {
        "aggregation_strategy_keyword": "trust",
        "num_of_clients": 5,
        "num_of_malicious_clients": 1,
        "trust_threshold": 0.7,
        "beta_value": 0.5,
        # Note: min_fit_clients, min_evaluate_clients not specified
    }

    print("Before strategy-based config:")
    print(f"  Strategy: {trust_config['aggregation_strategy_keyword']}")
    print(f"  Total clients: {trust_config['num_of_clients']}")
    print(f"  min_fit_clients: {trust_config.get('min_fit_clients', 'NOT SET')}")
    print(
        f"  min_evaluate_clients: {trust_config.get('min_evaluate_clients', 'NOT SET')}"
    )

    # Apply strategy-based configuration
    trust_config_updated = apply_client_config(trust_config.copy())

    print("\nAfter strategy-based config:")
    print(f"  min_fit_clients: {trust_config_updated['min_fit_clients']}")
    print(f"  min_evaluate_clients: {trust_config_updated['min_evaluate_clients']}")
    print(f"  min_available_clients: {trust_config_updated['min_available_clients']}")

    # Example 2: Multi-Krum strategy (NEEDS consistent participation)
    print("\n\nEXAMPLE 2: Multi-Krum Strategy")
    krum_config = {
        "aggregation_strategy_keyword": "multi-krum",
        "num_of_clients": 8,
        "num_of_malicious_clients": 2,
        "num_krum_selections": 5,
        # Note: Client participation not specified
    }

    print("Before strategy-based config:")
    print(f"  Strategy: {krum_config['aggregation_strategy_keyword']}")
    print(f"  Total clients: {krum_config['num_of_clients']}")
    print(f"  min_fit_clients: {krum_config.get('min_fit_clients', 'NOT SET')}")

    # Apply strategy-based configuration
    krum_config_updated = apply_client_config(krum_config.copy())

    print("\nAfter strategy-based config:")
    print(f"  min_fit_clients: {krum_config_updated['min_fit_clients']}")
    print(f"  min_evaluate_clients: {krum_config_updated['min_evaluate_clients']}")

    # Example 3: Strategy requiring all clients
    print("\n\nEXAMPLE 3: Byzantine-Robust Strategy")
    byzantine_config = {
        "aggregation_strategy_keyword": "trust",
        "num_of_clients": 10,
        "num_of_malicious_clients": 2,
        "trust_threshold": 0.6,
    }

    byzantine_config_updated = apply_client_config(byzantine_config.copy())

    print("Byzantine-robust strategy auto-configured:")
    print(
        f"  min_fit_clients: {byzantine_config_updated['min_fit_clients']} (= {byzantine_config['num_of_clients']})"
    )
    print("  Trust strategies require all clients for reputation building")

    # Example 4: Manual configuration with warnings
    print("\n\nEXAMPLE 4: Manual Config with Warnings")
    manual_config = {
        "aggregation_strategy_keyword": "pid",
        "num_of_clients": 6,
        "min_fit_clients": 3,  # Manually set to less than total
        "min_evaluate_clients": 3,
        "Kp": 1.0,
        "Ki": 0.1,
        "Kd": 0.01,
    }

    print("Manual configuration detected:")
    print(f"  User set min_fit_clients = {manual_config['min_fit_clients']}")
    print(
        f"  But strategy '{manual_config['aggregation_strategy_keyword']}' needs all {manual_config['num_of_clients']} clients"
    )

    # Show validation warnings
    issues = validate_client_config(manual_config)
    for issue in issues:
        print(f"  {issue}")


def show_strategy_analysis():
    """Show detailed analysis for different strategies."""

    print("\n\nSTRATEGY ANALYSIS")
    print("=" * 50)

    strategies_to_test = [
        ("trust", 5, 1),
        ("multi-krum", 8, 2),
        ("pid", 6, 1),
        ("trimmed_mean", 10, 2),
        ("unknown_strategy", 4, 0),
    ]

    for strategy, num_clients, num_malicious in strategies_to_test:
        recommendation = analyze_client_requirements(
            strategy, num_clients, num_malicious
        )

        print(f"\n{strategy.upper()}:")
        print(f"   Clients: {num_clients} total, {num_malicious} malicious")
        print(f"   Recommendation: min_fit = {recommendation.min_fit_clients}")
        print(f"   Reasoning: {recommendation.reasoning}")
        if recommendation.warning_message:
            print(f"   WARNING: {recommendation.warning_message}")


if __name__ == "__main__":
    demonstrate_strategy_config()
    show_strategy_analysis()

    print("\n\nSUMMARY:")
    print("The strategy-based client configuration automatically:")
    print("• Sets min_clients = total_clients for Byzantine-robust strategies")
    print("• Uses flexible participation (80%) for strategies that support it")
    print("• Provides clear explanations for configuration choices")
    print("• Validates configurations and warns about potential issues")

    print("\nByzantine-robust strategies now work correctly by default!")
    print("Flexible strategies allow variable participation for experimentation.")
