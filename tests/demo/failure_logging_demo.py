#!/usr/bin/env python3
"""
Demo showcasing the intelligent test failure logging system.

This demonstrates the custom pytest hooks in tests/conftest.py that provide
context-aware failure analysis and debugging hints.

USAGE:
------
To run this demo and see the logging in action:

    PYTHONPATH=. python -m pytest tests/demo/failure_logging_demo.py -v

Then check the generated log file in tests/logs/ for intelligent failure analysis.

WHAT THIS DEMONSTRATES:
-----------------------
- Automatic failure detection and logging
- Context-aware debugging hints based on error types
- Heuristic analysis of common testing issues
- Timestamped failure logs for debugging sessions
"""

from unittest.mock import Mock

from flwr.common import parameters_to_ndarrays
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)
from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class TestFailureLoggingDemo:
    """Demo class showing different types of test failures and logging responses."""

    def test_success_example(self):
        """This test passes - no logging triggered."""
        mock_strategy_history = Mock(spec=SimulationStrategyHistory)
        strategy = TrustBasedRemovalStrategy(
            trust_threshold=0.7,
            remove_clients=True,
            beta_value=0.5,
            begin_removing_from_round=2,
            strategy_history=mock_strategy_history,
        )
        assert strategy.trust_threshold == 0.7
        print("✅ This test passes - no failure logging")

    def test_import_error_demo(self):
        """Demo of ImportError logging (commented out to avoid actual failure)."""
        # Uncomment this to see ImportError logging in action:
        # from non_existent_module import something
        print("🔍 ImportError demo (commented out - uncomment to trigger)")

    def test_file_not_found_demo(self):
        """Demo of FileNotFoundError logging (commented out to avoid actual failure)."""
        # Uncomment this to see FileNotFoundError logging:
        # with open("non_existent_file.txt") as f:
        #     content = f.read()
        print("🔍 FileNotFoundError demo (commented out - uncomment to trigger)")

    def test_pytorch_shape_error_demo(self):
        """Demo of PyTorch shape error logging (commented out to avoid actual failure)."""
        # Uncomment this to see RuntimeError shape logging:
        # import torch
        # x = torch.randn(5, 10)
        # y = torch.randn(3, 8)
        # result = torch.matmul(x, y)  # Shape mismatch!
        print("🔍 PyTorch shape error demo (commented out - uncomment to trigger)")

    def test_assertion_error_demo(self):
        """Demo of AssertionError logging (commented out to avoid actual failure)."""
        # Uncomment this to see AssertionError logging:
        # expected = 42
        # actual = 24
        # assert expected == actual, f"Expected {expected}, got {actual}"
        print("🔍 AssertionError demo (commented out - uncomment to trigger)")

    def test_strategy_assertion_demo(self):
        """Demo of strategy-specific assertion logging (commented out to avoid actual failure)."""
        # This would trigger the strategy-specific logging heuristics:
        # strategy = TrustBasedRemovalStrategy(trust_threshold=0.7)
        # mock_results = []  # Empty results
        # aggregated = strategy.aggregate_fit(1, mock_results, [])
        # assert len(aggregated) > 0, "Should have aggregated parameters"
        print("🔍 Strategy assertion demo (commented out - uncomment to trigger)")

    def test_show_logging_features(self):
        """Show the key features of the logging system."""
        print("\n" + "=" * 60)
        print("🚀 INTELLIGENT TEST FAILURE LOGGING FEATURES")
        print("=" * 60)

        print("\n📋 What gets logged automatically:")
        print("• Test name and location")
        print("• Exception type and message")
        print("• Timestamp for debugging sessions")
        print("• Context-aware debugging hints")

        print("\n🧠 Smart heuristics detect:")
        print("• ImportError → Environment/path issues")
        print("• FileNotFoundError → Missing files/incorrect paths")
        print("• RuntimeError (shape/dimension) → PyTorch tensor mismatches")
        print("• AssertionError in strategies → Algorithmic problems")
        print("• General AssertionErrors → Value comparison issues")

        print("\n📁 Log files are saved to:")
        print("• tests/logs/test_failures_YYYYMMDD_HHMMSS.log")
        print("• Separate log per test session")
        print("• Only created when failures occur (no clutter)")

        print("\n💡 Benefits:")
        print("• Faster debugging with targeted hints")
        print("• Reduces time spent on common issues")
        print("• Helps new developers understand test failures")
        print("• Maintains detailed failure history")

        print("\n🎯 To see it in action:")
        print("1. Uncomment one of the demo methods above")
        print("2. Run: python -m pytest tests/demo/failure_logging_demo.py -v")
        print("3. Check tests/logs/ for the generated failure analysis")

        print("=" * 60)

    def test_mock_data_showcase(self):
        """Show the mock data generation capabilities."""
        from tests.conftest import generate_mock_client_data

        print("\n" + "=" * 60)
        print("🎭 MOCK DATA GENERATION SHOWCASE")
        print("=" * 60)

        # Generate realistic FL client data
        client_results = generate_mock_client_data(num_clients=5, param_shape=(20, 10))

        print(f"\n📊 Generated {len(client_results)} mock clients")

        for i, (client_proxy, fit_res) in enumerate(client_results[:3]):  # Show first 3
            print(f"\n🔍 Client {i}:")
            print(f"   • ID: {client_proxy.cid}")
            print(f"   • Examples: {fit_res.num_examples}")
            print(f"   • Metrics: {fit_res.metrics}")
            param_arrays = parameters_to_ndarrays(fit_res.parameters)
            print(f"   • Parameter shapes: {[p.shape for p in param_arrays]}")

        print("\n💫 Features of mock data:")
        print("• Realistic federated learning structure")
        print("• Varied parameters per client (heterogeneous)")
        print("• Proper FitRes format for framework compatibility")
        print("• Deterministic (seed=42) for reproducible tests")
        print("• Configurable client count and parameter shapes")

        print("=" * 60)

    def test_fixture_ecosystem_demo(self):
        """Show the comprehensive fixture ecosystem available."""
        print("\n" + "=" * 60)
        print("🏗️ TESTING FIXTURE ECOSYSTEM")
        print("=" * 60)

        print("\n📦 Available fixtures in tests/conftest.py:")
        print("• mock_output_directory - Temporary test output dirs")
        print("• mock_strategy_configs - Pre-configured strategy parameters")
        print("• strategy_config - Parameterized testing across strategies")
        print("• dataset_type - Parameterized testing across datasets")
        print("• temp_dataset_dir - Temporary dataset directories")
        print("• mock_client_parameters - Simple mock FL parameters")

        print("\n🎯 Strategy configurations available:")
        strategies = ["trust", "pid", "krum", "multi-krum", "trimmed_mean"]
        for strategy in strategies:
            print(f"• {strategy} - Ready-to-use config with appropriate parameters")

        print("\n📊 Dataset types supported:")
        datasets = ["its", "femnist_iid", "pneumoniamnist", "bloodmnist"]
        for dataset in datasets:
            print(f"• {dataset} - Parameterized testing across dataset types")

        print("\n💡 Usage patterns:")
        print("• Use @pytest.fixture for reusable test data")
        print("• Use @pytest.mark.parametrize for testing variations")
        print("• Combine fixtures for complex test scenarios")
        print("• Mock real components to isolate units under test")

        print("=" * 60)


if __name__ == "__main__":
    # Run the demo as a regular script to see the showcase
    demo = TestFailureLoggingDemo()
    demo.test_show_logging_features()
    demo.test_mock_data_showcase()
    demo.test_fixture_ecosystem_demo()
    print("\n🚀 Run with pytest to see failure logging in action!")
