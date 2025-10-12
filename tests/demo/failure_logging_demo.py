#!/usr/bin/env python3
"""
Test failure logging demo.

Demonstrates pytest hooks for failure analysis.

Usage: python -m pytest tests/demo/failure_logging_demo.py -v
"""

import logging
from unittest.mock import Mock

from flwr.common import parameters_to_ndarrays

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)
from tests.common import generate_mock_client_data

logger = logging.getLogger(__name__)


class TestFailureLoggingDemo:
    """Demo test failures and logging responses."""

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
        logger.info("✅ This test passes - no failure logging")

    def test_import_error_demo(self):
        """Demo ImportError logging."""
        # Uncomment to trigger: from non_existent_module import something
        logger.info("🔍 ImportError demo (commented out)")

    def test_file_not_found_demo(self):
        """Demo FileNotFoundError logging."""
        # Uncomment to trigger: with open("non_existent_file.txt") as f: content = f.read()
        logger.info("🔍 FileNotFoundError demo (commented out)")

    def test_pytorch_shape_error_demo(self):
        """Demo PyTorch shape error logging."""
        # Uncomment to trigger: torch.matmul(torch.randn(5,10), torch.randn(3,8))
        logger.info("🔍 PyTorch shape error demo (commented out)")

    def test_assertion_error_demo(self):
        """Demo AssertionError logging."""
        # Uncomment to trigger: assert 42 == 24, "Expected 42, got 24"
        logger.info("🔍 AssertionError demo (commented out)")

    def test_strategy_assertion_demo(self):
        """Demo strategy-specific assertion logging."""
        # Uncomment to trigger strategy logging heuristics
        logger.info("🔍 Strategy assertion demo (commented out)")

    def test_show_logging_features(self):
        """Show the key features of the logging system."""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 INTELLIGENT TEST FAILURE LOGGING FEATURES")
        logger.info("=" * 60)

        logger.info("\n📋 What gets logged automatically:")
        logger.info("• Test name and location")
        logger.info("• Exception type and message")
        logger.info("• Timestamp for debugging sessions")
        logger.info("• Context-aware debugging hints")

        logger.info("\n🧠 Smart heuristics detect:")
        logger.info("• ImportError → Environment/path issues")
        logger.info("• FileNotFoundError → Missing files/incorrect paths")
        logger.info("• RuntimeError (shape/dimension) → PyTorch tensor mismatches")
        logger.info("• AssertionError in strategies → Algorithmic problems")
        logger.info("• General AssertionErrors → Value comparison issues")

        logger.info("\n📁 Log files are saved to:")
        logger.info("• tests/logs/test_failures_YYYYMMDD_HHMMSS.log")
        logger.info("• Separate log per test session")
        logger.info("• Only created when failures occur (no clutter)")

        logger.info("\n💡 Benefits:")
        logger.info("• Faster debugging with targeted hints")
        logger.info("• Reduces time spent on common issues")
        logger.info("• Helps new developers understand test failures")
        logger.info("• Maintains detailed failure history")

        logger.info("\n🎯 To see it in action:")
        logger.info("1. Uncomment one of the demo methods above")
        logger.info("2. Run: python -m pytest tests/demo/failure_logging_demo.py -v")
        logger.info("3. Check tests/logs/ for the generated failure analysis")

        logger.info("=" * 60)

    def test_mock_data_showcase(self):
        """Show mock data generation."""

        logger.info("\n" + "=" * 50)
        logger.info("🎭 MOCK DATA GENERATION")
        logger.info("=" * 50)

        # Generate realistic FL client data
        client_results = generate_mock_client_data(num_clients=5, param_shape=(20, 10))

        logger.info(f"\n📊 Generated {len(client_results)} mock clients")

        for i, (client_proxy, fit_res) in enumerate(client_results[:3]):  # Show first 3
            logger.info(f"\n🔍 Client {i}:")
            logger.info(f"   • ID: {client_proxy.cid}")
            logger.info(f"   • Examples: {fit_res.num_examples}")
            logger.info(f"   • Metrics: {fit_res.metrics}")
            param_arrays = parameters_to_ndarrays(fit_res.parameters)
            logger.info(f"   • Parameter shapes: {[p.shape for p in param_arrays]}")

        logger.info("\nFeatures:")
        logger.info("• Realistic FL structure")
        logger.info("• Heterogeneous parameters")
        logger.info("• Deterministic (seed=42)")
        logger.info("• Configurable shapes")
        logger.info("=" * 50)

    def test_fixture_ecosystem_demo(self):
        """Show fixture ecosystem."""
        logger.info("\n" + "=" * 50)
        logger.info("🏗️ TESTING FIXTURES")
        logger.info("=" * 50)

        logger.info("\nFixtures:")
        logger.info("• mock_output_directory")
        logger.info("• mock_strategy_configs")
        logger.info("• strategy_config")
        logger.info("• dataset_type")
        logger.info("• temp_dataset_dir")
        logger.info("• mock_client_parameters")

        logger.info("\nStrategy configs:")
        strategies = ["trust", "pid", "krum", "multi-krum", "trimmed_mean"]
        for strategy in strategies:
            logger.info(f"• {strategy}")

        logger.info("\nDataset types:")
        datasets = ["its", "femnist_iid", "pneumoniamnist", "bloodmnist"]
        for dataset in datasets:
            logger.info(f"• {dataset}")

        logger.info("\nUsage:")
        logger.info("• @pytest.fixture for reusable data")
        logger.info("• @pytest.mark.parametrize for variations")
        logger.info("• Mock components for isolation")
        logger.info("=" * 50)


if __name__ == "__main__":
    demo = TestFailureLoggingDemo()
    demo.test_show_logging_features()
    demo.test_mock_data_showcase()
    demo.test_fixture_ecosystem_demo()
    logger.info("\n🚀 Run with pytest to see failure logging in action!")
