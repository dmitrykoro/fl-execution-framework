#!/usr/bin/env python3
"""
Mock data generation demo.

Usage:
  python -m tests.demo.mock_data_showcase
  python -m pytest tests/demo/mock_data_showcase.py -v -s
"""

import logging
import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.simulation_strategies.krum_based_removal_strategy import (
    KrumBasedRemovalStrategy,
)
from src.simulation_strategies.trust_based_removal_strategy import (
    TrustBasedRemovalStrategy,
)
from tests.common import (
    Mock,
    generate_mock_client_data,
    init_demo_output,
    init_test_environment,
    ndarrays_to_parameters,
    np,
    parameters_to_ndarrays,
)

logger = logging.getLogger(__name__)


def demonstrate_basic_mock_data():
    """Show basic mock data generation."""
    logger.info("🎭 BASIC MOCK DATA GENERATION")
    logger.info("=" * 50)

    client_results = generate_mock_client_data(num_clients=5, param_shape=(10, 5))

    logger.info(f"Generated {len(client_results)} clients")
    for i, (client_proxy, fit_res) in enumerate(client_results):
        logger.info(f"Client {i}: {client_proxy.cid} examples={fit_res.num_examples}")
        logger.info(f"  Metrics: {fit_res.metrics}")
        param_arrays = parameters_to_ndarrays(fit_res.parameters)
        param_shapes = [p.shape for p in param_arrays]
        logger.info(f"  Parameter shapes: {param_shapes}")

    logger.info("Mock data matches real FL structure\n")


def demonstrate_heterogeneous_vs_homogeneous():
    """Show different client data distributions."""
    logger.info("🌐 HETEROGENEOUS vs HOMOGENEOUS DATA")
    logger.info("=" * 50)

    client_results = generate_mock_client_data(num_clients=6, param_shape=(20, 10))

    logger.info("Parameter diversity:")
    param_norms = []
    for i, (client_proxy, fit_res) in enumerate(client_results):
        param_arrays = parameters_to_ndarrays(fit_res.parameters)
        first_layer = param_arrays[0]
        norm = np.linalg.norm(first_layer)
        param_norms.append(norm)
        logger.info(f"Client {i}: Parameter norm = {norm:.3f}")

    variance = np.var(param_norms)
    logger.info(f"\nParameter variance: {variance:.3f}")
    if variance > 1.0:
        logger.info("Data is heterogeneous (realistic FL scenario)")
    else:
        logger.info("ℹ️  Data is homogeneous (simplified scenario)")

    logger.info("First 2 clients are similar, others are varied by design\n")


def demonstrate_byzantine_attack_simulation():
    """Show Byzantine attack simulation with mock data."""
    logger.info("🛡️ BYZANTINE ATTACK SIMULATION")
    logger.info("=" * 50)

    # First generate honest clients to determine max magnitude
    honest_clients = generate_mock_client_data(num_clients=7, param_shape=(10, 5))

    # Calculate max honest client magnitude
    max_honest_magnitude = 0.0
    for _, fit_res in honest_clients:
        param_arrays = parameters_to_ndarrays(fit_res.parameters)
        first_layer = param_arrays[0]
        magnitude = np.mean(np.abs(first_layer))
        max_honest_magnitude = max(max_honest_magnitude, magnitude)

    def create_byzantine_client(client_id: str, base_magnitude: float):
        """Create a mock Byzantine client with malicious parameters."""
        client_proxy = Mock()
        client_proxy.cid = client_id

        # Extract numeric part from client_id (e.g., "malicious_0" -> 0)
        numeric_part = int(client_id.split("_")[-1]) if "_" in client_id else 0
        rng = np.random.default_rng(42 + numeric_part)

        # Ensure Byzantine magnitude is significantly higher than honest clients
        attack_intensity = base_magnitude * (2.0 + numeric_part)  # 2x to 3x multiplier
        malicious_params = [
            rng.standard_normal((10, 5)) * attack_intensity,
            rng.standard_normal(5) * attack_intensity,
        ]

        fit_res = Mock()
        fit_res.parameters = ndarrays_to_parameters(malicious_params)
        fit_res.num_examples = 100
        fit_res.metrics = {"accuracy": 0.1, "loss": 10.0}

        return (client_proxy, fit_res)

    byzantine_clients = [
        create_byzantine_client("malicious_0", float(max_honest_magnitude)),
        create_byzantine_client("malicious_1", float(max_honest_magnitude)),
    ]

    all_clients = honest_clients + byzantine_clients

    logger.info(
        f"Created {len(honest_clients)} honest + {len(byzantine_clients)} Byzantine clients"
    )

    logger.info("\nParameter magnitude analysis:")
    honest_magnitudes = []
    byzantine_magnitudes = []

    for i, (client_proxy, fit_res) in enumerate(all_clients):
        param_arrays = parameters_to_ndarrays(fit_res.parameters)
        first_layer = param_arrays[0]
        magnitude = np.mean(np.abs(first_layer))
        client_type = "Byzantine" if "malicious" in client_proxy.cid else "Honest"
        logger.info(
            f"{client_type:>9} Client {client_proxy.cid}: Avg magnitude = {magnitude:.3f}"
        )

        if "malicious" in client_proxy.cid:
            byzantine_magnitudes.append(magnitude)
        else:
            honest_magnitudes.append(magnitude)

    # Dynamic comparison message
    avg_honest = np.mean(honest_magnitudes)
    avg_byzantine = np.mean(byzantine_magnitudes)
    ratio = avg_byzantine / avg_honest if avg_honest > 0 else float("inf")

    logger.info(f"\nAvg honest magnitude: {avg_honest:.3f}")
    logger.info(f"Avg Byzantine magnitude: {avg_byzantine:.3f}")
    logger.info(f"Byzantine clients have {ratio:.1f}x higher parameter magnitudes\n")


def demonstrate_performance_benefits():
    """Show performance benefits of mock data."""
    logger.info("PERFORMANCE BENEFITS")
    logger.info("=" * 50)

    start_time = time.time()
    mock_clients = generate_mock_client_data(num_clients=100, param_shape=(1000, 500))
    mock_time = time.time() - start_time

    total_params = 0
    for _, fit_res in mock_clients:
        param_arrays = parameters_to_ndarrays(fit_res.parameters)
        for param in param_arrays:
            total_params += param.size

    logger.info("Mock data generation:")
    logger.info("   • 100 clients with 1000x500 parameters")
    logger.info(f"   • Total parameters: {total_params:,}")
    logger.info(f"   • Generation time: {mock_time:.4f} seconds")
    logger.info("   • Memory efficient: Generated on-demand")
    logger.info("\nReal data loading typically:")
    logger.info("   • Take 10-30 seconds to load from disk")
    logger.info("   • Require gigabytes of storage")
    logger.info("   • Need network downloads for public datasets")
    logger.info("   • Have inconsistent formats across datasets")

    speedup = 20 / mock_time if mock_time > 0 else float("inf")
    logger.info(f"\nMock data is ~{speedup:.0f}x faster than real data loading")
    logger.info("Suitable for development and CI/CD\n")


def demonstrate_strategy_testing():
    """Show mock data with strategy testing."""
    logger.info("🧪 STRATEGY TESTING WITH MOCK DATA")
    logger.info("=" * 50)

    mock_clients = generate_mock_client_data(num_clients=10, param_shape=(50, 25))
    mock_strategy_history = Mock(spec=SimulationStrategyHistory)
    strategies = [
        (
            "Trust-based",
            TrustBasedRemovalStrategy(
                trust_threshold=0.7,
                remove_clients=True,
                beta_value=0.5,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
            ),
        ),
        (
            "Krum",
            KrumBasedRemovalStrategy(
                remove_clients=True,
                num_malicious_clients=2,
                num_krum_selections=5,
                begin_removing_from_round=2,
                strategy_history=mock_strategy_history,
            ),
        ),
    ]

    logger.info("Testing strategies with identical mock data:")
    for name, strategy in strategies:
        try:
            start_time = time.time()
            result = strategy.aggregate_fit(
                server_round=1, results=mock_clients, failures=[]
            )
            test_time = time.time() - start_time

            if result:
                params, metrics = result
                logger.info(
                    f"✅ {name:>12}: {test_time:.4f}s, params={len(params)}, metrics={len(metrics)}"
                )
            else:
                logger.info(f"⚠️  {name:>12}: No result returned")

        except Exception as e:
            logger.info(f"❌ {name:>12}: Failed - {str(e)[:50]}...")

    logger.info("\nSame mock data tests multiple strategies consistently")
    logger.info("Enables comparative analysis and regression testing\n")


def demonstrate_edge_case_testing():
    """Show edge case testing with mock data."""
    logger.info("🔍 EDGE CASE TESTING")
    logger.info("=" * 50)

    edge_cases = [
        ("Empty clients", []),
        ("Single client", generate_mock_client_data(1, (5, 3))),
        ("Two identical", generate_mock_client_data(2, (5, 3))[:1] * 2),  # Duplicate
        ("Large parameters", generate_mock_client_data(3, (500, 200))),
        ("Tiny parameters", generate_mock_client_data(3, (2, 1))),
    ]

    mock_strategy_history = Mock(spec=SimulationStrategyHistory)
    strategy = TrustBasedRemovalStrategy(
        trust_threshold=0.5,
        remove_clients=True,
        beta_value=0.5,
        begin_removing_from_round=2,
        strategy_history=mock_strategy_history,
    )

    for case_name, mock_data in edge_cases:
        try:
            result = strategy.aggregate_fit(
                server_round=1, results=mock_data, failures=[]
            )
            status = "✅ Handled" if result else "⚠️  No result"
            logger.info(f"{status} {case_name:>15}: {len(mock_data)} clients")
        except Exception as e:
            logger.info(f"❌ Failed {case_name:>15}: {str(e)[:40]}...")

    logger.info("\nMock data makes edge case testing fast and systematic\n")


class TestMockDataShowcase:
    """Test class version for running with pytest."""

    def test_run_all_demonstrations(self):
        """Run all demonstrations as a test."""
        init_demo_output()
        logger.info("\n" + "=" * 60)
        logger.info("🎭 COMPREHENSIVE MOCK DATA SHOWCASE")
        logger.info("=" * 60)

        demonstrate_basic_mock_data()
        demonstrate_heterogeneous_vs_homogeneous()
        demonstrate_byzantine_attack_simulation()
        demonstrate_performance_benefits()
        demonstrate_strategy_testing()
        demonstrate_edge_case_testing()

        logger.info("MOCK DATA BENEFITS:")
        logger.info("• 20x faster than real data")
        logger.info("• Deterministic and reproducible")
        logger.info("• Supports heterogeneous FL scenarios")
        logger.info("• Enables Byzantine attack testing")
        logger.info("• Perfect for edge case coverage")
        logger.info("• Consistent across all strategies")
        logger.info("• No storage requirements")
        logger.info("• Highly configurable")
        logger.info("=" * 50)


if __name__ == "__main__":
    try:
        logger = init_test_environment(include_timestamp=True)
        logger.info("🎭 Mock Data Showcase initialized successfully")
        showcase = TestMockDataShowcase()
        showcase.test_run_all_demonstrations()
        logger.info("✅ Mock Data Showcase completed successfully")
    except Exception as e:
        import logging

        logging.error(f"❌ Error running showcase: {e}")
        import traceback

        traceback.print_exc()
