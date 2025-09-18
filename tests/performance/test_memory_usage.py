"""
Memory usage tests for federated learning simulation framework.

Tests memory consumption monitoring, leak detection, and resource cleanup
during simulation execution.
"""

import gc
import os
from typing import Any, Generator, List, Optional, Tuple
from unittest.mock import Mock, patch

import numpy as np
import psutil
import pytest
from src.data_models.client_info import ClientInfo
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory

from tests.fixtures.mock_datasets import (
    MockFederatedDataset,
    generate_mock_client_parameters,
)
from tests.fixtures.sample_models import create_mock_client_models


class MemoryMonitor:
    """Monitor memory usage during tests."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.measurements: List[Tuple[str, float]] = []

    def get_memory_usage(self) -> float:
        """Return current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def record_measurement(self, label: str = "") -> float:
        """Record memory measurement with optional label."""
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        self.measurements.append((label, current_memory))
        return current_memory

    def get_memory_increase(self) -> float:
        """Return memory increase since initialization."""
        return self.get_memory_usage() - self.initial_memory

    def get_peak_increase(self) -> float:
        """Return peak memory increase since initialization."""
        return self.peak_memory - self.initial_memory


@pytest.fixture
def memory_monitor() -> Generator[MemoryMonitor, None, None]:
    """Provide memory monitoring capabilities."""
    monitor = MemoryMonitor()
    yield monitor
    # Force garbage collection after test
    gc.collect()


class TestMemoryUsageMonitoring:
    """Test memory consumption monitoring."""

    def test_strategy_config_memory_usage(self, memory_monitor: MemoryMonitor) -> None:
        """Test memory usage creating multiple StrategyConfig objects."""
        memory_monitor.record_measurement("before_config_creation")

        # Create multiple strategy configurations
        configs: List[StrategyConfig] = []
        for i in range(1000):
            config_data = {
                "aggregation_strategy_keyword": "trust",
                "num_of_rounds": 5,
                "num_of_clients": 10,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            }
            configs.append(StrategyConfig.from_dict(config_data))

        memory_monitor.record_measurement("after_config_creation")

        # Memory increase should be reasonable (less than 100MB for 1000 configs)
        memory_increase = memory_monitor.get_memory_increase()
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f}MB"

        # Clean up
        del configs
        gc.collect()

        memory_monitor.record_measurement("after_cleanup")

        # Memory should be mostly reclaimed (within 20MB of initial)
        final_increase = memory_monitor.get_memory_increase()
        assert (
            final_increase < 20
        ), f"Memory not properly cleaned up: {final_increase:.2f}MB"

    def test_client_info_memory_scaling(self, memory_monitor: MemoryMonitor) -> None:
        """Test memory usage scaling with clients and rounds."""
        memory_monitor.record_measurement("initial")

        # Test with increasing numbers of clients
        client_counts: List[int] = [10, 50, 100, 500]
        memory_measurements: List[Tuple[int, float]] = []

        for num_clients in client_counts:
            # Create client info objects
            clients: List[ClientInfo] = []
            for client_id in range(num_clients):
                client_info = ClientInfo(client_id=client_id, num_of_rounds=10)

                # Add some history data
                for round_num in range(10):
                    client_info.add_history_entry(
                        current_round=round_num + 1,  # Rounds are 1-indexed
                        loss=np.random.random(),
                        accuracy=np.random.random(),
                        removal_criterion=np.random.random(),
                    )
                clients.append(client_info)

            memory_usage = memory_monitor.record_measurement(f"clients_{num_clients}")
            memory_measurements.append((num_clients, memory_usage))

            # Clean up for next iteration
            del clients
            gc.collect()

        # Memory usage should scale roughly linearly with client count
        # Check that memory doesn't grow exponentially
        for i in range(1, len(memory_measurements)):
            prev_clients, prev_memory = memory_measurements[i - 1]
            curr_clients, curr_memory = memory_measurements[i]

            client_ratio = curr_clients / prev_clients
            memory_ratio = curr_memory / prev_memory if prev_memory > 0 else 1

            # Memory growth should not exceed 2x the client ratio
            assert (
                memory_ratio <= client_ratio * 2
            ), f"Memory scaling issue: {client_ratio}x clients led to {memory_ratio}x memory"

    def test_simulation_history_memory_usage(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test simulation history memory usage."""
        memory_monitor.record_measurement("before_history")

        # Create a mock strategy config for the history
        config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "trust",
                "num_of_rounds": 100,
                "num_of_clients": 50,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            }
        )

        # Create mock dataset handler
        from tests.fixtures.mock_datasets import MockDatasetHandler

        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=50)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )

        # Simulate 100 rounds with client data
        for round_num in range(100):
            # Add round history entry
            history.insert_round_history_entry(
                score_calculation_time_nanos=int(np.random.random() * 1000000),
                removal_threshold=np.random.random(),
                loss_aggregated=np.random.random(),
            )

            # Add client data for this round
            for client_id in range(50):
                history.insert_single_client_history_entry(
                    client_id=client_id,
                    current_round=round_num + 1,  # Rounds are 1-indexed
                    loss=np.random.random(),
                    accuracy=np.random.random(),
                    removal_criterion=np.random.random(),
                )

        memory_monitor.record_measurement("after_history_creation")

        # Memory increase should be reasonable (less than 100MB for this amount of data)
        memory_increase = memory_monitor.get_memory_increase()
        assert (
            memory_increase < 100
        ), f"History memory usage too high: {memory_increase:.2f}MB"

        # Test memory usage when accessing history data
        all_clients: List[ClientInfo] = history.get_all_clients()
        for client_info in all_clients:
            _ = client_info.get_metric_by_name("loss_history")
            _ = client_info.get_metric_by_name("accuracy_history")

        memory_monitor.record_measurement("after_history_access")

        # Memory shouldn't increase significantly from just accessing data
        access_increase = memory_monitor.get_memory_increase()
        assert (
            access_increase < memory_increase + 20
        ), f"Memory increased too much during access: {access_increase:.2f}MB"


class TestMemoryLeakDetection:
    """Test memory leak detection."""

    def test_repeated_strategy_execution_no_leaks(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test repeated strategy execution for memory leaks."""
        memory_monitor.record_measurement("initial")

        # Simulate multiple strategy executions
        for iteration in range(10):
            # Create strategy configuration
            config_data = {
                "aggregation_strategy_keyword": "trust",
                "num_of_rounds": 3,
                "num_of_clients": 10,
                "trust_threshold": 0.7,
                "beta_value": 0.5,
            }
            config = StrategyConfig.from_dict(config_data)

            # Create client data
            client_params: np.ndarray = generate_mock_client_parameters(10, 1000)

            # Simulate strategy processing (mock the actual strategy execution)
            with patch(
                "src.simulation_strategies.trust_based_removal_strategy.TrustBasedRemovalStrategy"
            ) as mock_strategy:
                mock_instance = Mock()
                mock_instance.aggregate_parameters.return_value = np.random.randn(1000)
                mock_strategy.return_value = mock_instance

                # Process parameters
                _ = mock_instance.aggregate_parameters(client_params)

            # Force cleanup
            del config, client_params
            gc.collect()

            memory_monitor.record_measurement(f"iteration_{iteration}")

        # Memory should not continuously increase
        final_memory = memory_monitor.get_memory_increase()
        assert (
            final_memory < 150
        ), f"Potential memory leak detected: {final_memory:.2f}MB increase"

    def test_client_creation_cleanup_cycle(self, memory_monitor: MemoryMonitor) -> None:
        """Test memory cleanup in client creation/destruction cycles."""
        memory_monitor.record_measurement("initial")

        baseline_memory: Optional[float] = None

        # Run multiple cycles of client creation and cleanup
        for cycle in range(5):
            # Create mock clients
            clients: List[Any] = create_mock_client_models(
                num_clients=20, dataset_type="its"
            )

            # Simulate some client operations
            for client in clients:
                # Mock training data
                mock_data: np.ndarray = np.random.randn(100, 3, 32, 32)

                # Simulate fit operation
                client.fit(mock_data, {"epochs": 1})

                # Simulate evaluate operation
                client.evaluate(mock_data, {"batch_size": 32})

            memory_monitor.record_measurement(f"cycle_{cycle}_created")

            # Clean up clients
            del clients
            gc.collect()

            memory_after_cleanup: float = memory_monitor.record_measurement(
                f"cycle_{cycle}_cleaned"
            )

            # After first cycle, establish baseline
            if cycle == 0:
                baseline_memory = memory_after_cleanup
            else:
                # Memory should not continuously grow between cycles
                memory_growth: float = memory_after_cleanup - (baseline_memory or 0.0)
                assert (
                    memory_growth < 20
                ), f"Memory leak in cycle {cycle}: {memory_growth:.2f}MB growth from baseline"

    def test_dataset_loading_memory_cleanup(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory cleanup when loading/unloading datasets."""
        memory_monitor.record_measurement("initial")

        # Test multiple dataset loading cycles
        for cycle in range(3):
            # Create federated dataset (reduced size to prevent excessive memory usage)
            fed_dataset = MockFederatedDataset(
                num_clients=10,
                samples_per_client=100,
                input_shape=(3, 64, 64),  # Smaller images to test memory usage
            )

            memory_monitor.record_measurement(f"cycle_{cycle}_dataset_loaded")

            # Access all client datasets
            for client_id in range(10):
                client_dataset = fed_dataset.get_client_dataset(client_id)
                # Simulate data access
                for i in range(0, len(client_dataset), 100):
                    _ = client_dataset[i]

            memory_monitor.record_measurement(f"cycle_{cycle}_data_accessed")

            # Clean up dataset
            del fed_dataset
            gc.collect()

            memory_monitor.record_measurement(f"cycle_{cycle}_cleaned")

        # Final memory should not be significantly higher than initial
        final_increase = memory_monitor.get_memory_increase()
        assert (
            final_increase < 200
        ), f"Dataset memory not properly cleaned: {final_increase:.2f}MB"


class TestResourceCleanup:
    """Test resource cleanup."""

    def test_simulation_component_cleanup(self, memory_monitor: MemoryMonitor) -> None:
        """Test simulation component cleanup."""
        memory_monitor.record_measurement("initial")

        # Create simulation components
        config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "pid",
                "num_of_rounds": 5,
                "num_of_clients": 15,
                "Kp": 1.0,
                "Ki": 0.1,
                "Kd": 0.01,
            }
        )

        # Create mock dataset handler
        from tests.fixtures.mock_datasets import MockDatasetHandler

        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=15)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )
        clients: List[Any] = create_mock_client_models(
            num_clients=15, dataset_type="femnist_iid"
        )

        memory_monitor.record_measurement("components_created")

        # Create mock dataset handler
        from tests.fixtures.mock_datasets import MockDatasetHandler

        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=15)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )

        # Simulate rounds of training
        for round_num in range(5):
            # Add round history entry
            history.insert_round_history_entry(
                score_calculation_time_nanos=int(np.random.random() * 1000000),
                removal_threshold=np.random.random(),
                loss_aggregated=np.random.random(),
            )

            for client_id, client in enumerate(clients):
                # Mock client training
                mock_params: np.ndarray = np.random.randn(1000)
                client.fit(mock_params, {"epochs": 1})

                # Add client history entry
                history.insert_single_client_history_entry(
                    client_id=client_id,
                    current_round=round_num + 1,  # Rounds are 1-indexed
                    loss=np.random.random(),
                    accuracy=np.random.random(),
                    removal_criterion=np.random.random(),
                )

        memory_monitor.record_measurement("simulation_complete")

        # Clean up components in proper order
        del clients
        gc.collect()
        memory_monitor.record_measurement("clients_cleaned")

        del history
        gc.collect()
        memory_monitor.record_measurement("history_cleaned")

        del config
        gc.collect()
        memory_monitor.record_measurement("config_cleaned")

        # Memory should be mostly reclaimed
        final_increase = memory_monitor.get_memory_increase()
        assert (
            final_increase < 30
        ), f"Components not properly cleaned: {final_increase:.2f}MB"

    def test_large_parameter_handling(self, memory_monitor: MemoryMonitor) -> None:
        """Test large parameter array handling."""
        memory_monitor.record_measurement("initial")

        # Create large parameter arrays (simulating large neural networks)
        large_param_size: int = 1_000_000  # 1M parameters
        num_clients: int = 10

        # Generate parameters
        client_params: List[np.ndarray] = []
        for client_id in range(num_clients):
            params: np.ndarray = np.random.randn(large_param_size).astype(np.float32)
            client_params.append(params)

        memory_monitor.record_measurement("large_params_created")

        # Simulate parameter aggregation
        aggregated_params: np.ndarray = np.mean(client_params, axis=0)

        memory_monitor.record_measurement("params_aggregated")

        # Memory usage should be reasonable (less than 200MB for this test)
        memory_increase = memory_monitor.get_memory_increase()
        assert (
            memory_increase < 200
        ), f"Large parameter memory usage too high: {memory_increase:.2f}MB"

        # Clean up parameters
        del client_params, aggregated_params
        gc.collect()

        memory_monitor.record_measurement("large_params_cleaned")

        # Memory should be reclaimed
        final_increase = memory_monitor.get_memory_increase()
        assert (
            final_increase < 50
        ), f"Large parameters not properly cleaned: {final_increase:.2f}MB"

    def test_concurrent_client_memory_usage(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory usage with concurrent client handling."""
        memory_monitor.record_measurement("initial")

        # Create multiple client groups to simulate concurrent processing
        client_groups: List[List[Any]] = []

        for group_id in range(5):
            group_clients: List[Any] = create_mock_client_models(
                num_clients=10, dataset_type="its"
            )

            # Simulate concurrent client operations
            for client in group_clients:
                # Mock training with different data sizes
                data_size: int = 500 + group_id * 100
                mock_data: np.ndarray = np.random.randn(data_size, 3, 32, 32)
                client.fit(mock_data, {"epochs": 1})

            client_groups.append(group_clients)
            memory_monitor.record_measurement(f"group_{group_id}_processed")

        # Memory should scale reasonably with number of client groups
        peak_increase = memory_monitor.get_peak_increase()
        assert (
            peak_increase < 300
        ), f"Concurrent client memory usage too high: {peak_increase:.2f}MB"

        # Clean up groups one by one
        for group_id, group in enumerate(client_groups):
            del group
            gc.collect()
            memory_monitor.record_measurement(f"group_{group_id}_cleaned")

        # Final cleanup
        del client_groups
        gc.collect()

        final_increase = memory_monitor.get_memory_increase()
        assert (
            final_increase < 50
        ), f"Concurrent clients not properly cleaned: {final_increase:.2f}MB"


@pytest.mark.slow
class TestLongRunningMemoryBehavior:
    """Test long-running memory behavior."""

    def test_extended_simulation_memory_stability(
        self, memory_monitor: MemoryMonitor
    ) -> None:
        """Test memory stability during extended simulation runs."""
        memory_monitor.record_measurement("initial")

        # Simulate a long-running simulation with many rounds
        config = StrategyConfig.from_dict(
            {
                "aggregation_strategy_keyword": "krum",
                "num_of_rounds": 50,
                "num_of_clients": 20,
                "num_krum_selections": 15,
            }
        )

        # Create mock dataset handler
        from tests.fixtures.mock_datasets import MockDatasetHandler

        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=20)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )
        memory_samples: List[float] = []

        # Create mock dataset handler
        from tests.fixtures.mock_datasets import MockDatasetHandler

        dataset_handler = MockDatasetHandler()
        dataset_handler.setup_dataset(num_clients=20)

        # Create history with proper initialization
        history = SimulationStrategyHistory(
            strategy_config=config,
            dataset_handler=dataset_handler,
            rounds_history=None,  # Will be created in __post_init__
        )

        # Run simulation for many rounds
        for round_num in range(50):
            # Add round history entry
            history.insert_round_history_entry(
                score_calculation_time_nanos=int(np.random.random() * 1000000),
                removal_threshold=np.random.random(),
                loss_aggregated=np.random.random(),
            )

            # Create clients for this round
            round_clients: List[Any] = create_mock_client_models(
                num_clients=20, dataset_type="pneumoniamnist"
            )

            for client_id, client in enumerate(round_clients):
                # Simulate client training
                mock_data: np.ndarray = np.random.randn(200, 1, 28, 28)
                client.fit(mock_data, {"epochs": 1})

                # Add client history entry
                history.insert_single_client_history_entry(
                    client_id=client_id,
                    current_round=round_num + 1,  # Rounds are 1-indexed
                    loss=np.random.random(),
                    accuracy=np.random.random(),
                    removal_criterion=np.random.random(),
                )

            # Clean up round clients
            del round_clients
            gc.collect()

            # Sample memory every 10 rounds
            if round_num % 10 == 0:
                current_memory: float = memory_monitor.record_measurement(
                    f"round_{round_num}"
                )
                memory_samples.append(current_memory)

        # Memory should not continuously increase
        if len(memory_samples) > 1:
            # Check that memory doesn't grow more than 50MB over the simulation
            memory_growth: float = memory_samples[-1] - memory_samples[0]
            assert (
                memory_growth < 50
            ), f"Memory continuously growing in long simulation: {memory_growth:.2f}MB"

        # Clean up
        del history, config
        gc.collect()

        final_increase = memory_monitor.get_memory_increase()
        assert (
            final_increase < 100
        ), f"Long simulation memory not cleaned: {final_increase:.2f}MB"
